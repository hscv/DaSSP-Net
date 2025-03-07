import os
import os.path
import torch
import numpy as np
import pandas
from glob import glob
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import cv2
import numpy as np

class HSI3D(BaseVideoDataset):
    """ HSI dataset.

    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        super().__init__('HSI3D', root, image_loader)

        self.RGBHDataRootDir = root
        self.cur_modalName = split # 'VIS' 'NIR' 'RedNIR'
        self.sequence_list = os.listdir(self.RGBHDataRootDir)
        self.sequence_list .sort()
        self.all_img_info_dic = self.getAllImageInfo()
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def getImgs(self, videoname, suffix='.png'):
        imgTmpLis = os.listdir(videoname)
        imgLis = []
        for imgTmp in imgTmpLis:
            if imgTmp.find(suffix) != -1:
                imgLis.append(imgTmp)
        imgLis.sort()
        return imgLis

    def getGTBox(self, gt_filename):
        f = open(gt_filename,'r')
        gt_arr = []
        gt_res = f.readlines()
        for gt in gt_res:
            gt = gt.strip()
            kk = gt.split('\t')
            x = list(map(int, kk))
            gt_arr.append(x)
        return np.array(gt_arr)

    def getAllImageInfo(self):
        all_img_info_dic = {}
        for videoname in self.sequence_list:
            hsi_imgs = self.getImgs(os.path.join(self.RGBHDataRootDir, videoname, 'img'), '.png')
            gt_arr_hsi = self.getGTBox(os.path.join(self.RGBHDataRootDir, videoname, 'groundtruth_rect.txt'))
            hsi_fc_imgs = self.getImgs(os.path.join(self.RGBHDataRootDir.replace('HSI-'+self.cur_modalName, 'HSI-'+self.cur_modalName+'-FalseColor'), videoname, 'img'), '.jpg')
            assert len(hsi_imgs) == len(gt_arr_hsi) and len(hsi_imgs) == len(hsi_fc_imgs) 

            all_img_info_dic[videoname] = {}
            all_img_info_dic[videoname]["anno-hsi"] = []
            all_img_info_dic[videoname]["hsi"] = []
            all_img_info_dic[videoname]["hsi_fc"] = []
            for i in range(len(hsi_imgs)):
                all_img_info_dic[videoname]["anno-hsi"].append(gt_arr_hsi[i])
                all_img_info_dic[videoname]["hsi"].append(hsi_imgs[i])
                all_img_info_dic[videoname]["hsi_fc"].append(hsi_fc_imgs[i])
        return all_img_info_dic

    def get_name(self):
        return 'hsi3d'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        try:
            gt = np.loadtxt(bb_anno_file, delimiter='\t', dtype=np.float32)
        except:
            gt = np.loadtxt(bb_anno_file, dtype=np.float32)

        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        seq_name_rgb = self.sequence_list[seq_id]
        seq_path_rgb = os.path.join(self.root, seq_name_rgb)
        return seq_path_rgb

    def _get_anno(self, seq_id):
        bbox = self.all_img_info_dic[self.sequence_list[seq_id]]['anno-hsi']
        return bbox

    def get_sequence_info(self, seq_id):
        bbox = self._get_anno(seq_id)
        bbox = torch.Tensor(bbox)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_rgb_hsi_frame_path(self, seq_path, frame_id):
        seq_paths = sorted(glob(os.path.join(seq_path, '*.jpg')))
        seq_paths_rgb = seq_paths[frame_id]
        seq_paths_hsi = seq_paths_rgb.replace("FalseColor", "3D").replace("jpg", "npy")
        return (seq_paths_rgb, seq_paths_hsi)


    def X2Cube(self, img, cellSize=4):

        B = [cellSize, cellSize]
        skip = [cellSize, cellSize]
        # Parameters
        M, N = img.shape
        col_extent = N - B[1] + 1
        row_extent = M - B[0] + 1

        # Get Starting block indices
        start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])

        # Generate Depth indeces
        didx = M * N * np.arange(1)
        start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))

        # Get offsetted indices across the height and width of input array
        offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)

        # Get all actual indices & index into input array for final output
        out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
        out = np.transpose(out)
        DataCube = out.reshape(M//cellSize, N//cellSize, cellSize*cellSize)

        hsi_min = np.min(DataCube, axis=(0,1), keepdims=True)
        hsi_max = np.max(DataCube, axis=(0,1), keepdims=True)
        hsi_normed = (DataCube - hsi_min) / ((hsi_max - hsi_min) + 1e-6) * 255
        hsi_normed = hsi_normed.astype(np.uint8)
        return hsi_normed


    def get_image_loader(self, img_file: str, cellSize=-1) -> np.array:
        if cellSize > 0:
            img = cv2.imread(img_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img = self.X2Cube(img, cellSize)
        elif cellSize == -1:
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        else:
            raise Exception
        return img


    def _get_frames(self, seq_id, frame_ids):
        frame_list = []
        for frame_id in frame_ids:
            frame_name = '%04d.png' % (frame_id+1)
            image_path = os.path.join(self.RGBHDataRootDir, self.sequence_list[seq_id], 'img', frame_name)
            if self.cur_modalName == 'NIR':
                hsi_image = self.get_image_loader(image_path, cellSize=5)
            elif self.cur_modalName == 'VIS':
                hsi_image = self.get_image_loader(image_path, cellSize=4)
            elif self.cur_modalName == 'RedNIR':
                hsi_image = self.get_image_loader(image_path, cellSize=4)
                hsi_image = hsi_image[:,:,:-1]

            frame_name = '%04d.jpg' % (frame_id+1)

            image_path = os.path.join(self.RGBHDataRootDir.replace('HSI-'+self.cur_modalName, 'HSI-'+self.cur_modalName+'-FalseColor'), self.sequence_list[seq_id], 'img', frame_name)
            hsi_fc_image = self.get_image_loader(image_path, cellSize=-1)
            frame_list.append(np.concatenate((hsi_fc_image, hsi_image), axis=2))

        return frame_list 


    def get_frames(self, seq_id, frame_ids, anno=None):
        frame_list = self._get_frames(seq_id, frame_ids)

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if key == 'seq_belong_mask':
                continue
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

