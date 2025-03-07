import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.vipt import ViPTTrack
import lib.test.parameter.vipt_hsi as rgbt_params
import multiprocessing
import torch
import time
from glob import glob

def X2Cube(img, B=[4, 4], skip = [4, 4], bandNumber=16):
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
    DataCube = out.reshape(M//B[0], N//B[1], bandNumber)

    hsi_min = np.min(DataCube, axis=(0,1), keepdims=True)
    hsi_max = np.max(DataCube, axis=(0,1), keepdims=True)
    hsi_normed = (DataCube - hsi_min) / ((hsi_max - hsi_min) + 1e-6) * 255
    hsi_normed = hsi_normed.astype(np.uint8)
    return hsi_normed

def getGTBbox(gt_filename):
    f = open(gt_filename, 'r')
    arr = f.readlines()
    gt = []
    for dd in arr:
        dd = dd.strip()
        kk = dd.split('\t')
        gt.append(list(map(int, kk)))
    return gt

def genConfig(seq_path, set_type=None, train_data_type=''):
    HSI_img_list = sorted(glob(join(seq_path, 'img', '*.png')))
    if train_data_type == 'vis':
        RGB_img_list = sorted(glob(join(seq_path.replace('HSI-VIS','HSI-VIS-FalseColor'), 'img', '*.jpg')))
    elif train_data_type == 'nir':
        RGB_img_list = sorted(glob(join(seq_path.replace('HSI-NIR','HSI-NIR-FalseColor'), 'img', '*.jpg')))
    elif train_data_type == 'rednir':
        RGB_img_list = sorted(glob(join(seq_path.replace('HSI-RedNIR','HSI-RedNIR-FalseColor'), 'img', '*.jpg')))
    else:
        raise Exception
    RGB_gt = getGTBbox(join(seq_path, 'groundtruth_rect.txt'))
    assert len(HSI_img_list) == len(RGB_img_list)
    assert len(RGB_img_list) == len(RGB_gt)
    return RGB_img_list, RGB_gt, HSI_img_list



def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=2, debug=0, epoch=60, model_path='', mode_type=''):
    train_data_type=mode_type 
    if mode_type == 'vis':
        cellSize = 4
    elif mode_type == 'nir':
        cellSize = 5
    elif mode_type == 'rednir':
        cellSize = 4
    else:
        raise Exception

    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    params = rgbt_params.parameters(yaml_name, epoch)
    params.checkpoint = model_path
    vipt = ViPTTrack(params) 
    tracker = OSTrack_HOT(tracker=vipt)
    seq_path = seq_home + '/' + seq_name
    print ('seq_path = ', seq_path)
    
    print('——————————Process sequence: '+ seq_name +'——————————————')
    RGB_img_list, RGB_gt, HSI_img_list = genConfig(seq_path, dataset_name, train_data_type)

    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
        result.dtype = np.float64
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=np.float64)
    result[0] = np.copy(RGB_gt[0])
    for frame_idx, rgb_img in enumerate(RGB_img_list):
        image = cv2.cvtColor(cv2.imread(rgb_img), cv2.COLOR_BGR2RGB)
        hsi_img = cv2.imread(HSI_img_list[frame_idx], -1)
        hsi_img_norm = X2Cube(hsi_img, B=[cellSize,cellSize], skip=[cellSize,cellSize], bandNumber=cellSize*cellSize)
        if train_data_type == 'rednir':
            hsi_img_norm = hsi_img_norm[:,:,:-1]

        frame = np.concatenate((image, hsi_img_norm), axis=2)
        if frame_idx == 0:
            # initialization
            print ('cnt = ', frame_idx+1, ' , bbox = ', np.array(RGB_gt[0]), ' gt_arr[cnt] = ', RGB_gt[0])
            tracker.initialize(frame, RGB_gt[0])
        elif frame_idx > 0:
            # track
            region = tracker.track(frame, train_data_type)
            print ('cnt = ', frame_idx+1, ' , bbox = ', np.array(region), ' gt_arr[cnt] = ', RGB_gt[frame_idx])
            result[frame_idx] = np.array(region)

    return RGB_gt, result

class OSTrack_HOT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize STARK for specific video'''
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB, train_data_type):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB, train_data_type=train_data_type)
        pred_bbox = outputs['target_bbox']
        # pred_score = outputs['best_score']
        return pred_bbox



def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def cal_iou(box1, box2):
    r"""

    :param box1: x1,y1,w,h
    :param box2: x1,y1,w,h
    :return: iou
    """
    x11 = box1[0]
    y11 = box1[1]
    x21 = box1[0] + box1[2] - 1
    y21 = box1[1] + box1[3] - 1
    area_1 = (x21 - x11 + 1) * (y21 - y11 + 1)

    x12 = box2[0]
    y12 = box2[1]
    x22 = box2[0] + box2[2] - 1
    y22 = box2[1] + box2[3] - 1
    area_2 = (x22 - x12 + 1) * (y22 - y12 + 1)

    x_left = max(x11, x12)
    x_right = min(x21, x22)
    y_top = max(y11, y12)
    y_down = min(y21, y22)

    inter_area = max(x_right - x_left + 1, 0) * max(y_down - y_top + 1, 0)
    iou = inter_area / (area_1 + area_2 - inter_area)
    return iou


def cal_success(iou):
    success_all = []
    overlap_thresholds = np.arange(0, 1.05, 0.05)
    for overlap_threshold in overlap_thresholds:
        success = sum(np.array(iou) > overlap_threshold) / len(iou)
        success_all.append(success)
    return np.array(success_all)

def calAUC(gtArr,resArr,video_dir_arr,lent):
    # ------------ starting evaluation  -----------
    success_all_video = []
    for idx in range(lent):
        result_boxes = resArr[idx]
        result_boxes_gt = gtArr[idx]
        result_boxes_gt = [np.array(box) for box in result_boxes_gt]
        iou = list(map(cal_iou, result_boxes, result_boxes_gt))
        success = cal_success(iou)
        auc = np.mean(success)
        success_all_video.append(success)
        print ('video = ',video_dir_arr[idx],' , auc = ',auc)
    print('np.mean(success_all_video) = ', np.mean(success_all_video))

def save_txt_results(video_name, detRes, epoch_type, mode_type):
    dstRootDir = '../results/'+epoch_type
    if not os.path.exists(dstRootDir):
        os.makedirs(dstRootDir)
    savename = mode_type+'-'+video_name+'.txt'

    f = open(os.path.join(dstRootDir, savename), 'w')

    for data in detRes:
        for dd in data:
            f.write(str(dd))
            f.write('\t')
        f.write('\n')
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBE dataset.')
    parser.add_argument('--yaml_name', type=str, default='deep_all', help='Name of tracking method.')
    parser.add_argument('--dataset_name', type=str, default='HOT23VAL', help='Name of dataset (HOT23VAL, HOT23TEST).')
    parser.add_argument('--threads', default=2, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of gpus')
    parser.add_argument('--mode', default='parallel', type=str, help='running mode: [sequential , parallel]')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--epoch', default=60, type=int, help='to vis tracking results')
    parser.add_argument('--video', type=str, default='', help='Sequence name for debug.')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    cur_dir = abspath(dirname(__file__))
    ## path initialization
    seq_list = None
    seq_home = args.data_path 
    seq_list = os.listdir(seq_home)
    seq_list.sort()
    print ('seq_list = ', seq_list)

    if seq_home.find('VIS') != -1:
        mode_type = 'vis'
    elif seq_home.find('RedNIR') != -1:
        mode_type = 'rednir'
    elif seq_home.find('NIR') != -1:
        mode_type = 'nir'
    else:
        raise Exception
    print ('mode_type = ', mode_type)

    start = time.time()
    detArr = []
    gtArr = []
    tmp_vds_arr = []

    if False and args.mode == 'parallel':
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.debug, args.epoch, mode_type) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.debug, args.epoch, args.model_path, mode_type) for s in seq_list]
        for seqlist in sequence_list[:]:
            RGB_gt, detRes = run_sequence(*seqlist)
            save_txt_results(seqlist[0], detRes, 'save_res', mode_type)
            overlap_arr = []
            for cnt in range(len(detRes)):
                overlap_arr.append(overlap_ratio(np.array(RGB_gt[cnt]), np.array(detRes[cnt]))[0])
            detArr.append(detRes)
            gtArr.append(RGB_gt)
            tmp_vds_arr.append(seqlist[0])
            print ('video_dir_arr = ',seqlist[0],' , overlap_arr = ',np.array(overlap_arr).mean())
    calAUC(gtArr,detArr,tmp_vds_arr,len(tmp_vds_arr))
    print(f"Totally cost {time.time()-start} seconds!")

