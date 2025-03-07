import torch
import numpy as np

class Preprocessor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std 
        return img_tensor_norm

class PreprocessorMM(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view((1, 6, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view((1, 6, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std 
        return img_tensor_norm

class PreprocessorHSI(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()
        

    def process(self, img_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_rgb_norm = ((img_tensor[:, :3, :, :] / 255.0) - self.mean) / self.std 
        bands = img_tensor[:, 3:, :, :].shape[1]

        mean_hsi = [0.485, 0.456, 0.406] * 9
        mean_hsi = mean_hsi[:bands]
        std_hsi = [0.229, 0.224, 0.225] * 9
        std_hsi = std_hsi[:bands]
        mean_hsi_tensor = torch.tensor(mean_hsi).view((1, bands, 1, 1)).cuda()
        std_hsi_tensor = torch.tensor(std_hsi).view((1, bands, 1, 1)).cuda()

        img_tensor_hsi_norm = ((img_tensor[:, 3:, :, :] / 255.0) - mean_hsi_tensor) / std_hsi_tensor 
        img_tensor_norm = torch.cat([img_tensor_rgb_norm, img_tensor_hsi_norm], dim=1)
        return img_tensor_norm


class PreprocessorX(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std 
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0) 
        return img_tensor_norm, amask_tensor


class PreprocessorX_onnx(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        """img_arr: (H,W,3), amask_arr: (H,W)"""
        # Deal with the image patch
        img_arr_4d = img_arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img_arr_4d = (img_arr_4d / 255.0 - self.mean) / self.std 
        # Deal with the attention mask
        amask_arr_3d = amask_arr[np.newaxis, :, :] 
        return img_arr_4d.astype(np.float32), amask_arr_3d.astype(np.bool)
