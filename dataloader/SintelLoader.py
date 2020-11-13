from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from utils.preprocess import *
from torchvision import transforms
import time
from dataloader.EXRloader import load_exr

class SintelDataset(Dataset):

    def __init__(self, txt_file, root_dir, phase='train', load_disp=True, load_norm=True, to_angle=False, scale_size=(448, 1024)):
        """
        Args:
            txt_file [string]: Path to the image list
            transform (callable, optional): Optional transform to be applied                on a sample
        """
        with open(txt_file, "r") as f:
            self.imgPairs = f.readlines()

        self.root_dir = root_dir
        self.phase = phase
        self.load_disp = load_disp
        self.load_norm = load_norm
        self.to_angle = to_angle
        self.scale_size = scale_size
        self.fx = 1120.0
        self.fy = 1120.0
        self.img_size = (436, 1024)

        self.load_norm = False

    def get_focal_length(self):
        return self.fx, self.fy

    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size

    def __len__(self):
        return len(self.imgPairs)

    def __getitem__(self, idx):

        img_names = self.imgPairs[idx].rstrip().split()

        img_left_name = os.path.join(self.root_dir, img_names[0])
        img_right_name = os.path.join(self.root_dir, img_names[1])
        if self.load_disp:
            gt_disp_name = os.path.join(self.root_dir, img_names[2])
        if self.load_norm:
            gt_norm_name = os.path.join(self.root_dir, img_names[3])

        def load_rgb(filename):
            img = None
            if filename.find('.npy') > 0:
                img = np.load(filename)
            else:
                img = io.imread(filename)
                if len(img.shape) == 2:
                    img = img[:,:,np.newaxis]
                    img = np.pad(img, ((0, 0), (0, 0), (0, 2)), 'constant')
                    img[:,:,1] = img[:,:,0]
                    img[:,:,2] = img[:,:,0]
                h, w, c = img.shape
                if c == 4:
                    img = img[:,:,:3]
            return img
           
        def load_disp(filename):
            gt_disp = None
            if gt_disp_name.endswith('pfm'):
                gt_disp, scale = load_pfm(gt_disp_name)
                gt_disp = gt_disp[::-1, :]
            elif gt_disp_name.endswith('npy'):
                gt_disp = np.load(gt_disp_name)
                gt_disp = gt_disp[::-1, :]
            elif gt_disp_name.endswith('exr'):
                gt_disp = load_exr(filename)
            else:
                f_in = np.array(Image.open(gt_disp_name))
                d_r = f_in[:,:,0].astype('float32')
                d_g = f_in[:,:,1].astype('float32')
                d_b = f_in[:,:,2].astype('float32')
                gt_disp = d_r * 4 + d_g / (2**6) + d_b / (2**14)

            return gt_disp

        def load_norm(filename):
            gt_norm = None
            if filename.endswith('exr'):
                gt_norm = load_exr(filename)
                
                # transform visualization normal to its true value
                gt_norm = gt_norm * 2.0 - 1.0

                ## fix opposite normal
                #m = gt_norm >= 0
                #m[:,:,0] = False
                #m[:,:,1] = False
                #gt_norm[m] = - gt_norm[m]
                is_nan = np.isnan(gt_norm)
                gt_norm[is_nan] = 1.0

            return gt_norm

        s = time.time()
        img_left = load_rgb(img_left_name)
        img_right = load_rgb(img_right_name)
        if self.load_disp:
            gt_disp = load_disp(gt_disp_name)
        if self.load_norm:
            gt_norm = load_norm(gt_norm_name)
        #print("load data in %f s." % (time.time() - s))

        s = time.time()
        if self.phase == 'detect' or self.phase == 'test':
            img_left = transform.resize(img_left, self.scale_size, preserve_range=True)
            img_right = transform.resize(img_right, self.scale_size, preserve_range=True)

            # change image pixel value type ot float32
            img_left = img_left.astype(np.float32)
            img_right = img_right.astype(np.float32)
            #scale = RandomRescale((1024, 1024))
            #sample = scale(sample)

            #self.fx = self.fx * self.scale_size[1] / self.sx
            #self.fy = self.fy * self.scale_size[0] / self.sy

        if self.phase == 'detect' or self.phase == 'test':
            rgb_transform = default_transform()
        else:
            rgb_transform = inception_color_preproccess()

        img_left = rgb_transform(img_left)
        img_right = rgb_transform(img_right)

        if self.load_disp:
            gt_disp = gt_disp[np.newaxis, :]
            gt_disp = torch.from_numpy(gt_disp.copy()).float()

        if self.load_norm:
            gt_norm = gt_norm.transpose([2, 0, 1])
            gt_norm = torch.from_numpy(gt_norm.copy()).float()

        if self.phase == 'train':

            h, w = img_left.shape[1:3]
            th, tw = 384, 768
            top = random.randint(0, h - th)
            left = random.randint(0, w - tw)

            img_left = img_left[:, top: top + th, left: left + tw]
            img_right = img_right[:, top: top + th, left: left + tw]
            if self.load_disp:
                gt_disp = gt_disp[:, top: top + th, left: left + tw]
            if self.load_norm:
                gt_norm = gt_norm[:, top: top + th, left: left + tw]
    
        if self.to_angle:
            norm_size = gt_norm.size()
            gt_angle = torch.empty(2, norm_size[1], norm_size[2], dtype=torch.float)
            gt_angle[0, :, :] = torch.atan(gt_norm[0, :, :] / gt_norm[2, :, :])
            gt_angle[1, :, :] = torch.atan(gt_norm[1, :, :] / gt_norm[2, :, :])
 

        sample = {  'img_left': img_left, 
                    'img_right': img_right, 
                    'img_names': img_names
                 }

        if self.load_disp:
            sample['gt_disp'] = gt_disp
        if self.load_norm:
            if self.to_angle:
                sample['gt_angle'] = gt_angle
            else:
                sample['gt_norm'] = gt_norm

        #print("deal data in %f s." % (time.time() - s))

        return sample

