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

class StereoDataset(Dataset):

    def __init__(self, txt_file, root_dir, phase='train', load_disp=True, load_norm=True, scale_size=(576, 960)):
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
        self.scale_size = scale_size

    def __len__(self):
        return len(self.imgPairs)

    def __getitem__(self, idx):

        img_names = self.imgPairs[idx].rstrip().split()

        img_left_name = os.path.join(self.root_dir, img_names[0])
        img_right_name = os.path.join(self.root_dir, img_names[1])

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
           
        img_left = load_rgb(img_left_name)
        img_right = load_rgb(img_right_name)

        if self.phase == 'detect' or self.phase == 'test':
            img_left = transform.resize(img_left, self.scale_size, preserve_range=True)
            img_right = transform.resize(img_right, self.scale_size, preserve_range=True)

            # change image pixel value type ot float32
            img_left = img_left.astype(np.float32)
            img_right = img_right.astype(np.float32)
            #scale = RandomRescale((1024, 1024))
            #sample = scale(sample)

        if self.phase == 'detect' or self.phase == 'test':
            rgb_transform = default_transform()
        else:
            rgb_transform = inception_color_preproccess()

        img_left = rgb_transform(img_left)
        img_right = rgb_transform(img_right)

        if self.phase == 'train':

            h, w = img_left.shape[1:3]
            th, tw = 384, 768
            top = random.randint(0, h - th)
            left = random.randint(0, w - tw)

            img_left = img_left[:, top: top + th, left: left + tw]
            img_right = img_right[:, top: top + th, left: left + tw]


        sample = {'img_left': img_left, 
                  'img_right': img_right,
                  'img_names': img_names 
                 }

        return sample

