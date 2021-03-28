from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
from networks.DispNetC import DispNetC
from networks.DispNetRes import DispNetRes
from networks.submodules import *

class FADNet(nn.Module):

    def __init__(self, batchNorm=True, lastRelu=False, resBlock=True, maxdisp=-1, input_channel=3):
        super(FADNet, self).__init__()
        self.input_channel = input_channel
        self.batchNorm = batchNorm
        self.lastRelu = lastRelu
        self.maxdisp = maxdisp
        self.resBlock = resBlock

        # First Block (DispNetC)
        self.dispnetc = DispNetC(self.batchNorm, maxdisp=self.maxdisp, input_channel=input_channel)

        # Second Block (DispNetRes), input is 11 channels(img0, img1, img1->img0, flow, diff-mag)
        in_planes = 3 * 3 + 1 + 1
        self.dispnetres = DispNetRes(in_planes, self.batchNorm, lastRelu=self.lastRelu, maxdisp=self.maxdisp, input_channel=input_channel)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, inputs):

        # split left image and right image
        imgs = torch.chunk(inputs, 2, dim = 1)
        img_left = imgs[0]
        img_right = imgs[1]

        # dispnetc
        dispnetc_flows = self.dispnetc(inputs)
        dispnetc_final_flow = dispnetc_flows[0]

        # warp img1 to img0; magnitude of diff between img0 and warped_img1,
        resampled_img1 = warp_right_to_left(inputs[:, self.input_channel:, :, :], -dispnetc_final_flow)
        diff_img0 = inputs[:, :self.input_channel, :, :] - resampled_img1
        norm_diff_img0 = channel_length(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-img
        inputs_net2 = torch.cat((inputs, resampled_img1, dispnetc_final_flow, norm_diff_img0), dim = 1)

        # dispnetres
        dispnetres_flows = self.dispnetres([inputs_net2, dispnetc_flows])
        index = 0
        dispnetres_final_flow = dispnetres_flows[index]
        

        if self.training:
            return dispnetc_flows, dispnetres_flows
        else:
            return dispnetc_final_flow, dispnetres_final_flow

    def weight_parameters(self):
    	return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
    	return [param for name, param in self.named_parameters() if 'bias' in name]


