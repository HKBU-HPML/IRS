from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
from layers_package.resample2d_package.resample2d import Resample2d
from layers_package.channelnorm_package.channelnorm import ChannelNorm
from networks.DispNetC import DispNetC
from networks.NormNetDF import NormNetDF
from networks.submodules import *

class DToNFusionNet(nn.Module):

    def __init__(self, batchNorm=False, lastRelu=True, input_channel=3, maxdisp=-1):
        super(DToNFusionNet, self).__init__()
        self.input_channel = input_channel
        self.batchNorm = batchNorm
        self.lastRelu = lastRelu
        self.maxdisp = maxdisp

        # First Block (DispNetC)
        self.dispnetc = DispNetC(self.batchNorm, input_channel=input_channel, get_features=True)
        # Second and third Block (DispNetS), input is 6+3+1+1=11
        self.normnetdf = NormNetDF(input_channel=3+3+3+1)

        self.fx = None
        self.fy = None

        # # parameter initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         if m.bias is not None:
        #             init.uniform(m.bias)
        #         init.xavier_uniform(m.weight)

        #     if isinstance(m, nn.ConvTranspose2d):
        #         if m.bias is not None:
        #             init.uniform(m.bias)
        #         init.xavier_uniform(m.weight)

    def set_focal_length(self, fx, fy):
        self.fx = fx
        self.fy = fy

    def forward(self, inputs):

        # split left image and right image
        # inputs = inputs_target[0]
        # target = inputs_target[1]
        imgs = torch.chunk(inputs, 2, dim = 1)
        img_left = imgs[0]
        img_right = imgs[1]

        # dispnetc
        dispnetc_flows, dispnetc_features = self.dispnetc(inputs)
        dispnetc_flow = dispnetc_flows[0]
        #depthc = dispnetc_flow.clone()
        #depthc[depthc == 0] = 0.01
        #depthc = 48.0 / depthc
        #depthc[depthc > 30] = 30

        # convert disparity to normal
        init_normal = disp2norm(dispnetc_flow+0.01, self.fx, self.fy)

        # normnetdf
        inputs_normnetdf = torch.cat((inputs, init_normal, dispnetc_flow), dim = 1)
        normal = self.normnetdf(inputs_normnetdf, dispnetc_features)

        #normal = normal / torch.norm(normal, 2, dim=1, keepdim=True)

        if self.training:
            return dispnetc_flows, normal
        else:
            return dispnetc_flow, normal# , inputs[:, :3, :, :], inputs[:, 3:, :, :], resampled_img1


    def weight_parameters(self):
	return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
	return [param for name, param in self.named_parameters() if 'bias' in name]

    
