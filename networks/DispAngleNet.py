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
from networks.AngleNetDF import AngleNetDF
from networks.submodules import *

class DispAngleNet(nn.Module):

    def __init__(self, batchNorm=False, lastRelu=True, input_channel=3, maxdisp=-1):
        super(DispAngleNet, self).__init__()
        self.input_channel = input_channel
        self.batchNorm = batchNorm
        self.lastRelu = lastRelu
        self.maxdisp = maxdisp

        # First Block (DispNetC)
        self.dispnetc = DispNetC(self.batchNorm, input_channel=input_channel, get_features=True)
        # Second and third Block (DispNetS), input is 6+3+1+1=11
        self.anglenetdf = AngleNetDF(self.batchNorm, input_channel=3+3+1)

        self.relu = nn.ReLU(inplace=False)

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

        # normnetdf
        inputs_anglenetdf = torch.cat((inputs, dispnetc_flow), dim = 1)
        angle = self.anglenetdf(inputs_anglenetdf, dispnetc_features)

        angle = F.tanh(angle) * 1.5708

        if self.training:
            return dispnetc_flows, angle
        else:
            return dispnetc_flow, angle# , inputs[:, :3, :, :], inputs[:, 3:, :, :], resampled_img1


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    
