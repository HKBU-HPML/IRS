from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
#from layers_package.submodules import *
from networks.submodules import *

class NormNetS(nn.Module):

    def __init__(self, input_channel=6):
        super(NormNetS, self).__init__()
        
        self.input_channel = input_channel

        # shrink and extract features
        self.conv1   = conv(self.input_channel, 64, 7, 2)
        self.conv2   = ResBlock(64, 128, 2)
        self.conv3   = ResBlock(128, 256, 2)
        self.conv3_1 = ResBlock(256, 256)
        self.conv4   = ResBlock(256, 512, stride=2)
        self.conv4_1 = ResBlock(512, 512)
        self.conv5   = ResBlock(512, 512, stride=2)
        self.conv5_1 = ResBlock(512, 512)
        self.conv6   = ResBlock(512, 1024, stride=2)
        self.conv6_1 = ResBlock(1024, 1024)

        #self.conv2   = conv(self.batchNorm, 64, 128, 5, 2)
        #self.conv3   = conv(self.batchNorm, 128, 256, 5, 2)
        #self.conv3_1 = conv(self.batchNorm, 256, 256)
        #self.conv4   = conv(self.batchNorm, 256, 512, stride=2)
        #self.conv4_1 = conv(self.batchNorm, 512, 512)
        #self.conv5   = conv(self.batchNorm, 512, 512, stride=2)
        #self.conv5_1 = conv(self.batchNorm, 512, 512)
        #self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        #self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        #self.pred_flow6 = predict_flow(1024, 3)

        # expand and produce disparity
        self.upconv5 = deconv(1024, 512)
        #self.upflow6to5 = nn.ConvTranspose2d(4, 4, 4, 2, 1, bias=False)
        self.iconv5 = nn.ConvTranspose2d(512+512, 512, 3, 1, 1)
        #self.pred_flow5 = predict_flow(512, 3)

        self.upconv4 = deconv(512, 256)
        #self.upflow5to4 = nn.ConvTranspose2d(4, 4, 4, 2, 1, bias=False)
        self.iconv4 = nn.ConvTranspose2d(512+256, 256, 3, 1, 1)
        #self.pred_flow4 = predict_flow(256, 3)

        self.upconv3 = deconv(256, 128)
        #self.upflow4to3 = nn.ConvTranspose2d(4, 4, 4, 2, 1, bias=False)
        self.iconv3 = nn.ConvTranspose2d(256+128, 128, 3, 1, 1)
        #self.pred_flow3 = predict_flow(128, 3)

        self.upconv2 = deconv(128, 64)
        #self.upflow3to2 = nn.ConvTranspose2d(4, 4, 4, 2, 1, bias=False)
        self.iconv2 = nn.ConvTranspose2d(128+64, 64, 3, 1, 1)
        #self.pred_flow2 = predict_flow(64, 3)

        self.upconv1 = deconv(64, 32)
        #self.upflow2to1 = nn.ConvTranspose2d(4, 4, 4, 2, 1, bias=False)
        self.iconv1 = nn.ConvTranspose2d(64+32, 32, 3, 1, 1)
        #self.pred_flow1 = predict_flow(32, 3)

        self.upconv0 = deconv(32, 16)
        #self.upflow1to0 = nn.ConvTranspose2d(4, 4, 4, 2, 1, bias=False)
        self.iconv0 = nn.ConvTranspose2d(16+3, 16, 3, 1, 1)
        self.pred_flow0 = predict_flow(16, 3)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, 0.02 / n)
                # m.weight.data.normal_(0, 0.02)
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, input):

        # split left image and right image
        # print(input.size())
        #imgs = torch.chunk(input[:, :6, :, :], 2, dim = 1)
        img_left = input[:, :3, :, :]
        #img_right = input[:, 3:, :, :]

        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3a = self.conv3(conv2)
        conv3b = self.conv3_1(conv3a)
        conv4a = self.conv4(conv3b)
        conv4b = self.conv4_1(conv4a)
        conv5a = self.conv5(conv4b)
        conv5b = self.conv5_1(conv5a)
        conv6a = self.conv6(conv5b)
        conv6b = self.conv6_1(conv6a)

        #pr6 = self.pred_flow6(conv6b)

        upconv5 = self.upconv5(conv6b)
        concat5 = torch.cat((upconv5, conv5b), 1)
        iconv5 = self.iconv5(concat5)
        #pr5 = self.pred_flow5(iconv5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, conv4b), 1)
        iconv4 = self.iconv4(concat4)
        #pr4 = self.pred_flow4(iconv4)
        
        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, conv3b), 1)
        iconv3 = self.iconv3(concat3)
        #pr3 = self.pred_flow3(iconv3)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, conv2), 1)
        iconv2 = self.iconv2(concat2)
        #pr2 = self.pred_flow2(iconv2)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, conv1), 1)
        iconv1 = self.iconv1(concat1)
        #pr1 = self.pred_flow1(iconv1)

        upconv0 = self.upconv0(iconv1)
        concat0 = torch.cat((upconv0, img_left), 1)
        iconv0 = self.iconv0(concat0)
        pr0 = self.pred_flow0(iconv0)

        normal = pr0 / (torch.norm(pr0, 2, dim=1, keepdim=True) + 1e-8)
	# img_right_rec = warp(img_left, pr0)

        # if self.training:
        #     # print("finish forwarding.")
        #     return pr0, pr1, pr2, pr3, pr4, pr5, pr6
        # else:
        #     return pr0

        # can be chosen outside
        return normal

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


