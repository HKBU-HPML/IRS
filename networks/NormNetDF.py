from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
#from layers_package.resample2d_package.modules.resample2d import Resample2d
#from layers_package.channelnorm_package.modules.channelnorm import ChannelNorm
#from correlation_package.modules.corr import Correlation1d # from PWC-Net
from networks.submodules import *


class NormNetDF(nn.Module):

    def __init__(self, input_channel=7):
        super(NormNetDF, self).__init__()
        
        self.input_channel = input_channel
        self.res_scale = 7  # number of residuals

        # improved with shrink res-block layers
        in_planes = input_channel 
        self.conv1   = conv(in_planes, 64, 7, 2)
        self.conv2   = ResBlock(64, 128, 2)
        self.conv3   = ResBlock(128, 256, 2)
        self.conv3_1 = ResBlock(256, 256)
        self.conv4   = ResBlock(256, 512, stride=2)
        self.conv4_1 = ResBlock(512, 512)
        self.conv5   = ResBlock(512, 512, stride=2)
        self.conv5_1 = ResBlock(512, 512)
        self.conv6   = ResBlock(512, 1024, stride=2)
        self.conv6_1 = ResBlock(1024, 1024)

        # original shrink conv layers
        #self.conv2   = conv(self.batchNorm, 64, 128, 5, 2)
        #self.conv3   = conv(self.batchNorm, 128, 256, 5, 2)
        #self.conv3_1 = conv(self.batchNorm, 256, 256)
        #self.conv4   = conv(self.batchNorm, 256, 512, stride=2)
        #self.conv4_1 = conv(self.batchNorm, 512, 512)
        #self.conv5   = conv(self.batchNorm, 512, 512, stride=2)
        #self.conv5_1 = conv(self.batchNorm, 512, 512)
        #self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        #self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        # iconv with deconv layers
        '''
        self.iconv5 = nn.ConvTranspose2d(1025, 512, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d(769, 256, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(385, 128, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(193, 64, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d(97, 32, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(17+self.input_channel, 16, 3, 1, 1)
        '''
        self.iconv5 = nn.ConvTranspose2d(512 * 3, 512, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d(256 + 512 + 256, 256, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(128 +256 + 128, 128, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(64 + 128 + 64, 64, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d(32 + 64 + 32, 32, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(16+3+16, 16, 3, 1, 1)

        # expand and produce disparity
        self.upconv5 = deconv(1024, 512)
        self.upconv4 = deconv(512, 256)
        self.upconv3 = deconv(256, 128)
        self.upconv2 = deconv(128, 64)
        self.upconv1 = deconv(64, 32)
        self.upconv0 = deconv(32, 16)
        self.pred_res0 = predict_flow(16, 3)

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
        
    def forward(self, inputs, depth_features):

        df5, df4, df3, df2, df1, df0 = depth_features

        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3a = self.conv3(conv2)
        conv3b = self.conv3_1(conv3a)
        conv4a = self.conv4(conv3b)
        conv4b = self.conv4_1(conv4a)
        conv5a = self.conv5(conv4b)
        conv5b = self.conv5_1(conv5a)
        conv6a = self.conv6(conv5b)
        conv6b = self.conv6_1(conv6a)

        upconv5 = self.upconv5(conv6b)
        concat5 = torch.cat((upconv5, conv5b, df5), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, conv4b, df4), 1)
        iconv4 = self.iconv4(concat4)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, conv3b, df3), 1)
        iconv3 = self.iconv3(concat3)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, conv2, df2), 1)
        iconv2 = self.iconv2(concat2)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, conv1, df1), 1)
        iconv1 = self.iconv1(concat1)

        upconv0 = self.upconv0(iconv1)
        concat0 = torch.cat((upconv0, inputs[:, :3, :, :], df0), 1)
        iconv0 = self.iconv0(concat0)

        # predict flow residual
        normal = self.pred_res0(iconv0)
        normal = normal / torch.norm(normal, 2, dim=1, keepdim=True)

        # # predict flow residual with dropout output
        # pr6_res = self.pred_res6(F.dropout2d(conv6b))
        # pr5_res = self.pred_res5(F.dropout2d(iconv5))
        # pr4_res = self.pred_res4(F.dropout2d(iconv4))
        # pr3_res = self.pred_res3(F.dropout2d(iconv3))
        # pr2_res = self.pred_res2(F.dropout2d(iconv2))
        # pr1_res = self.pred_res1(F.dropout2d(iconv1))
        # pr0_res = self.pred_res0(F.dropout2d(iconv0))

        return normal

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

