import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F


def SL_EPE(input_flow, target_flow):
    target_valid = (target_flow < 192) & (target_flow > 0)
    return F.smooth_l1_loss(input_flow[target_valid], target_flow[target_valid], size_average=True)

def EPE(input_flow, target_flow):
    
    target_valid = target_flow < 192
    return F.l1_loss(input_flow[target_valid], target_flow[target_valid], size_average=True)

    #return F.smooth_l1_loss(input_flow, target_flow, size_average=True)

    #EPE_map = torch.norm(target_flow - input_flow + 1e-16, 2, 1)
    #return EPE_map.mean()

class MultiScaleLoss(nn.Module):

    def __init__(self, scales, downscale, weights=None, loss='L1', mask=False):
        super(MultiScaleLoss, self).__init__()
        self.downscale = downscale
        self.mask = mask
        self.weights = torch.Tensor(scales).fill_(1).cuda() if weights is None else torch.Tensor(weights).cuda()
        assert(len(self.weights) == scales)

        if type(loss) is str:

            if loss == 'L1':
                self.loss = nn.L1Loss()
            elif loss == 'MSE':
                self.loss = nn.MSELoss()
            elif loss == 'SmoothL1':
                self.loss = nn.SmoothL1Loss()
            elif loss == 'MAPE':
                self.loss = MAPELoss()
        else:
            self.loss = loss
        #self.multiScales = [nn.AvgPool2d(self.downscale*(2**i), self.downscale*(2**i)) for i in range(scales)]
        self.multiScales = [nn.MaxPool2d(self.downscale*(2**i), self.downscale*(2**i)) for i in range(scales)]
        #self.multiScales = [nn.Upsample(scale_factor=self.downscale*(2**i), mode='bilinear', align_corners=True) for i in range(scales)]
        print('self.multiScales: ', self.multiScales, ' self.downscale: ', self.downscale)
        # self.multiScales = [nn.functional.adaptive_avg_pool2d(self.downscale*(2**i), self.downscale*(2**i)) for i in range(scales)]

    def forward(self, input, target):
        #print(len(input))
        if (type(input) is tuple) or (type(input) is list):
            out = 0
          
            #for i, input_ in enumerate(input):
            #    print(i, len(input_))
            #    print(input_[0].size())
            for i, input_ in enumerate(input):
                #target_ = target.clone()
                #input_ = self.multiScales[i](input_)
                target_ = self.multiScales[i](target)
                ## consider the scale effects
                #target_ = self.multiScales[i](target) / (2**i)
                #print('target shape: ', target_.shape, ' input shape: ', input_.shape)
                if self.mask:
                    ## work for sparse
                    #mask = target > 0
                    #mask.detach_()
                    #
                    #mask = mask.type(torch.cuda.FloatTensor)
                    #pooling_mask = self.multiScales[i](mask) 

                    ## use unbalanced avg
                    #target_ = target_ / pooling_mask

                    mask = target_ > 0
                    mask.detach_()
                    input_ = input_[mask]
                    target_ = target_[mask]

                EPE_ = SL_EPE(input_, target_)
                out += self.weights[i] * EPE_
        else:
            out = self.loss(input, self.multiScales[0](target))
        return out

def multiscaleloss(scales=5, downscale=4, weights=None, loss='L1', sparse=False, mask=False):
    if weights is None:
        weights = (0.005, 0.01, 0.02, 0.08, 0.32)
    if scales == 1 and type(weights) is not tuple:
        weights = (weights, )
    return MultiScaleLoss(scales, downscale, weights, loss, mask)
