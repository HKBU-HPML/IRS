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
        self.multiScales = [nn.AvgPool2d(self.downscale*(2**i), self.downscale*(2**i)) for i in range(scales)]
        print('self.multiScales: ', self.multiScales, ' self.downscale: ', self.downscale)

    def forward(self, input, target):
        if (type(input) is tuple) or (type(input) is list):
            out = 0
          
            for i, input_ in enumerate(input):
                target_ = self.multiScales[i](target)
                if self.mask:
                    # work for sparse
                    mask = target > 0
                    mask.detach_()
                    
                    mask = mask.type(torch.cuda.FloatTensor)
                    pooling_mask = self.multiScales[i](mask) 

                    # use unbalanced avg
                    target_ = target_ / pooling_mask

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
