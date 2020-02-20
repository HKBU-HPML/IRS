# freda (todo) : 

import torch.nn as nn
import torch
import numpy as np 
from torch.autograd import Variable
import torch.nn.functional as F
#from correlation_package.modules.corr import Correlation1d # from PWC-Net

class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out

def conv(in_planes, out_planes, kernel_size=3, stride=1, batchNorm=False):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias = True):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
            nn.BatchNorm2d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
        )

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

def predict_flow(in_planes, out_planes = 1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=1,padding=1,bias=False)
#def predict_flow(in_planes):
#    return nn.Conv2d(in_planes,1,kernel_size=1,stride=1,padding=0,bias=False)
           
#def corr(in_planes, max_disp=40):
#    return Correlation1d(pad_size=max_disp, kernel_size=1, max_displacement=max_disp, stride1=1, stride2=2, corr_multiply=1)

def build_corr(img_left, img_right, max_disp=40):
    B, C, H, W = img_left.shape
    volume = img_left.new_zeros([B, max_disp, H, W])
    for i in range(max_disp):
        if i > 0:
            volume[:, i, :, i:] = (img_left[:, :, :, i:] * img_right[:, :, :, :-i]).mean(dim=1)
        else:
            volume[:, i, :, :] = (img_left[:, :, :, :] * img_right[:, :, :, :]).mean(dim=1)

    volume = volume.contiguous()
    return volume

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left  = F.pad(torch.index_select(left,  3, Variable(torch.LongTensor([i for i in range(shift,width)])).cuda()),(shift,0,0,0))
        shifted_right = F.pad(torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width-shift)])).cuda()),(shift,0,0,0))
        out = torch.cat((shifted_left,shifted_right),1).view(batch,filters*2,1,height,width)
        return out

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1)
        return out

def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1) 
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)


        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature


class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):
    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


def init_deconv_bilinear(weight):
    f_shape = weight.size()
    heigh, width = f_shape[-2], f_shape[-1]
    f = np.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([heigh, width])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weight.data.fill_(0.)
    for i in range(f_shape[0]):
        for j in range(f_shape[1]):
            weight.data[i,j,:,:] = torch.from_numpy(bilinear)


def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad
    return hook


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


#def save_grad(grads, name):
#    def hook(grad):
#        grads[name] = grad
#    return hook
#import torch
#from channelnorm_package.modules.channelnorm import ChannelNorm 
#model = ChannelNorm().cuda()
#grads = {}
#a = 100*torch.autograd.Variable(torch.randn((1,3,5,5)).cuda(), requires_grad=True)
#a.register_hook(save_grad(grads, 'a'))
#b = model(a)
#y = torch.mean(b)
#y.backward()

def make_grid(batch_size, height, width):
    x = np.arange(0, width)
    y = np.arange(0, height)
    xc, yc = np.meshgrid(x, y)

    x_coord = torch.from_numpy(xc)
    y_coord = torch.from_numpy(yc)

    x_coord = width * 0.5 - x_coord
    y_coord = height * 0.5 - y_coord

    # unsqueeze, final size should be (n, c, h, w)
    x_coord = torch.unsqueeze(x_coord, 0)
    y_coord = torch.unsqueeze(y_coord, 0)

    x_coord = torch.unsqueeze(x_coord, 0)
    y_coord = torch.unsqueeze(y_coord, 0)

    x_coord = x_coord.expand(batch_size, 1, height, width)
    y_coord = y_coord.expand(batch_size, 1, height, width)

    return x_coord, y_coord

def disp2norm(disp, fx, fy):
    # the shape of disp should be (n, c, h, w), and disp should be tensor
    n, c, h, w = disp.size()
    cur_device = disp.device
    xc, yc = make_grid(n, h, w)
    xc = xc.float().to(cur_device)
    yc = yc.float().to(cur_device)

    dx = (disp[:, :, :, :-2] - disp[:, :, :, 2:]) * 0.5
    dy = (disp[:, :, :-2, :] - disp[:, :, 2:, :]) * 0.5

    #nx = -self.fl.fx * dx / torch.abs(disp[:,:,:,1:-1] - self.xc[:,:,:,1:-1] * dx)
    #ny = -self.fl.fy * dy / torch.abs(disp[:,:,1:-1,:] - self.yc[:,:,1:-1,:] * dy)
    divider_nx = torch.clamp(torch.abs(disp[:,:,:,1:-1] - xc[:,:,:,1:-1] * dx), 1e-8, 1e+16)
    divider_ny = torch.clamp(torch.abs(disp[:,:,1:-1,:] - yc[:,:,1:-1,:] * dy), 1e-8, 1e+16)
    nx = fx * dx / divider_nx
    ny = fy * dy / divider_ny
    nz = -torch.ones(n, 1, h, w).float().to(cur_device)

    nx = F.pad(nx, (1, 1, 0, 0), 'replicate')
    ny = F.pad(ny, (0, 0, 1, 1), 'replicate')
    #print "mean of nx:", torch.mean(nx), torch.std(nx)
    #print "mean of ny:", torch.mean(ny), torch.std(ny)

    norm = torch.cat((nx, ny, nz), dim=1)
    norm = norm / torch.norm(norm, 2, dim=1, keepdim=True)

    #norm[norm < 0] += 1e-6
    #print "min-max:", torch.min(norm), torch.max(norm)
    
    return norm

def vote(norm1, norm2):
    #normal should be normalized
    dot = torch.sum(norm1 * norm2, dim=1).unsqueeze(1)
    dot = torch.clamp(dot, 0, 1)
    return dot 

def norm_adjust_disp_vote(init_disp, norm, fx=1050.0, fy=1050.0, ori_vote=1.0, k_threshold=0.3):
    n, c, h, w = init_disp.size()
    cur_device = init_disp.device
    xc, yc = make_grid(n, h, w)
    xc = xc.float().to(cur_device)
    yc = yc.float().to(cur_device)

    nx = fx * norm[:, 2, :, :] / (norm[:, 0, :, :] + 1e-12)
    ny = fy * norm[:, 2, :, :] / (norm[:, 1, :, :] + 1e-12)
    nx = nx.unsqueeze(1)
    ny = ny.unsqueeze(1)

    nx = F.pad(nx, (2, 2, 2, 2), 'replicate')
    ny = F.pad(ny, (2, 2, 2, 2), 'replicate')
    pad_disp = F.pad(init_disp, (1, 1, 1, 1), 'constant', 0.0)
    xc = -F.pad(xc, (1, 1, 1, 1), 'replicate')
    yc = -F.pad(yc, (1, 1, 1, 1), 'replicate')

    kx_l = torch.clamp(-1 / (xc + (nx[:,:,1:-1,1:-1] + nx[:,:,1:-1,:-2]) * 0.5 + 1e-12), -k_threshold, k_threshold) 
    kx_r = torch.clamp( 1 / (xc + (nx[:,:,1:-1,1:-1] + nx[:,:,1:-1, 2:]) * 0.5 + 1e-12), -k_threshold, k_threshold)
    ky_u = torch.clamp(-1 / (yc + (ny[:,:,1:-1,1:-1] + ny[:,:,:-2,1:-1]) * 0.5 + 1e-12), -k_threshold, k_threshold)
    ky_d = torch.clamp( 1 / (yc + (ny[:,:,1:-1,1:-1] + ny[:,:, 2:,1:-1]) * 0.5 + 1e-12), -k_threshold, k_threshold)

    disp_l = pad_disp + pad_disp * kx_l
    disp_r = pad_disp + pad_disp * kx_r
    disp_u = pad_disp + pad_disp * ky_u
    disp_d = pad_disp + pad_disp * ky_d

    norm = F.pad(norm, (1, 1, 1, 1), 'constant', 0.0)
    vote_l = vote(norm[:,:,1:-1,1:-1], norm[:,:,1:-1,2:])
    vote_r = vote(norm[:,:,1:-1,1:-1], norm[:,:,1:-1,:-2])
    vote_u = vote(norm[:,:,1:-1,1:-1], norm[:,:,2:,1:-1])
    vote_d = vote(norm[:,:,1:-1,1:-1], norm[:,:,:-2,1:-1])

    merge_disp = disp_r[:,:,1:-1,:-2] * vote_r + disp_l[:,:,1:-1,2:] * vote_l + disp_d[:,:,:-2,1:-1] * vote_d + disp_u[:,:,2:,1:-1] * vote_u + init_disp * ori_vote
    merge_disp = merge_disp / ((vote_r + vote_l + vote_d + vote_u + ori_vote) + 1e-12)

    #select =  (merge_disp - init_disp).abs() > (init_disp * 0.5)
    #merge_disp[select] = init_disp[select]

    #print "mean of init_disp, mean of norm, mean of refined disp:", torch.mean(init_disp).data.item(), torch.mean(norm).data.item(), torch.mean(merge_disp).data.item()

    return merge_disp

class Disp2Norm:
    def __init__(self, batch_size, width, height, fx, fy):
        self.fx = fx
        self.fy = fy
        self.batch_size = batch_size
        self.xc, self.yc = self.make_grid(batch_size, width, height)

    def make_grid(self, batch_size, width, height):
        x = np.arange(0, width)
        y = np.arange(0, height)
        x_coord, y_coord = np.meshgrid(x, y)

        x_coord = torch.from_numpy(x_coord).cuda().float()
        y_coord = torch.from_numpy(y_coord).cuda().float()

        x_coord = width * 0.5 - x_coord
        y_coord = height * 0.5 - y_coord

        # unsqueeze, final size should be (n, c, h, w)
        x_coord = torch.unsqueeze(x_coord, 0)
        y_coord = torch.unsqueeze(y_coord, 0)

        x_coord = torch.unsqueeze(x_coord, 0)
        y_coord = torch.unsqueeze(y_coord, 0)

        x_coord = x_coord.expand(batch_size, 1, height, width)
        y_coord = y_coord.expand(batch_size, 1, height, width)

        return x_coord, y_coord

    def disp2norm(self, disp):
        # the shape of disp should be (n, c, h, w), and disp should be tensor
        n, c, h, w = disp.size()

        dx = (disp[:, :, :, :-2] - disp[:, :, :, 2:]) * 0.5
        dy = (disp[:, :, :-2, :] - disp[:, :, 2:, :]) * 0.5

        nx = -self.fx * dx / torch.abs(disp[:,:,:,1:-1] - self.xc[:,:,:,1:-1] * dx)
        ny = -self.fy * dy / torch.abs(disp[:,:,1:-1,:] - self.yc[:,:,1:-1,:] * dy)
        nz = torch.ones(n, 1, h, w).cuda()

        nx = F.pad(nx, (1, 1, 0, 0), 'replicate')
        ny = F.pad(ny, (0, 0, 1, 1), 'replicate')

        norm = torch.cat((nx, ny, nz), dim=1)
        norm = norm / torch.norm(norm, 2, dim=1, keepdim=True)
        
        return norm

    def disp2angle(self, disp):
        # the shape of disp should be (n, c, h, w), and disp should be tensor
        n, c, h, w = disp.size()

        dx = (disp[:, :, :, :-2] - disp[:, :, :, 2:]) * 0.5
        dy = (disp[:, :, :-2, :] - disp[:, :, 2:, :]) * 0.5

        ax = torch.atan(-self.fx * dx / torch.abs(disp[:,:,:,1:-1] - self.xc[:,:,:,1:-1] * dx))
        ay = torch.atan(-self.fy * dy / torch.abs(disp[:,:,1:-1,:] - self.yc[:,:,1:-1,:] * dy))

        ax = F.pad(ax, (1, 1, 0, 0), 'replicate')
        ay = F.pad(ay, (0, 0, 1, 1), 'replicate')

        angle = torch.cat((ax, ay), dim=1)
        
        return norm

    def norm_adjust_disp(self, init_disp, norm, ori_ratio=0.5, edge_threshold=1):
        n, c, h, w = init_disp.size()

        nx = self.fx * norm[:, 2:3, :, :] / norm[:, 0:1, :, :] 
        ny = self.fy * norm[:, 2:3, :, :] / norm[:, 1:2, :, :]

        nx = F.pad(nx, (1, 1, 1, 1), 'replicate')
        ny = F.pad(ny, (1, 1, 1, 1), 'replicate')

        pad_disp = F.pad(init_disp, (1, 1, 1, 1), 'constant', 0.0)
        ones = torch.ones([n, c, h, w], dtype=torch.float32).cuda()
        ones = F.pad(ones, (1, 1, 1, 1), 'constant', 0)
        xc = -F.pad(self.xc, (1, 1, 1, 1), 'replicate')
        yc = -F.pad(self.yc, (1, 1, 1, 1), 'replicate')

        disp_l = pad_disp - pad_disp / (xc + nx)
        disp_r = pad_disp + pad_disp / (xc + nx)
        disp_u = pad_disp - pad_disp / (yc + ny)
        disp_d = pad_disp + pad_disp / (yc + ny)


        merge_disp = (disp_r[:,:,1:-1,:-2] + disp_l[:,:,1:-1,2:] + disp_d[:,:,:-2,1:-1] + disp_u[:,:,2:,1:-1]) / (ones[:,:,1:-1,:-2] + ones[:,:,1:-1,2:] + ones[:,:,:-2,1:-1] + ones[:,:,2:,1:-1])
        #merge_disp = (disp_r[:,:,1:-1,:-2] + disp_l[:,:,1:-1,2:]) / (ones[:,:,1:-1,:-2] + ones[:,:,1:-1,2:])
        #merge_disp = (disp_r[:,:,1:-1,:-2] + disp_d[:,:,:-2,1:-1]) / (ones[:,:,1:-1,:-2] + ones[:,:,:-2,1:-1] + 1e-15)

        #merge_disp[0,0,0,0] = init_disp[0,0,0,0]
        #merge_disp[0,0,h/2,w/2] = init_disp[0,0,h/2,w/2]


        rx = torch.abs(norm[:, 0:1, :, :] / norm[:, 2:3, :, :])
        ry = torch.abs(norm[:, 1:2, :, :] / norm[:, 2:3, :, :])
        edge = (rx > edge_threshold) | (ry > edge_threshold)
        merge_disp[edge] = init_disp[edge]

        if ori_ratio > 0.0:
            merge_disp = merge_disp * (1 - ori_ratio) + init_disp * ori_ratio

        return merge_disp

    def vote(self, norm1, norm2):
        #normal should be normalized
        mul = norm1 * norm2
        dot = mul[:, 0:1, :, :] + mul[:, 1:2, :, :] + mul[:, 2:3, :, :]
        dot = torch.clamp(dot, 0, 1)
        #dot = dot ** 5
        return dot 

    def norm_adjust_disp_vote(self, init_disp, norm, ori_vote=1.0, k_threshold=0.1):
        n, c, h, w = init_disp.size()

        nx = self.fx * norm[:, 2:3, :, :] / norm[:, 0:1, :, :] 
        ny = self.fy * norm[:, 2:3, :, :] / norm[:, 1:2, :, :]

        nx = F.pad(nx, (2, 2, 2, 2), 'replicate')
        ny = F.pad(ny, (2, 2, 2, 2), 'replicate')
        pad_disp = F.pad(init_disp, (1, 1, 1, 1), 'constant', 0.0)
        xc = -F.pad(self.xc, (1, 1, 1, 1), 'replicate')
        yc = -F.pad(self.yc, (1, 1, 1, 1), 'replicate')

        kx_l = torch.clamp(-1 / (xc + (nx[:,:,1:-1,1:-1] + nx[:,:,1:-1,:-2]) * 0.5), -k_threshold, k_threshold) 
        kx_r = torch.clamp( 1 / (xc + (nx[:,:,1:-1,1:-1] + nx[:,:,1:-1, 2:]) * 0.5), -k_threshold, k_threshold)
        ky_u = torch.clamp(-1 / (yc + (ny[:,:,1:-1,1:-1] + ny[:,:,:-2,1:-1]) * 0.5), -k_threshold, k_threshold)
        ky_d = torch.clamp( 1 / (yc + (ny[:,:,1:-1,1:-1] + ny[:,:, 2:,1:-1]) * 0.5), -k_threshold, k_threshold)

        disp_l = pad_disp + pad_disp * kx_l
        disp_r = pad_disp + pad_disp * kx_r
        disp_u = pad_disp + pad_disp * ky_u
        disp_d = pad_disp + pad_disp * ky_d

        norm = F.pad(norm, (1, 1, 1, 1), 'constant', 0.0)
        vote_l = self.vote(norm[:,:,1:-1,1:-1], norm[:,:,1:-1,2:])
        vote_r = self.vote(norm[:,:,1:-1,1:-1], norm[:,:,1:-1,:-2])
        vote_u = self.vote(norm[:,:,1:-1,1:-1], norm[:,:,2:,1:-1])
        vote_d = self.vote(norm[:,:,1:-1,1:-1], norm[:,:,:-2,1:-1])

        merge_disp = disp_r[:,:,1:-1,:-2] * vote_r + disp_l[:,:,1:-1,2:] * vote_l + disp_d[:,:,:-2,1:-1] * vote_d + disp_u[:,:,2:,1:-1] * vote_u + init_disp * ori_vote
        merge_disp = merge_disp / (vote_r + vote_l + vote_d + vote_u + ori_vote)

        #select =  (merge_disp - init_disp).abs() > (init_disp * 0.5)
        #merge_disp[select] = init_disp[select]

        return merge_disp
