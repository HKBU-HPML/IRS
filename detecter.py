from __future__ import print_function
import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import skimage
#from dispnet import *
#from networks.dispnet_v2 import *
import torch.cuda as ct
#from networks.DispNetCSRes import DispNetCSRes
from net_builder import SUPPORT_NETS, build_net
from losses.multiscaleloss import multiscaleloss
import torch.nn.functional as F
import torch.nn as nn
#from dataset import DispDataset, save_pfm, RandomRescale
from dataloader.StereoLoader import StereoDataset
from dataloader.SceneFlowLoader import SceneFlowDataset
from utils.preprocess import scale_disp, save_pfm, save_exr, scale_norm
from utils.common import count_parameters 
from torch.utils.data import DataLoader
from torchvision import transforms
import psutil

process = psutil.Process(os.getpid())
cudnn.benchmark = True

#input_transform = transforms.Compose([
#        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
#        # transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
#        ])
#
#target_transform = transforms.Compose([
#        transforms.Normalize(mean=[0],std=[1.0])
#        ])
def detect(opt):

    net_name = opt.net
    model = opt.model
    result_path = opt.rp
    file_list = opt.filelist
    filepath = opt.filepath

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    devices = [int(item) for item in opt.devices.split(',')]
    ngpu = len(devices)
    #net = DispNetC(ngpu, True)
    #net = DispNetCSRes(ngpu, False, True)
    #net = DispNetCSResWithMono(ngpu, False, True, input_channel=3)

    # build net according to the net name
    if net_name == "psmnet" or net_name == "ganet":
        net = build_net(net_name)(192)
    elif net_name in ["normnets", "normnetc"]:
        net = build_net(net_name)()
    elif net_name in ["dispnetcres", "dispnetc"]:
        net = build_net(net_name)(batchNorm=False, lastRelu=True)
    else:
        net = build_net(net_name)(batchNorm=False, lastRelu=True)
        if net_name in ["dispnormnet", "dnfusionnet", "dtonnet", "dtonfusionnet"]:
            net.set_focal_length(1050.0, 1050.0)
    net = torch.nn.DataParallel(net, device_ids=devices).cuda()
    #net.cuda()

    model_data = torch.load(model)
    print(model_data.keys())
    if 'state_dict' in model_data.keys():
        net.load_state_dict(model_data['state_dict'])
    else:
        net.load_state_dict(model_data)

    num_of_parameters = count_parameters(net)
    print('Model: %s, # of parameters: %d' % (net_name, num_of_parameters))

    net.eval()

    batch_size = int(opt.batchSize)
    #test_dataset = StereoDataset(txt_file=file_list, root_dir=filepath, phase='detect')
    test_dataset = SceneFlowDataset(txt_file=file_list, root_dir=filepath, phase='detect')
    test_loader = DataLoader(test_dataset, batch_size = batch_size, \
                        shuffle = False, num_workers = 1, \
                        pin_memory = True)

    s = time.time()
    #high_res_EPE = multiscaleloss(scales=1, downscale=1, weights=(1), loss='L1', sparse=False)

    avg_time = []
    display = 50
    warmup = 10
    for i, sample_batched in enumerate(test_loader):
        #if i > 215:
        #    break

        input = torch.cat((sample_batched['img_left'], sample_batched['img_right']), 1)
	if opt.disp_on:
	    target_disp = sample_batched['gt_disp']
	    target_disp = target_disp.cuda()
	if opt.norm_on:
	    target_norm = sample_batched['gt_norm']
	    target_norm = target_norm.cuda()

        # print('input Shape: {}'.format(input.size()))
        num_of_samples = input.size(0)

        #output, input_var = detect_batch(net, sample_batched, opt.net, (540, 960))

        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)

        if i > warmup:
            ss = time.time()
        if opt.net == "psmnet" or opt.net == "ganet":
            output = net(input_var)
        elif opt.net == "dispnetc":
            output = net(input_var)[0]
        elif opt.net == "normnets":
            output = net(input_var)
        elif opt.net in ["dispnormnet", "dtonnet", "dnfusionnet", "dtonfusionnet"]:
            output = net(input_var)
            disp = output[0]
            normal = output[1]
            output = torch.cat((normal, disp), 1)
        else:
            output = net(input_var)[-1] 
 
        if i > warmup:
            avg_time.append((time.time() - ss))
            if (i - warmup) % display == 0:
                print('Average inference time: %f' % np.mean(avg_time))
                mbytes = 1024.*1024
                print('GPU memory usage memory_allocated: %d MBytes, max_memory_allocated: %d MBytes, memory_cached: %d MBytes, max_memory_cached: %d MBytes, CPU memory usage: %d MBytes' %  \
                    (ct.memory_allocated()/mbytes, ct.max_memory_allocated()/mbytes, ct.memory_cached()/mbytes, ct.max_memory_cached()/mbytes, process.memory_info().rss/mbytes))
                avg_time = []

        # output = net(input_var)[1]
        if opt.disp_on and not opt.norm_on:
            output = scale_disp(output, (output.size()[0], 540, 960))
            disp = output[:, 0, :, :]
        elif opt.norm_on:
            output = scale_norm(output, (output.size()[0], output.size()[1], 540, 960))
            normal = output[:, :3, :, :]
            if opt.disp_on:
                disp = output[:, 3, :, :]

        for j in range(num_of_samples):

            name_items = sample_batched['img_names'][0][j].split('/')
            # write disparity to file
            if opt.disp_on:
		output_disp = disp[j]
		_target_disp = target_disp[j,0]
		target_valid = _target_disp < 192
		epe = F.smooth_l1_loss(output_disp[target_valid], _target_disp[target_valid], size_average=True)
		print('EPE: {}'.format(epe))

                np_disp = disp[j].data.cpu().numpy()

                print('Batch[{}]: {}, average disp: {}({}-{}).'.format(i, j, np.mean(np_disp), np.min(np_disp), np.max(np_disp)))
                save_name = '_'.join(name_items).replace(".png", "_d.png")# for girl02 dataset
                print('Name: {}'.format(save_name))

                skimage.io.imsave(os.path.join(result_path, save_name),(np_disp*256).astype('uint16'))
                #save_name = '_'.join(name_items).replace("png", "pfm")# for girl02 dataset
                #print('Name: {}'.format(save_name))
                #np_disp = np.flip(np_disp, axis=0)
                #save_pfm('{}/{}'.format(result_path, save_name), np_disp)
            
            if opt.norm_on:
                normal[j] = (normal[j] + 1.0) * 0.5
                #np_normal = normal[j].data.cpu().numpy().transpose([1, 2, 0])
                np_normal = normal[j].data.cpu().numpy()
                #save_name = '_'.join(name_items).replace('.png', '_n.png')
                save_name = '_'.join(name_items).replace('png', 'exr')
                print('Name: {}'.format(save_name))
                #skimage.io.imsave(os.path.join(result_path, save_name),(normal*256).astype('uint16'))
                #save_pfm('{}/{}'.format(result_path, save_name), img)
		save_exr(np_normal, '{}/{}'.format(result_path, save_name))
		

            print('')

            #save_name = '_'.join(name_items).replace(".png", "_left.png")# for girl02 dataset
            #img = input_var[0].detach().cpu().numpy()[:3,:,:]
            #img = np.transpose(img, (1, 2, 0))
            #print('Name: {}'.format(save_name))
            #print('')
            ##save_pfm('{}/{}'.format(result_path, save_name), img)
            #skimage.io.imsave(os.path.join(result_path, save_name),img)

    print('Evaluation time used: {}'.format(time.time()-s))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='indicate the name of net', default='dispnetcres', choices=SUPPORT_NETS)
    parser.add_argument('--model', type=str, help='model to load', default='best.pth')
    parser.add_argument('--filelist', type=str, help='file list', default='FlyingThings3D_release_TEST.list')
    parser.add_argument('--filepath', type=str, help='file path', default='./data')
    parser.add_argument('--devices', type=str, help='devices', default='0')
    parser.add_argument('--display', type=int, help='Num of samples to print', default=10)
    parser.add_argument('--rp', type=str, help='result path', default='./result')
    parser.add_argument('--flowDiv', type=float, help='flow division', default='1.0')
    parser.add_argument('--batchSize', type=int, help='mini batch size', default=1)
    parser.add_argument('--disp-on', action='store_true', help='enables, disparity')
    parser.add_argument('--norm-on', action='store_true', help='enables, normal')

    opt = parser.parse_args()
    detect(opt)
