from __future__ import print_function
import os, sys, gc
import time
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from net_builder import build_net
from dataloader.SceneFlowLoader import SceneFlowDataset
from dataloader.IRSLoader import IRSDataset
from dataloader.SintelLoader import SintelDataset
from dataloader.MiddleburyLoader import MiddleburyDataset
from utils.AverageMeter import AverageMeter
from utils.common import logger
from losses.multiscaleloss import EPE
from losses.normalloss import angle_diff_angle, angle_diff_norm
from utils.preprocess import scale_disp, scale_norm, scale_angle
import skimage

class DisparityTrainer(object):
    def __init__(self, net_name, lr, devices, dataset, trainlist, vallist, datapath, batch_size, maxdisp, pretrain=None):
        super(DisparityTrainer, self).__init__()
        self.net_name = net_name
        self.lr = lr
        self.current_lr = lr
        self.devices = devices
        self.devices = [int(item) for item in devices.split(',')]
        ngpu = len(devices)
        self.ngpu = ngpu
        self.trainlist = trainlist
        self.vallist = vallist
        self.dataset = dataset
        self.datapath = datapath
        self.batch_size = batch_size
        self.pretrain = pretrain 
        self.maxdisp = maxdisp

        #self.criterion = criterion
        self.criterion = None
        self.epe = EPE
        self.set_target()
        self.initialize()

    def _prepare_dataset(self):
        if self.dataset == 'sceneflow' or self.dataset == 'irs':
            train_dataset = SceneFlowDataset(txt_file = self.trainlist, root_dir = self.datapath, phase='train', load_disp=self.disp_on, load_norm=self.norm_on)
            test_dataset = SceneFlowDataset(txt_file = self.vallist, root_dir = self.datapath, phase='test', load_disp=self.disp_on, load_norm=self.norm_on)
        if self.dataset == 'middlebury':
            train_dataset = MiddleburyDataset(txt_file = self.trainlist, root_dir = self.datapath, phase='train')
            test_dataset = MiddleburyDataset(txt_file = self.vallist, root_dir = self.datapath, phase='test')
        if self.dataset == 'sintel':
            train_dataset = SintelDataset(txt_file = self.trainlist, root_dir = self.datapath, phase='train')
            test_dataset = SintelDataset(txt_file = self.vallist, root_dir = self.datapath, phase='test')
        
        self.img_size = test_dataset.get_img_size()
        self.scale_size = test_dataset.get_scale_size()
        self.focal_length = test_dataset.get_focal_length()

        datathread=4
        if os.environ.get('datathread') is not None:
            datathread = int(os.environ.get('datathread'))
        logger.info("Use %d processes to load data..." % datathread)
        self.train_loader = DataLoader(train_dataset, batch_size = self.batch_size, \
                                shuffle = True, num_workers = datathread, \
                                pin_memory = True)
        
        self.test_loader = DataLoader(test_dataset, batch_size = self.batch_size, \
                                shuffle = False, num_workers = datathread, \
                                pin_memory = True)
        self.num_batches_per_epoch = len(self.train_loader)


    def set_target(self):
        # set predicted target
        if self.net_name in ['dnfusionnet', 'dtonnet']:
            self.disp_on = True
            self.norm_on = True
            self.angle_on = False
        elif self.net_name in ["normnets"]:
            self.disp_on = False
            self.norm_on = True
            self.angle_on = False
        else:
            self.disp_on = True
            self.norm_on = False
            self.angle_on = False

    def _build_net(self):

        # build net according to the net name
        if self.net_name == "psmnet":
            self.net = build_net(self.net_name)(self.maxdisp)
        elif self.net_name in ["normnets"]:
            self.net = build_net(self.net_name)()
        else:
            self.net = build_net(self.net_name)(batchNorm=False, lastRelu=True, maxdisp=self.maxdisp)

        if self.net_name in ['dnfusionnet', 'dtonnet']:
            self.net.set_focal_length(self.focal_length[0], self.focal_length[1])

        self.is_pretrain = False

        if self.ngpu >= 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.devices).cuda()
        else:
            self.net.cuda()

        if self.pretrain == '':
            logger.info('Initial a new model...')
        else:
            if os.path.isfile(self.pretrain):
                model_data = torch.load(self.pretrain)
                logger.info('Load pretrain model: %s', self.pretrain)
                if 'state_dict' in model_data.keys():
                    self.net.load_state_dict(model_data['state_dict'])
                elif 'model' in model_data.keys():
                    self.net.load_state_dict(model_data['model'])
                else:
                    self.net.load_state_dict(model_data)
                self.is_pretrain = True
            else:
                logger.warning('Can not find the specific model %s, initial a new model...', self.pretrain)


    def _build_optimizer(self):
        beta = 0.999
        momentum = 0.9
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), self.lr,
                                        betas=(momentum, beta), amsgrad=True)

    def initialize(self):
        self._prepare_dataset()
        self._build_net()
        self._build_optimizer()

    def adjust_learning_rate(self, epoch):
        cur_lr = self.lr / (2**(epoch// 10))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr
        self.current_lr = cur_lr
        return cur_lr

    def set_criterion(self, criterion):
        self.criterion = criterion
 
    def train_one_epoch(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        flow2_EPEs = AverageMeter()
        norm_EPEs = AverageMeter()
        angle_EPEs = AverageMeter()
        # switch to train mode
        self.net.train()
        end = time.time()
        cur_lr = self.adjust_learning_rate(epoch)
        logger.info("learning rate of epoch %d: %f." % (epoch, cur_lr))

        for i_batch, sample_batched in enumerate(self.train_loader):
         
            left_input = torch.autograd.Variable(sample_batched['img_left'].cuda(), requires_grad=False)
            right_input = torch.autograd.Variable(sample_batched['img_right'].cuda(), requires_grad=False)
            input = torch.cat((left_input, right_input), 1)

            if self.disp_on:
                target_disp = sample_batched['gt_disp']
                target_disp = target_disp.cuda()
                target_disp = torch.autograd.Variable(target_disp, requires_grad=False)
            if self.norm_on:
                if self.angle_on:
                    target_angle = sample_batched['gt_angle']
                    target_angle = target_angle.cuda()
                    target_angle = torch.autograd.Variable(target_angle, requires_grad=False)
                else:
                    target_norm = sample_batched['gt_norm']
                    target_norm = target_norm.cuda()
                    target_norm = torch.autograd.Variable(target_norm, requires_grad=False)

            input_var = torch.autograd.Variable(input, requires_grad=False)
            data_time.update(time.time() - end)

            self.optimizer.zero_grad()
            if self.net_name in ['dtonnet', 'dnfusionnet']:
                disp_norm = self.net(input_var)

                disps = disp_norm[0]
                normal = disp_norm[1]
                #print("gt norm[%f-%f], predict norm[%f-%f]." % (torch.min(target_norm).data.item(), torch.max(target_norm).data.item(), torch.min(normal).data.item(), torch.max(normal).data.item()))
                loss_disp = self.criterion(disps, target_disp)
                valid_norm_idx = (target_norm >= -1.0) & (target_norm <= 1.0)
                loss_norm = F.mse_loss(normal[valid_norm_idx], target_norm[valid_norm_idx], size_average=True) * 3.0 
                loss = loss_disp + loss_norm 

                final_disp = disps[0]

                flow2_EPE = self.epe(final_disp, target_disp)
                norm_EPE = loss_norm

            elif self.net_name in ["normnets"]:
                normal = self.net(input_var)
                #print("gt norm[%f-%f], predict norm[%f-%f]." % (torch.min(target_norm).data.item(), torch.max(target_norm).data.item(), torch.min(normal).data.item(), torch.max(normal).data.item()))
                valid_norm_idx = (target_norm >= -1.0) & (target_norm <= 1.0)
                loss_norm = F.mse_loss(normal[valid_norm_idx], target_norm[valid_norm_idx], size_average=True) * 3.0
                #print(loss_disp, loss_norm)
                loss = loss_norm
                norm_EPE = loss_norm 

            elif self.net_name == "fadnet":
                output_net1, output_net2 = self.net(input_var)
                loss_net1 = self.criterion(output_net1, target_disp)
                loss_net2 = self.criterion(output_net2, target_disp)
                loss = loss_net1 + loss_net2
                output_net2_final = output_net2[0]
                flow2_EPE = self.epe(output_net2_final, target_disp)
            elif self.net_name == "dispnetcss":
                output_net1, output_net2, output_net3 = self.net(input_var)
                loss_net1 = self.criterion(output_net1, target_disp)
                loss_net2 = self.criterion(output_net2, target_disp)
                loss_net3 = self.criterion(output_net3, target_disp)
                loss = loss_net1 + loss_net2 + loss_net3
                output_net3_final = output_net3[0]
                flow2_EPE = self.epe(output_net3_final, target_disp)
            elif self.net_name == "psmnet":
                mask = target_disp < self.maxdisp
                mask.detach_()

                output1, output2, output3 = self.net(input_var)
                output1 = torch.unsqueeze(output1,1)
                output2 = torch.unsqueeze(output2,1)
                output3 = torch.unsqueeze(output3,1)

                loss = 0.5*F.smooth_l1_loss(output1[mask], target_disp[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], target_disp[mask], size_average=True) + F.smooth_l1_loss(output3[mask], target_disp[mask], size_average=True)
                flow2_EPE = self.epe(output3, target_disp)
            elif self.net_name == "gwcnet":
                mask = target_disp < self.maxdisp
                mask.detach_()

                output1, output2, output3, output4 = self.net(input_var)

                loss = 0.5*F.smooth_l1_loss(output1[mask], target_disp[mask], size_average=True) + 0.5*F.smooth_l1_loss(output2[mask], target_disp[mask], size_average=True) + 0.7*F.smooth_l1_loss(output3[mask], target_disp[mask], size_average=True) + F.smooth_l1_loss(output4[mask], target_disp[mask], size_average=True)
                flow2_EPE = self.epe(output3, target_disp)
            else:
                output = self.net(input_var)
                loss = self.criterion(output, target_disp)
                if type(loss) is list or type(loss) is tuple:
                    loss = np.sum(loss)
                if type(output) is list or type(output) is tuple:
                    flow2_EPE = self.epe(output[0], target_disp)
                else:
                    flow2_EPE = self.epe(output, target_disp)

            # record loss and EPE
            losses.update(loss.data.item(), input_var.size(0))
            if self.disp_on:
                flow2_EPEs.update(flow2_EPE.data.item(), input_var.size(0))
            if self.norm_on:
                if self.angle_on:
                    angle_EPEs.update(angle_EPE.data.item(), input_var.size(0))
                else:
                    norm_EPEs.update(norm_EPE.data.item(), input_var.size(0))

            # compute gradient and do SGD step
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i_batch % 10 == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'
                  'norm_EPE {norm_EPE.val:.3f} ({norm_EPE.avg:.3f})\t'
                  'angle_EPE {angle_EPE.val:.3f} ({angle_EPE.avg:.3f})'.format(
                  epoch, i_batch, self.num_batches_per_epoch, batch_time=batch_time, 
                  data_time=data_time, loss=losses, flow2_EPE=flow2_EPEs, norm_EPE=norm_EPEs, angle_EPE=angle_EPEs))

            #if i_batch > 20:
            #    break

        return losses.avg, flow2_EPEs.avg

    def validate(self):
        batch_time = AverageMeter()
        flow2_EPEs = AverageMeter()
        norm_EPEs = AverageMeter()
        angle_EPEs = AverageMeter()
        losses = AverageMeter()
        # switch to evaluate mode
        end = time.time()
        valid_norm = 0
        angle_lt = 0
        angle_thres = 11.25
        self.net.eval()
        for i, sample_batched in enumerate(self.test_loader):

            left_input = sample_batched['img_left'].cuda()
            right_input = sample_batched['img_right'].cuda()
            left_input = F.interpolate(left_input, self.scale_size, mode='bilinear')
            right_input = F.interpolate(right_input, self.scale_size, mode='bilinear')

            input_var = torch.cat((left_input, right_input), 1)
            #input_var = torch.autograd.Variable(inputs, requires_grad=False)

            if self.disp_on:
                target_disp = sample_batched['gt_disp']
                target_disp = target_disp.cuda()
                target_disp = torch.autograd.Variable(target_disp, requires_grad=False)
            if self.norm_on:
                if self.angle_on:
                    target_angle = sample_batched['gt_angle']
                    target_angle = target_angle.cuda()
                    target_angle = torch.autograd.Variable(target_angle, requires_grad=False)
                else:
                    target_norm = sample_batched['gt_norm']
                    target_norm = target_norm.cuda()
                    target_norm = torch.autograd.Variable(target_norm, requires_grad=False)

            if self.net_name in ['dnfusionnet', 'dtonnet']:
                with torch.no_grad():
                    disp, normal = self.net(input_var)

                # scale the result
                disp_norm = torch.cat((normal, disp), 1)

                # upsampling the predicted disparity map
                size = target_disp.size()
                disp_norm = scale_norm(disp_norm, (size[0], 4, size[-2], size[-1]), True)

                disp = disp_norm[:, 3, :, :].unsqueeze(1)
                normal = disp_norm[:, :3, :, :]

                valid_norm_idx = (target_norm >= -1.0) & (target_norm <= 1.0)

                norm_EPE = F.mse_loss(normal[valid_norm_idx], target_norm[valid_norm_idx], size_average=True) * 3.0

                flow2_EPE = self.epe(disp, target_disp)
                norm_angle = angle_diff_norm(normal, target_norm).squeeze()

                valid_angle_idx = valid_norm_idx[:,0,:,:] & valid_norm_idx[:,1,:,:] & valid_norm_idx[:,2,:,:]	
                valid_angle_idx = valid_angle_idx.squeeze()

                angle_EPE = torch.mean(norm_angle[valid_angle_idx])

                valid_norm += float(torch.sum(valid_angle_idx))
                angle_lt += float(torch.sum(norm_angle[valid_angle_idx] < angle_thres))
                
                logger.info('percent of < {}: {}.'.format(angle_thres, angle_lt * 1.0 / valid_norm))

                angle_EPE = torch.mean(norm_angle)
                loss = norm_EPE + flow2_EPE

            elif self.net_name in ["normnets"]:
                normal = self.net(input_var)
                size = normal.size()

                # scale the result
                normal = scale_norm(normal, (size[0], 3, self.img_height, self.img_width), True)

                valid_norm_idx = (target_norm >= -1.0) & (target_norm <= 1.0)
                norm_EPE = F.mse_loss(normal[valid_norm_idx], target_norm[valid_norm_idx], size_average=True) * 3.0

                norm_angle = angle_diff_norm(normal, target_norm).squeeze()
                valid_angle_idx = valid_norm_idx[:,0,:,:] & valid_norm_idx[:,1,:,:] & valid_norm_idx[:,2,:,:]	
                valid_angle_idx = valid_angle_idx.squeeze()

                angle_EPE = torch.mean(norm_angle[valid_angle_idx])
                valid_norm += float(torch.sum(valid_angle_idx))
                angle_lt += float(torch.sum(norm_angle[valid_angle_idx] < angle_thres))
                
                logger.info('percent of < {}: {}.'.format(angle_thres, angle_lt * 1.0 / valid_norm))
                loss = norm_EPE
            elif self.net_name == 'fadnet':
                output_net1, output_net2 = self.net(input_var)
                output_net1 = scale_disp(output_net1, (output_net1.size()[0], self.img_size[0], self.img_size[1]))
                output_net2 = scale_disp(output_net2, (output_net2.size()[0], self.img_size[0], self.img_size[1]))

                loss_net1 = self.epe(output_net1, target_disp)
                loss_net2 = self.epe(output_net2, target_disp)
                loss = loss_net1 + loss_net2
                flow2_EPE = self.epe(output_net2, target_disp)
            elif self.net_name == "psmnet" or self.net_name == "gwcnet":
                with torch.no_grad():
                    output_net3 = self.net(input_var)
                if output_net3.dim == 3:
                    output_net3 = output_net3.unsqueeze(1)
                output_net3 = scale_disp(output_net3, (output_net3.size()[0], self.img_size[0], self.img_size[1]))
                loss = self.epe(output_net3, target_disp)
                flow2_EPE = loss
            elif self.net_name == 'dispnetcss':
                output_net1, output_net2, output_net3 = self.net(input_var)
                output_net1 = scale_disp(output_net1, (output_net1.size()[0], self.img_size[0], self.img_size[1]))
                output_net2 = scale_disp(output_net2, (output_net2.size()[0], self.img_size[0], self.img_size[1]))
                output_net3 = scale_disp(output_net3, (output_net3.size()[0], self.img_size[0], self.img_size[1]))

                loss_net1 = self.epe(output_net1, target_disp)
                loss_net2 = self.epe(output_net2, target_disp)
                loss_net3 = self.epe(output_net3, target_disp)
                loss = loss_net1 + loss_net2 + loss_net3
                flow2_EPE = self.epe(output_net3, target_disp)
            else:
                output = self.net(input_var)[0]

                output = scale_disp(output, (output.size()[0], self.img_size[0], self.img_size[1]))
                loss = self.epe(output, target_disp)
                flow2_EPE = loss

            # record loss and EPE
            if loss.data.item() == loss.data.item():
                losses.update(loss.data.item(), input_var.size(0))
            if self.disp_on and (flow2_EPE.data.item() == flow2_EPE.data.item()):
                flow2_EPEs.update(flow2_EPE.data.item(), input_var.size(0))
            if self.norm_on:
                if self.angle_on:
                    if (angle_EPE.data.item() == angle_EPE.data.item()):
                        angle_EPEs.update(angle_EPE.data.item(), input_var.size(0))
                else:
                    if (norm_EPE.data.item() == norm_EPE.data.item()):
                        norm_EPEs.update(norm_EPE.data.item(), input_var.size(0))
                    if (angle_EPE.data.item() == angle_EPE.data.item()):
                        angle_EPEs.update(angle_EPE.data.item(), input_var.size(0))
        
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % 1 == 0:
                logger.info('Test: [{0}/{1}]\t Time {2}\t EPE {3}\t norm_EPE {4}\t angle_EPE {5}'
                      .format(i, len(self.test_loader), batch_time.val, flow2_EPEs.val, norm_EPEs.val, angle_EPEs.val))

        logger.info(' * EPE {:.3f}'.format(flow2_EPEs.avg))
        logger.info(' * normal EPE {:.3f}'.format(norm_EPEs.avg))
        logger.info(' * angle EPE {:.3f}'.format(angle_EPEs.avg))
        return flow2_EPEs.avg

    def get_model(self):
        return self.net.state_dict()


