from __future__ import print_function
import os, sys, gc
import time
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from net_builder import build_net
#from dataset import DispDataset
from dataloader.SceneFlowLoader import SceneFlowDataset
from dataloader.SIRSLoader import SIRSDataset
from dataloader.SintelLoader import SintelDataset
from dataloader.MiddleburyLoader import MiddleburyDataset
from dataloader.GANet.data import get_training_set, get_test_set
from utils.AverageMeter import AverageMeter
from utils.common import logger
from losses.multiscaleloss import EPE
from losses.normalloss import angle_diff_angle, angle_diff_norm
from utils.preprocess import scale_disp, scale_norm, scale_angle
from networks.submodules import disp2norm, norm_adjust_disp_vote
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
        if self.dataset == 'irs':
            train_dataset = SIRSDataset(txt_file = self.trainlist, root_dir = self.datapath, phase='train', load_disp = self.disp_on, load_norm = self.norm_on, to_angle = self.angle_on)
            test_dataset = SIRSDataset(txt_file = self.vallist, root_dir = self.datapath, phase='test', load_disp = self.disp_on, load_norm = self.norm_on, to_angle=self.angle_on)
        if self.dataset == 'sceneflow':
            train_dataset = SceneFlowDataset(txt_file = self.trainlist, root_dir = self.datapath, phase='train')
            test_dataset = SceneFlowDataset(txt_file = self.vallist, root_dir = self.datapath, phase='test')
        if self.dataset == 'middlebury':
            train_dataset = MiddleburyDataset(txt_file = self.trainlist, root_dir = self.datapath, phase='train')
            test_dataset = MiddleburyDataset(txt_file = self.vallist, root_dir = self.datapath, phase='test')
        if self.dataset == 'sintel':
            train_dataset = SintelDataset(txt_file = self.trainlist, root_dir = self.datapath, phase='train')
            test_dataset = SintelDataset(txt_file = self.vallist, root_dir = self.datapath, phase='test')
        
        self.fx, self.fy = train_dataset.get_focal_length()
        self.img_height, self.img_width = train_dataset.get_img_size()
        self.scale_height, self.scale_width = test_dataset.get_scale_size()

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

        ## GANet loader
        #print('===> Loading datasets')
        #train_set = get_training_set(self.datapath, self.trainlist, [256, 576], False, False, False, 0)
        #test_set = get_test_set(self.datapath, self.vallist, [576,960], False, False, False)
        #self.train_loader = DataLoader(train_set, batch_size = self.batch_size, \
        #                        shuffle = True, num_workers = 16, \
        #                        pin_memory = True)
        #
        #self.test_loader = DataLoader(test_set, batch_size = self.batch_size / 2, \
        #                        shuffle = False, num_workers = 4, \
        #                        pin_memory = True)
        #self.num_batches_per_epoch = len(self.train_loader)


    def set_target(self):
        # set predicted target
        if self.net_name in ["dispnormnet", "dtonfusionnet", 'dnfusionnet', 'dtonnet', 'dnirrnet']:
            self.disp_on = True
            self.norm_on = True
            self.angle_on = False
        elif self.net_name == "dispanglenet":
            self.disp_on = True
            self.norm_on = True
            self.angle_on = True
        elif self.net_name in ["normnets", "normnetc"]:
            self.disp_on = False
            self.norm_on = True
            self.angle_on = False
        else:
            self.disp_on = True
            self.norm_on = False
            self.angle_on = False

    def _build_net(self):

        # build net according to the net name
        if self.net_name == "psmnet" or self.net_name == "ganet":
            self.net = build_net(self.net_name)(self.maxdisp)
        elif self.net_name in ["normnets", "normnetc"]:
            self.net = build_net(self.net_name)()
        else:
            self.net = build_net(self.net_name)(batchNorm=False, lastRelu=True, maxdisp=self.maxdisp)

        if self.net_name in ["dispnormnet", "dtonfusionnet", 'dnfusionnet', 'dtonnet', 'dnirrnet']:
            self.net.set_focal_length(self.fx, self.fy)
        elif self.net_name == "dispanglenet":
            self.net.set_focal_length(self.fx, self.fy)

        self.is_pretrain = False

        if self.ngpu > 1:
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
        #if epoch < 10:
        #    cur_lr = 0.001
        #elif epoch < 20:
        #    cur_lr = 0.0001
        #else:
        #    cur_lr = 0.0001 / (2**((epoch - 20)// 10))
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
            if self.net_name in ["dispnormnet", "dtonnet", "dtonfusionnet", 'dnfusionnet', 'dnirrnet']:
                disp_norm = self.net(input_var)

                if self.net_name == 'dnirrnet':
                    refined_disp = disp_norm[0]
                    disps = disp_norm[1]
                    normal = disp_norm[2]
                    init_normal = disp_norm[3]
                else:
                    disps = disp_norm[0]
                    normal = disp_norm[1]
                #print("gt norm[%f-%f], predict norm[%f-%f]." % (torch.min(target_norm).data.item(), torch.max(target_norm).data.item(), torch.min(normal).data.item(), torch.max(normal).data.item()))
                loss_disp = self.criterion(disps, target_disp)
                if self.net_name == 'dnirrnet':
                    loss_disp += self.epe(refined_disp, target_disp) * 0.32

                #print("epe before refined: %f. epe after refined: %f." % (self.epe(disps[0], target_disp), self.epe(refined_disp, target_disp)))
                valid_norm_idx = (target_norm >= -1.0) & (target_norm <= 1.0)
                loss_norm = F.mse_loss(normal[valid_norm_idx], target_norm[valid_norm_idx], size_average=True) * 3.0 
                #loss_init_norm = F.mse_loss(init_normal[valid_norm_idx], target_norm[valid_norm_idx], size_average=True) * 3.0
                #print("norm l2 before refined: %f. norm l2 after refined: %f." % (F.mse_loss(init_normal[valid_norm_idx], target_norm[valid_norm_idx], size_average=True), F.mse_loss(normal[valid_norm_idx], target_norm[valid_norm_idx], size_average=True)))
                #print(loss_disp, loss_norm)
                loss = loss_disp + loss_norm #+ loss_init_norm
                #loss = loss_norm

                if self.net_name == 'dnirrnet':
                    final_disp = refined_disp
                else:
                    final_disp = disps[0]

                flow2_EPE = self.epe(final_disp, target_disp)
                norm_EPE = loss_norm

                #retrans_norm = disp2norm(target_disp, self.fx, self.fy)
                #print("average angle diff:", torch.mean(torch.abs(angle_diff_norm(retrans_norm, target_norm))))
                #print("test disp2norm error:", F.l1_loss(retrans_norm, target_norm, size_average=True))

                #target_norm = (target_norm + 1.0) * 0.5
                #print("lowest-highest of GT norm:", torch.min(target_norm), torch.max(target_norm))
                #normal = target_norm[0].data.cpu().numpy().transpose(1, 2, 0) * 256
                #normal[normal < 0] = 0.0
                #normal[normal > 255] = 255.0
                #skimage.io.imsave("gt_test.png",(normal).astype('uint16'))

                #retrans_norm = (retrans_norm + 1.0) * 0.5
                ##retrans_norm[retrans_norm < 0] = 0.0
                ##retrans_norm[retrans_norm > 1] = 1.0
                #print("lowest-highest of trans norm:", torch.min(retrans_norm), torch.max(retrans_norm))
                #retrans_norm = retrans_norm[0].data.cpu().numpy().transpose(1, 2, 0) * 256
                #retrans_norm[retrans_norm < 0] = 0.0
                #retrans_norm[retrans_norm > 255] = 255.0
                #skimage.io.imsave("retrans_test.png",(retrans_norm).astype('uint16'))
                #sys.exit(-1)

            elif self.net_name in ["normnets", "normnetc"]:
                normal = self.net(input_var)
                #print("gt norm[%f-%f], predict norm[%f-%f]." % (torch.min(target_norm).data.item(), torch.max(target_norm).data.item(), torch.min(normal).data.item(), torch.max(normal).data.item()))
                valid_norm_idx = (target_norm >= -1.0) & (target_norm <= 1.0)
                loss_norm = F.mse_loss(normal[valid_norm_idx], target_norm[valid_norm_idx], size_average=True) * 3.0
                #print(loss_disp, loss_norm)
                loss = loss_norm
                norm_EPE = loss_norm 

            elif self.net_name == "dispanglenet":
                disp_angle = self.net(input_var)
                disps = disp_angle[0]
                angle = disp_angle[1]
                loss_disp = self.criterion(disps, target_disp)
                #loss_angle = F.smooth_l1_loss(angle, target_angle, size_average=True)
                loss_angle = F.mse_loss(angle * 180.0 / 3.1415967, target_angle * 180.0 / 3.1415967, size_average=True) 
                loss = loss_disp + loss_angle
                final_disp = disps[0]
                flow2_EPE = self.epe(final_disp, target_disp)
                angle_EPE = F.l1_loss(angle, target_angle, size_average=True) * 180 / 3.1415967
                #angle_EPE = torch.mean(torch.acos(1 - 0.5 * ((torch.tan(angle[:, 0, :, :]) - torch.tan(target_angle[:, 0, :, :]))**2 + (torch.tan(angle[:, 1, :, :]) - torch.tan(target_angle[:, 1, :, :]))**2)))
            elif self.net_name == "dispnetcres":
                output_net1, output_net2 = self.net(input_var)
                loss_net1 = self.criterion(output_net1, target_disp)
                loss_net2 = self.criterion(output_net2, target_disp)
                loss = loss_net1 + loss_net2
                output_net2_final = output_net2[0]
                flow2_EPE = self.epe(output_net2_final, target_disp)
            elif self.net_name == "dispnetcs":
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
            elif self.net_name == "psmnet" or self.net_name == "ganet":
                mask = target_disp < self.maxdisp
                mask.detach_()

                output1, output2, output3 = self.net(input_var)
                output1 = torch.unsqueeze(output1,1)
                output2 = torch.unsqueeze(output2,1)
                output3 = torch.unsqueeze(output3,1)

                loss = 0.5*F.smooth_l1_loss(output1[mask], target_disp[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], target_disp[mask], size_average=True) + F.smooth_l1_loss(output3[mask], target_disp[mask], size_average=True)
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
        self.net.eval()
        end = time.time()
        valid_norm = 0
        angle_lt = 0
        angle_thres = 30.0
        for i, sample_batched in enumerate(self.test_loader):

            left_input = torch.autograd.Variable(sample_batched['img_left'].cuda(), requires_grad=False)
            right_input = torch.autograd.Variable(sample_batched['img_right'].cuda(), requires_grad=False)
            #left_input = torch.autograd.Variable(sample_batched[0].cuda(), requires_grad=False)
            #right_input = torch.autograd.Variable(sample_batched[1].cuda(), requires_grad=False)
            #target = sample_batched[2]

            input = torch.cat((left_input, right_input), 1)
            input_var = torch.autograd.Variable(input, requires_grad=False)
            input_var = F.interpolate(input_var, (576, 960), mode='bilinear')

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

            if self.net_name in ["dispnormnet", "dtonfusionnet", 'dnfusionnet', 'dnirrnet', 'dtonnet']:
                disp, normal = self.net(input_var)

                # scale the result
                disp_norm = torch.cat((normal, disp), 1)

                # upsampling the predicted disparity map
                size = target_disp.size()
                disp_norm = scale_norm(disp_norm, (size[0], 4, size[-2], size[-1]), True)

                ## upsampling the GT disparity map
                #size = disp_norm.size()
                #m = torch.nn.Upsample(size=(int(size[-2]), int(size[-1])), mode="bilinear")
                #origin_width = target_disp.size()[-1]
                #target_disp = m(target_disp) * size[-1] / origin_width

                disp = disp_norm[:, 3, :, :].unsqueeze(1)
                normal = disp_norm[:, :3, :, :]

		# normalize the surface normal
		#normal = normal / torch.norm(normal, 2, dim=1, keepdim=True) 

                valid_norm_idx = (target_norm >= -1.0) & (target_norm <= 1.0)

                #norm_EPE = self.epe(normal, target_disp[:, :3, :, :]) 
                norm_EPE = F.mse_loss(normal[valid_norm_idx], target_norm[valid_norm_idx], size_average=True) * 3.0

                #refined_disp = disp.clone()
                #for ri in range(3):
                #    refined_disp = norm_adjust_disp_vote(refined_disp, normal, self.fx, self.fy)
                #print("epe before refined: %f. epe after refined: %f." % (self.epe(disp, target_disp), self.epe(refined_disp, target_disp)))
                #flow2_EPE = self.epe(refined_disp, target_disp)

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

            elif self.net_name in ["normnets", "normnetc"]:
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
                #angle_EPE = torch.mean(norm_angle)
                loss = norm_EPE
            elif self.net_name == "dispanglenet":
                disp, angle = self.net(input_var)
                size = disp.size()

                # scale the result
                disp_angle = torch.cat((angle, disp), 1)
                disp_angle = scale_angle(disp_angle, (size[0], 3, 540, 960))
                disp = disp_angle[:, 2, :, :].unsqueeze(1)
                angle = disp_angle[:, :2, :, :]

                #angle_EPE = F.l1_loss(angle, target_angle, size_average=True) * 180 / 3.1415967
                norm_angle = angle_diff_angle(angle, target_angle)
                angle_EPE = torch.mean(norm_angle.squeeze()[target_disp.squeeze() > 2])

                #angle_EPE = torch.mean(torch.acos(1 - 0.5 * ((torch.tan(angle[:, 0, :, :]) - torch.tan(target_angle[:, 0, :, :]))**2 + (torch.tan(angle[:, 1, :, :]) - torch.tan(target_angle[:, 1, :, :]))**2)))
                flow2_EPE = self.epe(disp, target_disp)
                loss = angle_EPE + flow2_EPE
            elif self.net_name == 'dispnetcres':
                output_net1, output_net2 = self.net(input_var)
                output_net1 = scale_disp(output_net1, (output_net1.size()[0], 540, 960))
                output_net2 = scale_disp(output_net2, (output_net2.size()[0], 540, 960))

                loss_net1 = self.epe(output_net1, target_disp)
                loss_net2 = self.epe(output_net2, target_disp)
                loss = loss_net1 + loss_net2
                flow2_EPE = self.epe(output_net2, target_disp)
            elif self.net_name == "psmnet" or self.net_name == "ganet":
                output_net3 = self.net(input_var)
                output_net3 = output_net3.unsqueeze(1)
                output_net3 = scale_disp(output_net3, (output_net3.size()[0], 540, 960))
                loss = self.epe(output_net3, target_disp)
                flow2_EPE = loss
            elif self.net_name == 'dispnetcss':
                output_net1, output_net2, output_net3 = self.net(input_var)
                output_net1 = scale_disp(output_net1, (output_net1.size()[0], 540, 960))
                output_net2 = scale_disp(output_net2, (output_net2.size()[0], 540, 960))
                output_net3 = scale_disp(output_net3, (output_net3.size()[0], 540, 960))

                loss_net1 = self.epe(output_net1, target_disp)
                loss_net2 = self.epe(output_net2, target_disp)
                loss_net3 = self.epe(output_net3, target_disp)
                loss = loss_net1 + loss_net2 + loss_net3
                flow2_EPE = self.epe(output_net3, target_disp)
            else:
                output = self.net(input_var)
                output_net1 = output[0]
                size = output_net1.size()
                #output_net1 = output_net1.squeeze(1)

                #output_net1 = scale_disp(output_net1, (size[0], size[-2], size[-1]))
                m = torch.nn.Upsample(size=(int(size[-2]), int(size[-1])), mode="bilinear")
                origin_width = target_disp.size()[-1]
                target_disp = m(target_disp) * size[-1] / origin_width
                loss = self.epe(output_net1, target_disp)
                flow2_EPE = self.epe(output_net1, target_disp)

                #if type(loss) is list or type(loss) is tuple:
                #    loss = loss[0]
                #if type(output) is list or type(output_net1) :
                #    flow2_EPE = self.epe(output[0], target_disp)
                #else:
                #    flow2_EPE = self.epe(output, target_disp)

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

            #if i % 10 == 0:
            #    break

        logger.info(' * EPE {:.3f}'.format(flow2_EPEs.avg))
        logger.info(' * normal EPE {:.3f}'.format(norm_EPEs.avg))
        logger.info(' * angle EPE {:.3f}'.format(angle_EPEs.avg))
        return flow2_EPEs.avg

    def get_model(self):
        return self.net.state_dict()


