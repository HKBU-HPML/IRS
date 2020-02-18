from networks.submodules import *
from dataloader.EXRloader import exr2hdr
import torch
import torch.nn.functional as F
from dataloader.EXRloader import save_exr
import os
import numpy as np


#init_disp_file = 'test_norm2disp/init_disp.exr'
#gt_disp_file = 'test_norm2disp/d_401.exr'
#gt_norm_file = 'test_norm2disp/n_401.exr'

init_disp_file = 'test_norm2disp/init_disp_522.exr'
gt_disp_file = 'test_norm2disp/d_522.exr'
gt_norm_file = 'test_norm2disp/n_522.exr'


def EPE(gt_disp, disp):
    return F.smooth_l1_loss(gt_disp, disp, size_average=True)


init_disp = exr2hdr(init_disp_file)
init_disp = init_disp[::-1,:,1:2].copy()
init_disp = init_disp.transpose([2, 0, 1])
init_disp = init_disp[np.newaxis, :, :, :]
init_disp = torch.from_numpy(init_disp).cuda()


gt_disp = exr2hdr(gt_disp_file)
gt_disp = gt_disp.transpose([2, 0, 1])
gt_disp = gt_disp[np.newaxis, :, :, :]
gt_disp = torch.from_numpy(gt_disp).cuda()


gt_norm = exr2hdr(gt_norm_file)
gt_norm = gt_norm * 2.0 - 1.0
m = gt_norm >= 0
m[:,:,0] = False
m[:,:,1] = False
gt_norm[m] = -gt_norm[m]
gt_norm = gt_norm.transpose([2, 0, 1])
gt_norm = gt_norm[np.newaxis, :, :, :]
gt_norm = torch.from_numpy(gt_norm).cuda()


print ('gt mean ', gt_disp.mean())
print ('init mean', init_disp.mean())
print ('norm mean', gt_norm.mean())

print ('Ori EPE:', EPE(gt_disp, init_disp))
print ('Ori Variance :', (init_disp - gt_disp).std())


n, c, h, w = init_disp.size()
disp2norm = Disp2Norm(n, w, h, 480, 480) 


#print ('GT ', gt_disp)


disp = init_disp
times = 10
#disp = torch.ones([n, c, h, w]).cuda() * 100
#times = 10000
for t in range(times):
    #disp = disp + (gt_disp[0,0,0,0] - disp[0,0,0,0])
    #disp = disp2norm.norm_adjust_disp(disp, gt_norm, ori_ratio=0.5, edge_threshold=4)
    disp = disp2norm.norm_adjust_disp_vote(disp, gt_norm, ori_vote=1)
    #disp = disp + (gt_disp.mean() - disp.mean())
    print ('Adj EPE', t, ' : ', EPE(gt_disp, disp))
    print ('Variance :', (disp - gt_disp).std())
    #print ('Delta :', (disp - gt_disp).abs().mean())
    #print ('Mean :', disp.mean())
    #print ('Delta :', (disp - gt_disp))
    #print ('Disp ', disp)

    out_file = os.path.dirname(init_disp_file) + '/result_d_' + str(t) + '.exr'
    cpu_disp = disp.cpu().numpy().squeeze()
    cpu_disp = cpu_disp[:,:,np.newaxis]
    save_exr(cpu_disp, out_file)
    print ('out file: ', out_file)

    print ('')
    


