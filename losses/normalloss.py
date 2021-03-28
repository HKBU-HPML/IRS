#!/usr/local/bin/python3
import numpy
import OpenEXR
import Imath
import imageio
import glob
import os
import sys
import re
from struct import *
import array
import numpy as np
import math
import torch


in_gt_norm = '/root/data/ue_generated_data'
in_result_norm = '/root/data/dispnormnet_ue_separated'
#in_result_norm = 'D:/MLProjects/data/detect_results/norm_test2/ue_generated_data_restaurant_ContemporaryRestaurant2_l_62_n.exr'


def exr2hdr(exrpath):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    if (CNum > 1):
        Channels = ['R', 'G', 'B']
        CNum = 3
    else:
        Channels = ['G']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [numpy.fromstring(File.channel(c, PixType), dtype=numpy.float32) for c in Channels]
    hdr = numpy.zeros((Size[1],Size[0],CNum),dtype=numpy.float32)
    if (CNum == 1):
        hdr[:,:,0] = numpy.reshape(Pixels[0],(Size[1],Size[0]))
    else:
        hdr[:,:,0] = numpy.reshape(Pixels[0],(Size[1],Size[0]))
        hdr[:,:,1] = numpy.reshape(Pixels[1],(Size[1],Size[0]))
        hdr[:,:,2] = numpy.reshape(Pixels[2],(Size[1],Size[0]))
    return hdr

def correct_normal(normal):
    m = normal > 0
    m[:,:,0] = False
    m[:,:,1] = False
    normal[m] = - normal[m]
    return normal

def decode_normal(normal):
    normal = normal * 2 - 1
    normal = correct_normal(normal)
    return normal

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def vector_normalize(vectors):
    scalar = np.linalg.norm(vectors, axis=1, keepdims=True)
    #is_valid = (scalar > 0)
    #scalar[is_valid] = 1.0 / scalar[is_valid]
    scalar = 1.0 / scalar
    vectors = vectors * scalar
    return vectors

def angle_diff_angle(res_angle, gt_angle):

    size = res_angle.size()
    res_norm = torch.ones(size[0], 3, size[2], size[3])
    gt_norm = torch.ones(size[0], 3, size[2], size[3])

    res_norm[:, 0, :, :] = torch.tan(res_angle[:, 0, :, :])
    res_norm[:, 1, :, :] = torch.tan(res_angle[:, 1, :, :])

    gt_norm[:, 0, :, :] = torch.tan(gt_angle[:, 0, :, :])
    gt_norm[:, 1, :, :] = torch.tan(gt_angle[:, 1, :, :])

    return angle_diff_norm(res_norm, gt_norm)

    #diff_x_angle = torch.abs(res_angle[:, 0, :, :] - gt_angle[:, 0, :, :])
    #diff_y_angle = torch.abs(res_angle[:, 1, :, :] - gt_angle[:, 1, :, :])

    #print "mean of diff_x: %f.(%f-%f)" % (torch.mean(diff_x_angle) * 180.0 / math.pi, torch.min(diff_x_angle) * 180.0 / math.pi, torch.max(diff_x_angle) * 180.0 / math.pi)
    #print "mean of diff_y: %f.(%f-%f)" % (torch.mean(diff_y_angle) * 180.0 / math.pi, torch.min(diff_y_angle) * 180.0 / math.pi, torch.max(diff_y_angle) * 180.0 / math.pi)

    #angle_diff = torch.acos(torch.cos(diff_x_angle) * torch.cos(diff_y_angle)) * 180.0 / math.pi

    #return angle_diff

# inputs are 4-D numpy array (B, C, H, W)
def angle_diff_norm(res_norm, gt_norm):
    ##res_norm = decode_normal(res_norm)
    ##gt_norm = decode_normal(gt_norm)  
        ##print res_norm.shape, gt_norm.shape

    #res_norm = vector_normalize(res_norm)
    #gt_norm = vector_normalize(gt_norm)

    #delta = gt_norm - res_norm
    #delta = delta**2
    #l = np.sum(delta, axis=1)
        #print l.shape
        #print(np.max(l), np.min(l), np.mean(l))
    #alpha = np.arccos(1 - l * 0.5)
 
    ##angle = np.mean(alpha)
    #angle = alpha / math.pi * 180

    #return angle

    res_norm = res_norm / torch.norm(res_norm, 2, dim=1, keepdim=True)
    gt_norm = gt_norm / torch.norm(gt_norm, 2, dim=1, keepdim=True)

    delta = (gt_norm - res_norm) ** 2
    l = torch.sum(delta, dim=1)
    
    alpha = torch.acos(1 - l * 0.5)
    angle = alpha / math.pi * 180.0
    return angle

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print ('python CalcMeanDeltaNormalAngle.py [ResultDir] [GTBaseDir]')
        print ('example: ')
        print ('    python CalcMeanDeltaNormalAngle.py detect_results/dispnormnet_ue_separated C:/Data/SIRSDataset')

    gt_norm = exr2hdr(in_gt_norm)
    res_norm = exr2hdr(in_result_norm)

    diff = angle_diff_norm(res_norm, gt_norm)
    print ('Angle Diff', diff)
