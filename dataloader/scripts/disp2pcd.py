import numpy as np
import EXRloader
import readpfm
import cv2
import pclpy
from pclpy import pcl
from skimage import io, transform

dir_path = './'
input_disp = 'IRS_Store_l_224_d.exr'
input_rgb = 'l_224.png'
input_normal = 'IRS_Store_l_224_n.exr'
#input_normal = None
output_ply = 'ply/store_pcd_gt.ply'

disp_inverse_y = True

focus_length = 480
baseline = 0.1
max_distance = 50.0

compute_normal_radius = 0.2


print (dir_path + input_disp)
ext = input_disp.split('.')[-1]
if ext == 'pfm':
	disp, w, h, c = readpfm.load_pfm(dir_path + input_disp)
	disp = np.array(disp).reshape(w, h, c)
else:
	disp = EXRloader.exr2hdr(dir_path + input_disp)
	h, w, c = disp.shape
	if c == 3:
		disp = disp[:,:,1]
disp = disp.reshape(h, w)
if disp_inverse_y:
	disp = np.flip(disp, axis=0)

if input_normal is not None:
	print (dir_path + input_normal)
	normal = EXRloader.exr2hdr(dir_path + input_normal)
	normal = normal * 2.0 - 1.0

if input_rgb is not None:
	print (dir_path + input_rgb)
	rgb = io.imread(dir_path + input_rgb)
	if len(rgb.shape) == 2:
		rgb = rgb[:,:,np.newaxis]
		rgb = np.pad(rgb, ((0, 0), (0, 0), (0, 2)), 'constant')
		rgb[:,:,1] = rgb[:,:,0]
		rgb[:,:,2] = rgb[:,:,0]
	_h, _w, c = rgb.shape
	if c == 4:
		rgb = rgb[:,:,:3]
	if _w != w or _h != h:
		dim = (w, h)
		rgb = cv2.resize(rgb, dim, interpolation = cv2.INTER_AREA)


print ('Image shape: ', (w, h))


cx = (w - 1) * 0.5
cy = (h - 1) * 0.5
x = np.arange(0, w)
y = np.arange(0, h)
coord_x, coord_y = np.meshgrid(x, y)

coord_x = coord_x - cx
coord_y = coord_y - cy

pz = focus_length * baseline / disp
pz = np.clip(pz, -max_distance, max_distance)
px = coord_x * pz / focus_length
py = -coord_y * pz / focus_length
pz = -pz


np_points = np.stack((px, py, pz), axis=2)
np_points = np_points.reshape(w*h, 3)

if input_normal is not None:
	np_normal = normal.reshape(w*h, 3)
	np_normal[:,1] = -np_normal[:,1]
	np_normal[:,2] = -np_normal[:,2]

if input_rgb is not None:
	np_rgb = rgb.reshape(w*h, 3)
	points = pcl.PointCloud.PointXYZRGB.from_array(np_points, np_rgb)
else:
	points = pcl.PointCloud.PointXYZ.from_array(np_points)

if input_normal is None:
	points = points
	normals = points.compute_normals(radius=compute_normal_radius)
	np_normal = normals.normals
	
np_points = np.concatenate((np_points, np_normal), axis=1)
points = pcl.PointCloud.PointXYZRGBNormal(np_points, np_rgb)

viewer=pcl.visualization.PCLVisualizer('PCD viewer')
viewer.addPointCloud(points)
while(not viewer.wasStopped()):
	viewer.spinOnce(100)


pcl.io.savePLYFile(dir_path + output_ply, points)










