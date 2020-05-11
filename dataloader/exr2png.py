from EXRloader import load_exr
import skimage

i_name = "IRS_H_AV03_N_l_603_d.exr"
o_name = "IRS_H_AV03_N_l_603_d_gt.png"

data = load_exr(i_name)
skimage.io.imsave(o_name,(data*255).astype('uint16'))
