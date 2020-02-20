import re, sys, array
import torch
import torchvision.transforms as transforms
from skimage import transform
import random
import numpy as np
from torch import nn
import OpenEXR

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

#__imagenet_stats = {'mean': [0., 0., 0.],
#                   'std': [1, 1, 1]}

#__imagenet_stats = {'mean': [0., 0., 0.],
#                   'std': [255, 255, 255]}

#__imagenet_stats = {'mean': [0.5, 0.5, 0.5],
#                   'std': [0.5, 0.5, 0.5]}

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

pca_param = dict(__imagenet_pca)


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    #if scale_size != input_size:
    #t_list = [transforms.Scale((960,540))] + t_list

    return transforms.Compose(t_list)


def scale_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Scale(scale_size)] + t_list

    transforms.Compose(t_list)


def pad_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    padding = int((scale_size - input_size) / 2)
    return transforms.Compose([
        transforms.RandomCrop(input_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])


def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])

def inception_color_preproccess(input_size=256, normalize=__imagenet_stats):
    return transforms.Compose([
        #transforms.RandomSizedCrop(input_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(**normalize)
    ])


def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None, augment=True):
    normalize = __imagenet_stats
    input_size = 256
    if augment:
            return inception_color_preproccess(input_size, normalize=normalize)
    else:
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)

def default_transform(input_size=None, 
                  scale_size=None, normalize=None, augment=True):

    normalize = __imagenet_stats

    rgb_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]

    disp_list = [
        transforms.ToTensor(),
    ]

    return transforms.Compose(rgb_list)

def scale_transform(input_size=None, 
                  scale_size=(576, 960), normalize=None, augment=True):
    normalize = __imagenet_stats

    scale_list = [
        transforms.Resize(scale_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]

    return transforms.Compose(scale_list)


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))

class RandomRescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image_left, image_right, gt_disp = sample['img_left'], sample['img_right'], sample['gt_disp']

        #h, w = image_left.shape[:2]
        #if isinstance(self.output_size, int):
        #out_h, out_w = self.output_size
        #    if h > w:
        #        new_h, new_w = self.output_size * h / w, self.output_size
        #    else:
        #        new_h, new_w = self.output_size, self.output_size * w / h
        #else:
        #    new_h, new_w = self.output_size

        #new_h, new_w = int(new_h), int(new_w)

        image_left = transform.resize(image_left, self.output_size, preserve_range=True)
        image_right = transform.resize(image_right, self.output_size, preserve_range=True)

        # change image pixel value type ot float32
        image_left = image_left.astype(np.float32)
        image_right = image_right.astype(np.float32)
        gt_disp = gt_disp.astype(np.float32)
        new_sample = sample
        new_sample.update({'img_left': image_left, 
                      'img_right': image_right, 
                      'gt_disp': gt_disp})
        return new_sample

# disp is 4D tensor
def scale_disp(disp, output_size=(1, 540, 960)):
    #print('current shape:', disp.size())
    i_w = disp.size()[-1]
    o_w = output_size[-1]

    ## Using sklearn.transform
    #trans_disp = disp.squeeze(1).data.cpu().numpy()
    #trans_disp = transform.resize(trans_disp, output_size, preserve_range=True).astype(np.float32)
    #trans_disp = torch.from_numpy(trans_disp).unsqueeze(1).cuda()
    
    # Using nn.Upsample
    m = nn.Upsample(size=(output_size[-2], output_size[-1]), mode="bilinear")
    trans_disp = m(disp)

    trans_disp = trans_disp * (o_w * 1.0 / i_w)
    return trans_disp

# norm is 4D tensor, [:, :3, :, :] is normal, [:, 3, :, :] is disp
def scale_norm(norm, output_size=(1, 4, 540, 960), normalize=True):
    # print('current shape:', disp.shape)
    #_output_size = (output_size[0] * output_size[1], output_size[2], output_size[3])
    input_size = norm.shape
    #norm = np.reshape(norm, [input_size[0] * input_size[1], input_size[2], input_size[3]])
    i_w = input_size[-1]
    o_w = output_size[-1]
    #trans_norm = transform.resize(norm, _output_size, preserve_range=True)
    #trans_norm = np.reshape(trans_norm, [output_size[0], output_size[1], output_size[2], output_size[3]]) 

    #trans_disp = trans_norm[:,3:,:,:]
    #trans_norm = trans_norm[:,:3,:,:]

    #trans_disp = trans_disp * (o_w * 1.0 / i_w)
    #if normalize:
    #    scalar = np.linalg.norm(trans_norm, axis=1, keepdims=True)
    #    is_valid = (scalar > 0)
    #    scalar[is_valid] = 1.0 / scalar[is_valid]
    #    trans_norm = trans_norm * scalar

    #trans_norm = np.concatenate((trans_norm, trans_disp), 1)

    ## print('trans shape:', trans_disp.shape)
    #return trans_norm.astype(np.float32)

    m = nn.Upsample(size=(int(output_size[-2]), int(output_size[-1])), mode="bilinear")
    norm_disp = m(norm)
    if norm_disp.size()[1] == 4:
        norm_disp[:, -1, :, :] = norm_disp[:, -1, :, :] * (o_w * 1.0 / i_w)

    return norm_disp

# disp_angle is 4D tensor, [:, :2, :, :] is angle, [:, 2, :, :] is disp
def scale_angle(disp_angle, output_size=(1, 4, 540, 960)):
    # print('current shape:', disp.shape)
    #_output_size = (output_size[0] * output_size[1], output_size[2], output_size[3])
    input_size = disp_angle.shape
    #norm = np.reshape(norm, [input_size[0] * input_size[1], input_size[2], input_size[3]])
    i_w = input_size[3]
    o_w = output_size[3]
    #trans_norm = transform.resize(norm, _output_size, preserve_range=True)
    #trans_norm = np.reshape(trans_norm, [output_size[0], output_size[1], output_size[2], output_size[3]]) 

    #trans_disp = trans_norm[:,3:,:,:]
    #trans_norm = trans_norm[:,:3,:,:]

    #trans_disp = trans_disp * (o_w * 1.0 / i_w)
    #if normalize:
    #    scalar = np.linalg.norm(trans_norm, axis=1, keepdims=True)
    #    is_valid = (scalar > 0)
    #    scalar[is_valid] = 1.0 / scalar[is_valid]
    #    trans_norm = trans_norm * scalar

    #trans_norm = np.concatenate((trans_norm, trans_disp), 1)

    ## print('trans shape:', trans_disp.shape)
    #return trans_norm.astype(np.float32)

    m = nn.Upsample(size=(540, 960), mode="bilinear")
    norm_disp = m(disp_angle)
    norm_disp[:, -1, :, :] = norm_disp[:, -1, :, :] * (o_w * 1.0 / i_w)

    return norm_disp


class RandomCrop(object):
    """
    Crop the image randomly
    Args: int or tuple. tuple is (h, w)

    """
    def __init__(self, output_size, augment=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.augment = augment
        self.transform = ColorJitter() 

    def __call__(self, sample):
        image_left, image_right, gt_disp = sample['img_left'], sample['img_right'], sample['gt_disp']

        h, w = image_left.shape[1:3]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # top = 0
        # left = 0

        image_left = image_left[:, top: top + new_h, left: left + new_w]
        image_right = image_right[:, top: top + new_h, left: left + new_w]
        gt_disp = gt_disp[:, top: top + new_h, left: left + new_w]
        if self.augment:
            rd = np.random.randint(0,2)
            if rd == 0:
                image_left = self.transform(image_left)
                #imgtmp = image_left.cpu().numpy()
                #imgtmp = np.transpose(imgtmp, [2, 1, 0])
                #print('lighted shape:', imgtmp.shape)
                #io.imsave('test.png', imgtmp)
                image_right = self.transform(image_right)
        new_sample = sample
        new_sample.update({'img_left': image_left, 
                      'img_right': image_right, 
                      'gt_disp': gt_disp})

        return new_sample

class CenterCrop(object):
    """
    Crop the image at center
    Args: int or tuple. tuple is (h, w)

    """
    def __init__(self, output_size, augment=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.augment = augment
        self.transform = ColorJitter() 

    def __call__(self, sample):
        image_left, image_right, gt_disp = sample['img_left'], sample['img_right'], sample['gt_disp']

        h, w = image_left.shape[1:3]
        new_h, new_w = self.output_size

        top = int((h - new_h) / 2)
        left = int((w - new_w) / 2)
        # top = 0
        # left = 0

        image_left = image_left[:, top: top + new_h, left: left + new_w]
        image_right = image_right[:, top: top + new_h, left: left + new_w]
        gt_disp = gt_disp[:, top: top + new_h, left: left + new_w]
        if self.augment:
            rd = np.random.randint(0,2)
            if rd == 0:
                image_left = self.transform(image_left)
                #imgtmp = image_left.cpu().numpy()
                #imgtmp = np.transpose(imgtmp, [2, 1, 0])
                #print('lighted shape:', imgtmp.shape)
                #io.imsave('test.png', imgtmp)
                image_right = self.transform(image_right)
        new_sample = sample
        new_sample.update({'img_left': image_left, 
                      'img_right': image_right, 
                      'gt_disp': gt_disp})

        return new_sample

class ToTensor(object):

    def __call__(self, array):
        # image_left, image_right, gt_disp = sample['img_left'], sample['img_right'], sample['gt_disp']

        # image_left = image_left.transpose((2, 0, 1))
        # image_right = image_right.transpose((2, 0, 1))
        # gt_disp = gt_disp[np.newaxis, :]

        # new_sample = {'img_left': torch.from_numpy(image_left), \
        #               'img_right': torch.from_numpy(image_right), \
        #               'gt_disp': torch.from_numpy(gt_disp.copy()) \
        #               }
        # return new_sample
        if len(array.shape) == 3 and (array.shape[2] == 3 or array.shape[2] == 4):
            array = np.transpose(array, [2, 0, 1])
        if len(array.shape) == 2:
            array = array[np.newaxis, :]

        tensor = torch.from_numpy(array.copy())
        return tensor.float()


'''
Save a Numpy array to a PFM file.
'''
def save_pfm(filename, image, scale = 1):
  file = open(filename, 'w')
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)

  image.tofile(file) 
  file.close()


'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
def load_pfm(filename):

    file = open(filename, 'r')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True    
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    file.close()
    return np.reshape(data, shape), scale

def save_exr(img, filename):
    c, h, w = img.shape
    if c == 1:
        img = img.reshape(w * h)
        Gs = array.array('f', img).tostring()
        
        out = OpenEXR.OutputFile(filename, OpenEXR.Header(w, h))
        out.writePixels({'G' : Gs })
    else:
        data = np.array(img).reshape(c, w * h)
        Rs = array.array('f', data[0,:]).tostring()
        Gs = array.array('f', data[1,:]).tostring()
        Bs = array.array('f', data[2,:]).tostring()
    
        out = OpenEXR.OutputFile(filename, OpenEXR.Header(w, h))
        out.writePixels({'R' : Rs, 'G' : Gs, 'B' : Bs })

