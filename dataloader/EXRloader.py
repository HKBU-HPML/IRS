import numpy
import OpenEXR
import Imath
import imageio
import glob
import os

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

def writehdr(hdrpath,hdr):
	h, w, c = hdr.shape
	if c == 1:
		hdr = numpy.pad(hdr, ((0, 0), (0, 0), (0, 2)), 'constant')
		hdr[:,:,1] = hdr[:,:,0]
		hdr[:,:,2] = hdr[:,:,0]
	imageio.imwrite(hdrpath,hdr,format='hdr')

def load_exr(filename):
	hdr = exr2hdr(filename)
	h, w, c = hdr.shape
	if c == 1:
		hdr = numpy.squeeze(hdr)
	return hdr


def test_exr():
	files = glob.glob('D:/MLProjects/data/home/*.exr')
	savepath = 'D:/MLProjects/data/home'
	total = len(files)
	count = 0
	print ('Files Num:', total)	
	for file in files:
	    hdr = exr2hdr(file)
	    filename,file_ext = os.path.splitext(file)
	    filename = os.path.basename(filename)
	    filename = filename + '.hdr'
	    curpath = os.path.join(savepath,filename)
	    writehdr(curpath,hdr)
	    count = count + 1
	    print ('process:', count, '/', total)

if __name__ == '__main__':
    test_exr()
