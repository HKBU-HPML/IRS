from __future__ import print_function
import re
import numpy as np
import sys
import array
from struct import *

def load_pfm(pfmpath):
    img = np.array([])
    width, height, channels = 0, 0, 0
    debug = False
    with open(pfmpath, 'rb') as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type=f.readline().decode('latin-1')
        if "PF" in type:
            channels=3
        elif "Pf" in type:
            channels=1
        else:
            print("ERROR: Not a valid PFM file",file=sys.stderr)
            sys.exit(1)
        if(debug):
            print("DEBUG: channels={0}".format(channels))

        # Line 2: width height
        line=f.readline().decode('latin-1')
        width,height=re.findall('\d+',line)
        width=int(width)
        height=int(height)
        if(debug):
            print("DEBUG: width={0}, height={1}".format(width,height))

        # Line 3: +ve number means big endian, negative means little endian
        line=f.readline().decode('latin-1')
        BigEndian=True
        if "-" in line:
            BigEndian=False
        if(debug):
            print("DEBUG: BigEndian={0}".format(BigEndian))

        # Slurp all binary data
        samples = width*height*channels;
        buffer  = f.read(samples*4)

        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt=">"
        else:
            fmt="<"
        fmt= fmt + str(samples) + "f"
        img = array.array('f', unpack(fmt,buffer))
    return img, width, height, channels
