from __future__ import print_function

#from networks.simple_net import SimpleNet
#from networks.dispnet_corr2 import DispNetCorr2
from networks.DispNetC import DispNetC
from networks.DispNetS import DispNetS
from networks.DispNetCS import DispNetCS
from networks.DispNetCSS import DispNetCSS
from networks.DispNetCSRes import DispNetCSRes
from networks.DispNormNet import DispNormNet
from networks.DNFusionNet import DNFusionNet
from networks.DToNNet import DToNNet
from networks.DToNFusionNet import DToNFusionNet
from networks.DNIRRNet import DNIRRNet
from networks.DispAngleNet import DispAngleNet
from networks.DToNNet import DToNNet
from networks.stackhourglass import PSMNet
#from networks.GANet_deep import GANet
from networks.NormNetS import NormNetS
#from networks.NormNetC import NormNetC
#from networks.stackhourglass import PSMNet
#from networks.GANet_deep import GANet
#from networks.MultiCorrNet import MultiCorrNet

from utils.common import logger

SUPPORT_NETS = {
        #'simplenet': SimpleNet,
        'dispnetcres': DispNetCSRes,
        'dispnetc': DispNetC,
        'dispnets': DispNetS,
        'dispnetcs': DispNetCS,
        'dispnetcss': DispNetCSS,
        'psmnet': PSMNet,
        #'ganet':GANet,
        'dispnormnet':DispNormNet,
        'dnfusionnet':DNFusionNet,
        'dtonnet':DToNNet,
        'dtonfusionnet':DToNFusionNet,
        'dnirrnet':DNIRRNet,
        'dispanglenet':DispAngleNet,
        'normnets':NormNetS,
	'dtonnet':DToNNet,
        #'normnetc':NormNetC,
        #'multicorrnet': MultiCorrNet,
        #'dispnetcorr2': DispNetCorr2,
        }

def build_net(net_name):
    net  = SUPPORT_NETS.get(net_name, None)
    if net is None:
        logger.error('Current supporting nets: %s , Unsupport net: %s', SUPPORT_NETS.keys(), net_name)
        raise 'Unsupport net: %s' % net_name
    return net
