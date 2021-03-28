from __future__ import print_function

from networks.DispNetC import DispNetC
from networks.DispNetCSS import DispNetCSS
from networks.DNFusionNet import DNFusionNet
from networks.DToNNet import DToNNet
from networks.NormNetS import NormNetS
from networks.FADNet import FADNet
from networks.gwcnet import GwcNet

from utils.common import logger

SUPPORT_NETS = {
        'dispnetc': DispNetC,
        'dispnetcss': DispNetCSS,
        'fadnet': FADNet,
        'dnfusionnet':DNFusionNet,
        'dtonnet':DToNNet,
        'normnets':NormNetS,
        'gwcnet':GwcNet,
        }

def build_net(net_name):
    net  = SUPPORT_NETS.get(net_name, None)
    if net is None:
        logger.error('Current supporting nets: %s , Unsupport net: %s', SUPPORT_NETS.keys(), net_name)
        raise 'Unsupport net: %s' % net_name
    return net
