import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from functools import partial
import math
import torch.nn.init as init

config_task = {
    'mode': 'parallel_adapters', # series_adapters
    'isdropout1': True,
}
tau=64

def conv1x1_fonc(planes, out_planes=None, stride=1, bias=False):
    if out_planes is None:
        return nn.Sequential(
            nn.Linear(planes, planes // tau, bias=bias),
            nn.Linear(planes // tau, planes, bias=bias))
    else:
        return nn.Sequential(
            nn.Linear(planes, out_planes // tau, bias=bias),
            nn.Linear(planes // tau, out_planes, bias=bias))

class conv1x1(nn.Module):
    
    def __init__(self, planes, out_planes=None, stride=1):
        super(conv1x1, self).__init__()
        if config_task['mode'] == 'parallel_adapters':
            self.conv = conv1x1_fonc(planes, out_planes, stride) 
        else:
            self.conv = conv1x1_fonc(planes)
    def forward(self, x):
        y = self.conv(x)
        if config_task['mode'] == 'series_adapters':
            y += x
        return y

class Jing_conv_task(nn.Module):
    def __init__(self, planes, stride=1, nb_tasks=1, is_proj=1, second=0, norm_init=False):
        super(Jing_conv_task, self).__init__()
        self.is_proj = is_proj
        if config_task['mode'] == 'parallel_adapters' and is_proj:
            self.parallel_conv = nn.ModuleList([conv1x1(planes, planes, stride) for i in range(nb_tasks)])
            self.bns = nn.ModuleList([nn.LayerNorm(planes, eps=1e-6) for i in range(nb_tasks)])
        else:
            self.bns = nn.ModuleList([nn.LayerNorm(planes, eps=1e-6) for i in range(nb_tasks)])

    def forward(self, x, norm_mode=None):
        y = x
        y = y + self.parallel_conv[norm_mode](x)
        y = self.bns[norm_mode](y)
        return y
