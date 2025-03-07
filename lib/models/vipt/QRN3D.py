import torch
import torch.nn as nn
import torch.nn.functional as FF
import numpy as np
from functools import partial
from .batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

BatchNorm3d = SynchronizedBatchNorm3d


class BasicConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, bias=False, bn=True, tau=128):
        super(BasicConv3d, self).__init__()
        self.add_module('conv1', nn.Conv3d(in_channels, in_channels // tau, 1, s, 0, bias=bias))
        self.add_module('conv2', nn.Conv3d(in_channels // tau, channels, (k,1,1), s, (p,0,0), bias=bias))

"""F pooling"""
class QRNN3DLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, conv_layer, act='tanh'):
        super(QRNN3DLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # quasi_conv_layer
        self.conv = conv_layer
        self.act = act

    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F = gates.split(split_size=self.hidden_channels, dim=1)
        if self.act == 'tanh':
            return Z.tanh(), F.sigmoid()
        elif self.act == 'relu': 
            return Z.relu(), F.sigmoid()
        elif self.act == 'none':
            return Z, F.sigmoid
        else:
            raise NotImplementedError

    def _rnn_step(self, z, f, h):
        # uses 'f pooling' at each time step
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, reverse=False):
        h = None
        Z, F = self._conv_step(inputs)
        h_time = []
        
        if not reverse:
            for time, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):  # split along timestep            
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for time, (z, f) in enumerate((zip(
                reversed(Z.split(1, 2)), reversed(F.split(1, 2))
                ))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)
        
        return h[:,:,0,:,:], torch.cat(h_time, dim=2)

    def extra_repr(self):
        return 'act={}'.format(self.act)

class QRNNConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh', tau=128):
        super(QRNNConv3D, self).__init__(
            in_channels, hidden_channels, BasicConv3d(in_channels, hidden_channels*2, k, s, p, bn=bn, tau=tau), act=act)


if __name__ == '__main__':
    model = QRNNConv3D(768, 768, tau=128)
    data = torch.rand(32, 768, 8, 8, 8)
    res_single, res_all = model(data)
