from .local import Conv2dLocal as locally_layer

import torch

class Conv2dLocal(torch.nn.Module):
    def __init__(self,in_height, in_width, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True, dilation=1,
                 act=None):
        super(Conv2dLocal,self).__init__()
        self.loc=locally_layer(in_height, in_width, in_channels, out_channels,
                 kernel_size, stride, padding, bias, dilation)
        if act=='linear':
            self.act=None
        elif act=='leaky':
            self.act=torch.nn.LeakyReLU(0.1,inplace=True)
        else:
            raise ValueError('...')

    def forward(self, x):
        x=self.loc(x)
        if self.act:
            x=self.act(x)
        return x