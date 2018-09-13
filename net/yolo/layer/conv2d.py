# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn

class Conv2D(nn.Module):
    def __init__(self,input_channel,output_channel,
                 size,stride,same_padding,bn,activation):
        
        super(Conv2D, self).__init__()

        if bn:
            bias=False
        else:
            bias=True
        padding=int(size-1)//2 if same_padding else 0
        self.conv=nn.Conv2d(input_channel,output_channel,
                            size,stride,padding,bias=bias)
        self.bn=nn.BatchNorm2d(output_channel) if bn else None
        if activation=='linear':
            self.act=None
        elif activation=='leaky':
            self.act=nn.LeakyReLU(0.1,inplace=True)
        else:
            raise ValueError('Specify `linear` or `leaky` for activation')

    def forward(self, x):
        x=self.conv(x)
        if self.bn:
            x=self.bn(x)
        if self.act is not None:
            x=self.act(x)
        return x