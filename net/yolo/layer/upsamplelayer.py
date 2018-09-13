# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn

class UpsampleLayer(nn.Module):
    def __init__(self,stride=2):
        super(UpsampleLayer, self).__init__()
        self.stride=stride

    def forward(self, x):
        batch=x.shape[0]
        channel=x.shape[1]
        height=x.shape[2]
        width=x.shape[3]

        # wrong usage: x.view(batch,channel,1,width,1,height)...????????
        x=x.view(batch,
               channel,
               height,1,
               width,1).expand(batch,
                              channel,
                              height,
                              self.stride,
                              width,
                              self.stride).contiguous().view(batch,
                                                        channel,
                                                        self.stride*height,
                                                        self.stride*width)
        return x