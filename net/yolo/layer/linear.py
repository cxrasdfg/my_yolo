# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self,in_features,out_features,act,bn=False):
        super(Linear,self).__init__()
        self.lin=torch.nn.Linear(in_features,out_features,bias=not bn)
        
        self.bn=torch.nn.BatchNorm2d(out_features) if bn else None

        if act == 'linear':
            self.act=None
        elif act=='leaky':
            self.act=nn.LeakyReLU(0.1,inplace=True)
        else:
            raise ValueError('attr::`act` must be `linear` or `leaky`')
    
    def forward(self,x):
        x=self.lin(x)
        if self.bn:
            x=self.bn(x)
        if self.act:
            x=self.act(x)
        return x