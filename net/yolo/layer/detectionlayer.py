# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn

class DetectionLayer(torch.nn.Module):
    def __init__(self,classes,
        coords,
        rescore,
        side,
        num,
        softmax,
        sqrt,
        jitter,
        object_scale,
        noobject_scale,
        class_scale,
        coord_scale):
        
        super(DetectionLayer,self).__init__()
        
        self.coords,\
        self.rescore,\
        self.side,\
        self.num,\
        self.softmax,\
        self.sqrt,\
        self.jitter,\
        self.object_scale,\
        self.noobject_scale,\
        self.class_scale,\
        self.coord_scale=\
        coords,
        rescore,
        side,
        num,
        softmax,
        sqrt,
        jitter,
        object_scale,
        noobject_scale,
        class_scale,
        coord_scale
        
    def forward(self,x):
        raise NotImplementedError()