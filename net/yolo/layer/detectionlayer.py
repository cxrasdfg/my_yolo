# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn

from utility import t_meshgrid_2d,t_box_iou

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
        coords,\
        rescore,\
        side,\
        num,\
        softmax,\
        sqrt,\
        jitter,\
        object_scale,\
        noobject_scale,\
        class_scale,\
        coord_scale
        
    def forward(self,*args):
        r"""
        Args:
            if self.training:
                x (tensor[float32]): [b,f], a connected layer is before this layer...
                b_fixed_boxes (tensor[float32]): [b,MAX_BOX_NUM,4]
                b_fixed_labels (tensor[long]): [b,MAX_BOX_NUM]
                b_real_box_num (tensor[long]): [b], number of real boxes in element of batch
            else:
                x (tensor[float32]): [b,f]
        Return:
            if self.training:
                loss (tensor[float32]): the loss of it...
            else:
                
        """
        if self.training:
            x,b_fixed_boxes,\
            b_fixed_labels,\
            b_real_box_num=args
            # continue?
            # self.side*self.side*(self.num*(self.coord+self.rescore)+self.classes)
            
        else:

            raise NotImplementedError()