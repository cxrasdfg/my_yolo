# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 

class YoloLayer(torch.nn.Module):

    def __init__(self,mask,anchors,classes,num,jitter,
                 ignore_thresh,truth_thresh,random):
        super(YoloLayer, self).__init__()
        self.mask=mask
        self.anchors=anchors
        self.classes=classes
        self.num=num
        self.jitter=jitter
        self.ignore_thresh=ignore_thresh
        self.truth_thresh=truth_thresh
        self.random=random

        self.size_of_anchor=len(self.anchors)//self.num
        self.masked_anchor=[[self.anchors[i*self.size_of_anchor],
                             self.anchors[i*self.size_of_anchor+1]] for i in self.mask]

    def forward(self, x,net_height,net_width):
        batch,channel,height,width=x.shape

        m_anchor_num=len(self.masked_anchor)

        # transfer to batch * m_anchor_num * (width x height) * (5 + classes)
        x=x.view(
            [batch,channel,width*height]
        ).view(
            [batch,m_anchor_num,(5+self.classes),-1]
        ).transpose(2,3)

        coor_xy=x[:,:,:,:2]
        coor_wh=x[:,:,:,2:4]
        truth=x[:,:,:,4:5]
        class_prob=x[:,:,:,5:]

        # coor_xy=F.sigmoid(coor_xy)
        # truth=F.sigmoid(truth)
        # class_prob=F.sigmoid(class_prob)

        coor_xy=coor_xy.sigmoid()
        truth=truth.sigmoid()
        class_prob=class_prob.sigmoid()

        grid_x=torch.linspace(0,width-1,width).cuda()
        grid_y=torch.linspace(0,height-1,height).cuda()

        grid_x=grid_x.expand(
            [batch,m_anchor_num,height,width]
        ).contiguous().view(
            [batch,m_anchor_num,-1]
        )

        grid_y=grid_y.expand(
            [batch,m_anchor_num,height,width]
        ).transpose(2,3).contiguous().view(
            [batch,m_anchor_num,-1]
        )

        # add offset
        # coor_xy[:,:,:,0]+=grid_x
        # coor_xy[:,:,:,1]+=grid_y

        grid_xy=torch.cat(
            [grid_x.view(batch, m_anchor_num, -1, 1)
                , grid_y.view(batch, m_anchor_num, -1, 1)
             ]
            , dim=3
        )
        coor_xy+=grid_xy
        # renormalize
        # coor_xy[:,:,:,0]/=width
        # coor_xy[:,:,:,1]/=height
        coor_xy/=torch.tensor([int(width),int(height)]).type_as(coor_xy)

        #anchor
        m_anchor=torch.tensor(self.masked_anchor).type_as(coor_xy)

        # normalization
        m_anchor[:,0]/=net_width
        m_anchor[:,1]/=net_height

        # m_anchor=NP2Variable(m_anchor)
        m_anchor=m_anchor.expand(
            [batch,m_anchor_num,2]
        ).contiguous().view(
            [batch, m_anchor_num,1, 2]
        ).expand(batch,m_anchor_num,width * height,2)
        coor_wh=torch.exp(coor_wh)*m_anchor

        x=torch.cat([coor_xy,coor_wh,truth,class_prob],dim=3)

        return x