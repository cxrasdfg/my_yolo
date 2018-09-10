# coding=utf-8
import torch
from torchvision.models import vgg16
from config import cfg

class L2Norm(torch.nn.Module):
    def __init__(self,channels, scale_):
        super(L2Norm,self).__init__()

        # one channel for on scale..., when update, this weight will be updated synchorinized
        self.weights=torch.nn.Parameter(torch.Tensor(channels)) # [c]
        
        # use scale_ to initialize the weight, default scale_ is 20...
        torch.nn.init.constant_(self.weights.data,scale_)

    
    def forward(self,x):
        # x: [b,c,h,w]
        norm=(x**2).sum(dim=1,keepdim=True).sqrt() # [b,1,h,w]
        x=x/norm.expand_as(x) # [b,c,h,w]

        # use weigths to scale the norm features
        x=self.weights[None][...,None][...,None].expand_as(x)*x

        return x


def caffe_vgg16():
    r"""[2,m1,2,m2,3,m3,3,m4,3,m5,fc6,fc7,fc8]
    """
    if cfg.use_caffe:
        base_model=vgg16(False)
        base_model.load_state_dict(torch.load(cfg.caffe_model) )
    else:
        base_model=vgg16(True)
    # open ceil mode
    for _ in list(base_model.features):
        if isinstance(_,torch.nn.MaxPool2d):
            _.ceil_mode=True

    # 21 is Conv4-3, 30 is the max pooling
    # 23 is the max pooling..., must use 23 since 22 is relu of Conv4-3
    features=list(base_model.features)[:30] # list...
    conv4_3=features[:23]
    conv5_3=features[23:]

    # add the L2-Norm behind the conv4-3  
    # conv4_3+=[L2Norm(conv4_3[-2].out_channels,cfg.l2norm_scale)]

    # change the pool from 2x2-s2 to 3x3-s1
    conv5_3+=[torch.nn.MaxPool2d(3,1,padding=1,dilation=1,ceil_mode=True)]

    # freeze top4 conv\
    if cfg.freeze_top:
        for layer in conv4_3[:23]:
            for p in layer.parameters():
                p.requires_grad = False	

        for layer in conv5_3:
            for p in layer.parameters():
                p.requires_grad=False	
    conv4_3=torch.nn.Sequential(*conv4_3)
    conv5_3=torch.nn.Sequential(*conv5_3)

    # conv6
    conv6=torch.nn.Sequential(*[
        torch.nn.Conv2d(conv5_3[-3].out_channels,1024,3,dilation=6,padding=6),
        torch.nn.ReLU(inplace=True)
    ])

    # conv7
    conv7=torch.nn.Sequential(*[
        torch.nn.Conv2d(1024,1024,1),
        torch.nn.ReLU(inplace=True)
    ])

    # 0,3,6 is fc6,fc7,fc8 in classifier
    fc6=base_model.classifier[0]
    fc7=base_model.classifier[3]

    # subsample the parameters for conv6 and conv7
    
    # conv6
    # weight
    conv6[0].weight.data.view(-1).copy_(fc6.weight.data.view(-1)[:conv6[0].weight.numel()])
    # bias
    conv6[0].bias.data.view(-1).copy_(fc6.bias.data.view(-1)[:conv6[0].bias.numel()])

    # conv7
    # weight
    conv7[0].weight.data.view(-1).copy_(fc7.weight.data.view(-1)[:conv7[0].weight.numel()])   
    # bias
    conv7[0].bias.data.view(-1).copy_(fc7.bias.data.view(-1)[:conv7[0].bias.numel()])

    del base_model.classifier # drop the classifier

    return conv4_3,conv5_3,conv6,conv7
    
