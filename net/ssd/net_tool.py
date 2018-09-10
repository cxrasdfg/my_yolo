# coding=utf-8

import torch
from torch.autograd import Variable
from config import cfg
import numpy as np

def encode_box(real_boxes,anchor_boxes):
    """Encode the real_box to the corresponding parameterized coordinates
    Args:
        real_boxes (tensor):[N,4], whose format is `xyxy`
        anchor_boxes (tensor):[N,4], i-th anchor is responsible for the i-th real box,
    and it's format is `xyxy`
    Return:
        parameterized boxes (tensor): [N,4]
    """
    assert real_boxes.shape==anchor_boxes.shape,'`real_boxes.shape` must be the same sa the `anchor_boxes`'
    if real_boxes.is_cuda and not anchor_boxes.is_cuda:
        anchor_boxes=anchor_boxes.cuda(real_boxes.device.index)
    assert anchor_boxes.is_cuda == anchor_boxes.is_cuda
    
    # change the boxes to `ccwh`
    real_boxes=xyxy2ccwh(real_boxes,inplace=False)
    anchor_boxes=xyxy2ccwh(anchor_boxes,inplace=False)

    encoded_xy=(real_boxes[:,:2]-anchor_boxes[:,:2])/anchor_boxes[:,2:]  # [N,2]
    encoded_wh=torch.log(real_boxes[:,2:]/anchor_boxes[:,2:])  # [N,2]

    return torch.cat([encoded_xy,encoded_wh],dim=1) # [N,4]

def decode_box(param_boxes,anchor_boxes):
    """Translate the parameterized box to the real boxes, real boxes
    are not the ground truths, just refer to boxes with format `xyxy` 
    Args:
        param_boxes (tensor) : [b,N,4], contain parameterized coordinates
        anchor_boxes (tensor) : [b,N,4], fmt is `xyxy`
    Return:
        boxes (tensor) : [b,N,4], whose format is `xyxy`
    """
    assert param_boxes.shape == anchor_boxes.shape 
    if param_boxes.is_cuda and not anchor_boxes.is_cuda:
        anchor_boxes=anchor_boxes.cuda(param_boxes.device.index)
    b,n,_=param_boxes.shape
    # change anchors to `ccwh`
    anchor_boxes=xyxy2ccwh(anchor_boxes.contiguous().view(-1,4),inplace=False).view(b,n,4)

    decoded_xy=param_boxes[:,:,:2]*anchor_boxes[:,:,2:]+anchor_boxes[:,:,:2] # [b,N,2]
    decoded_wh=torch.exp(param_boxes[:,:,2:])*anchor_boxes[:,:,2:] # [b,N,2]

    decode_box=torch.cat([decoded_xy,decoded_wh],dim=2)
    # change to `xyxy`
    decode_box=ccwh2xyxy(decode_box.view(-1,4),inplace=True).view(b,n,4)

    return  decode_box # [b,N,4]

def ccwh2xyxy(boxes,inplace=False):
    r"""Change the format of boxes from `ccwh` to `xyxy`
    Args:
        boxes (tensor): [n,4]
        inplace (bool): will return a new object if not enabled
    Return:
        after_boxes (tensor): [n,4], the transformed boxes
    """
    if inplace:
        after_boxes=boxes
    else:
        after_boxes=boxes.clone()
    
    if isinstance(after_boxes,Variable) and inplace:
        after_boxes[:,:2]=after_boxes[:,:2].clone()-after_boxes[:,2:]/2
        after_boxes[:,2:]=after_boxes[:,2:].clone()+after_boxes[:,:2]
    else:
        after_boxes[:,:2]-=after_boxes[:,2:]/2
        after_boxes[:,2:]+=after_boxes[:,:2]
    
    return after_boxes

def xyxy2ccwh(boxes,inplace=False):
    r"""Change the format of boxes from `xyxy` to `ccwh`
    Args:
        boxes (tensor): [n,4]
        inplace (bool): will return a new object if not enabled
    Return:
        after_boxes (tensor): [n,4], the transformed boxes
    """
    if inplace:
        after_boxes=boxes
    else:
        after_boxes=boxes.clone()
    if isinstance(after_boxes,Variable) and inplace:
        # use clone, or it will raise the inplace error
        after_boxes[:,2:]=after_boxes[:,2:].clone()-after_boxes[:,:2]
        after_boxes[:,:2]=after_boxes[:,:2].clone()+after_boxes[:,2:]/2
    else:
        after_boxes[:,2:]-=after_boxes[:,:2]
        after_boxes[:,:2]+=after_boxes[:,2:]/2
    return after_boxes

def t_meshgrid_2d(x_axis,y_axis):
    r"""Return 2d coordinates matrices of the two arrays
    Args:
        x_axis (tensor): [a]
        y_axis (tensor): [b]
    Return:
        x_axist (tensor): [b,a]
        y_axist (tensor): [b,a]
    """
    a,b=len(x_axis),len(y_axis)
    x_axis=x_axis[None].expand(b,a).clone()
    y_axis=y_axis[:,None].expand(b,a).clone()

    return x_axis,y_axis
    
def get_anchors(loc_anchors,h,w,stride=16,is_cuda=False):
    r"""Get the anchors with the size of the feature map
    Args:
        loc_anchors (tensor): [n,4]
        h (int): height
        w (int): width
    Return:
        anchors (tensor): [n*h*w,4]
    """
    n=len(loc_anchors)
    x_axis=torch.linspace(0,w-1,w)*stride
    y_axis=torch.linspace(0,h-1,h)*stride

    x_axis,y_axis=t_meshgrid_2d(x_axis,y_axis) # [h,w]

    x_axis=x_axis[None,None].expand(n,2,h,w).contiguous() # [n,2,h,w]
    y_axis=y_axis[None,None].expand(n,2,h,w).contiguous() # [n,2,h,w]

    # NOTE: contiguous is necessary since there are inplace operations below
    anchors=loc_anchors[:,:,None,None].expand(-1,-1,h,w).contiguous() # [n,4,h,w]
    
    # local coordinate to world coordinate
    # NOTE: inplace operations
    anchors[:,[0,2],:,:]+=x_axis
    anchors[:,[1,3],:,:]+=y_axis

    # transpose
    # NOTE: contiguous is necessary
    anchors=anchors.permute(0,2,3,1).contiguous() # [n,h,w,4]
    
    # reshape
    anchors=anchors.view(-1,4) # [n*h*w,4]
    if is_cuda:
        anchors=anchors.cuda(cfg.device_id)

    return anchors
    

def get_locally_anchors(stride=16,scales=[8,16,32],ars=[.5,1,2]):
    r"""Get the anchors in a locally window's coordinate
    Args:
        stride (int): 
        scales (list):[a] stores the anchor's scale relative to the feature map
        ars (list):[b] stores the aspect ratio of the anchor
    Return:
        locally_anchors (tensor):[a*b,4], coordinates obey the format `xyxy`
    """
    stride=torch.tensor(stride).float()
    scales=torch.tensor(scales).float()
    ars=torch.tensor(ars).float()

    n_scale,n_ar=len(scales),len(ars)
    ars=ars.sqrt()[:,None] # [n_ar, 1]

    base_anchors=scales[:,None,None].expand(-1,n_ar,2)/\
        torch.cat([ars,1/ars],dim=1) # [n_scale,n_ar,2]
    base_anchors*=stride

    stride/=2
    base_anchors=torch.cat(
        [stride.expand(n_scale,n_ar,2),
        base_anchors],
        dim=2
        ) # [n_scale,n_ar,4],fmt is `ccwh`
    base_anchors=base_anchors.view(-1,4) # [n_scale*n_ar,4]

    # change to `xyxy`
    base_anchors=ccwh2xyxy(base_anchors,inplace=True)

    return base_anchors


def t_box_iou(A,B):
    r"""Calculate iou between two boxes :attr:`A`
    and :attr:`B` obeys the format `xyxy`

    Args:
        A (tensor): [a,4]
        B (tensor): [b,4]
    Return:
        ious (tensor): [a,b] ::math:`ious_{ij}`
    denotes iou of `A_i` and `B_j`
    """
    a=A.shape[0]
    b=B.shape[0]
    AreaA=A[:,2:]-A[:,:2]
    AreaA=AreaA[:,0]*AreaA[:,1] # [a]
    AreaB=B[:,2:]-B[:,:2]
    AreaB=AreaB[:,0]*AreaB[:,1] # [b]
    
    AreaA=AreaA[:,None].expand(a,b) 
    AreaB=AreaB[None].expand(a,b)
    A=A[:,None].expand(a,b,4)
    B=B[None].expand(a,b,4)
    

    max_l=torch.where(A[:,:,0]>B[:,:,0],A[:,:,0],B[:,:,0])
    min_r=torch.where(A[:,:,2]<B[:,:,2],A[:,:,2],B[:,:,2])
    max_t=torch.where(A[:,:,1]>B[:,:,1],A[:,:,1],B[:,:,1])
    min_b=torch.where(A[:,:,3]<B[:,:,3],A[:,:,3],B[:,:,3])

    width=min_r-max_l
    height=min_b-max_t
    width=width.clamp(min=0)
    height=height.clamp(min=0)

    union=width*height # [a,b]
    ious=union/(AreaA+AreaB-union)

    return ious


def get_default_boxes(smin=cfg.smin,smax=cfg.smax,
                      ars=cfg.ar,im_size=cfg.intput_wh,
                      feat_map=cfg.feat_map,
                      steps=cfg.steps):
    r"""Get default boxes
    """
    # smax and smin is percent...
    smin*=im_size
    smax*=im_size
    step=(smax-smin)//(len(ars)-2)

    # scale k, a little different from the paper....
    sk=[.1*im_size]+[smin+_*step for _ in range(len(ars)) ] # [7], 0-6 for the detector layer 0-6
    
    default_boxes=[]
    for i,(f,ar,step) in enumerate(zip(feat_map,ars,steps)):
        s=sk[i]

        x_axis=torch.linspace(0,f-1,f)+.5
        x_axis=x_axis*step
        y_axis=x_axis.clone()
        x_axis,y_axis=t_meshgrid_2d(x_axis,y_axis) # [h,w]

        xy=torch.cat([x_axis[None],y_axis[None]],dim=0) # [2,h,w]

        # prepare ar
        ar=torch.tensor(ar).float().sqrt() # [3]
        ar=torch.cat([ar,1./ar]) # [6]
        ar=ar[:,None] # [6,1]
        ar=torch.cat([ar,ar],dim=1) # [6,2]
        ar[:,1]=1./ar[:,1]

        wh=s*ar # [6,2] 
        wh[0]=(torch.tensor(s)*torch.tensor(sk[i+1])).sqrt() # [bnum,2] 有两个aspect_ratio=1的，这里直接改第一个
        
        # for: [bnum,4,h,w] -> [bnum*4,h,w]=[c,h,w]
        wh=wh[...,None,None] # [bnum,2,1,1]        
        wh=wh.expand(-1,-1,f,f) # [bnum,2,h,w]

        xy=xy[None].expand_as(wh) # [bnum,2,h,w]
        
        xywh=torch.cat([xy,wh],dim=1) # [bnum,4,h,w]

        # transpose
        xywh=xywh.permute(0,2,3,1).contiguous() # [bnum,h,w,4]

        xywh=xywh.view(-1,4) # [bnum*h*w,4]

        # change to `xyxy`
        xywh=ccwh2xyxy(xywh)

        # clip?
        if cfg.clip:
            # tail '_' denotes the in-place operation...
            xywh.clamp_(min=0,max=cfg.intput_wh)
        default_boxes.append(xywh)
    
    default_boxes=torch.cat(default_boxes,dim=0)

    return default_boxes

default_boxes=get_default_boxes()
boxes_num=len(default_boxes)
ssd_loc_mean=torch.tensor(cfg.loc_mean)[None].expand(boxes_num,-1)
ssd_loc_std=torch.tensor(cfg.loc_std)[None].expand(boxes_num,-1)

def calc_target_(gt_boxes,gt_labels,pos_thresh=cfg.pos_thresh):
    r"""Calculate the net target for SSD data generator
    Args:
        gt_boxes (np.ndarray[int32]): [n,4]
        gt_labels (np.ndarray[int32]): [n]
    Return 
        target_ (torch.tensor[float32]): [a_num,4], a_num is the number of the default boxes
        labels_(torch.tensor[long]): [a_num], indicates the class of the positive anchors(default box) and negative samples 
    """
    # type check...
    if isinstance(gt_boxes,np.ndarray):
        gt_boxes=torch.tensor(gt_boxes).type_as(default_boxes)
    elif not isinstance(gt_boxes,torch.Tensor):
        raise ValueError()
    if isinstance(gt_labels,np.ndarray):
        gt_labels=torch.tensor(gt_labels).type_as(default_boxes).long()
    elif not isinstance(gt_labels,torch.Tensor):
        raise ValueError()

    ious=t_box_iou(default_boxes,gt_boxes) # [a_num,n]

    # first, find the highest iou between deafult boxes and ground truth...
    temp_ious=ious.clone()

    # -1 denotes no assignment, 0 is the neg, >1 means the labels of positive samples
    final_labels=torch.full([len(ious)],-1).type_as(gt_labels) # [a_num], filled with -1
    final_target=torch.full([len(ious),4],0).type_as(gt_boxes) # [a_num,4]

    while 1:
        idx= temp_ious.argmax() # 
        r_,c_=np.unravel_index(idx,temp_ious.shape)
        miou=temp_ious[r_,c_]
        if miou<1e-10:
           break       
        # NOTE: Attention, we have already plused one...
        final_labels[r_]=gt_labels[c_]+1 
        final_target[r_]=gt_boxes[c_]
        temp_ious[r_,:]=0
        temp_ious[:,c_]=0
    
    # assign for the iou > threshold
    # NOTE: a default box can only be matched to one gt box, but a gt box can be matched to more than one gt boxes   
    miou,idx=ious.max(dim=1) # [a_num]

    mask=(miou>pos_thresh)*(final_labels<0) # [a_num], (final_labels<0) can prevent to cover the assignment before.
    # NOTE: plus one
    final_labels[mask]=gt_labels[ idx[mask] ]+1
    final_target[mask]=gt_boxes[ idx[mask] ] 

    final_target=encode_box(final_target,default_boxes) # [a_num,4]

    final_target-=ssd_loc_mean
    final_target/=ssd_loc_std

    target_=final_target
    labels_=final_labels
    return target_,labels_

