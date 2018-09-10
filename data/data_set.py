# coding=UTF-8

import torch
import torchvision
from chainercv.datasets.voc import voc_utils
from torchvision import transforms
from config import cfg
from torch.utils.data import Dataset  
import numpy as np

from net.ssd.net_tool import calc_target_
from tqdm import tqdm

from .vocdataset import VOCBboxDataset
from .transforms import random_crop,random_distort,\
    random_flip,random_paste,random_paste,scale_jitter,\
    resize as resize_box

name_list=voc_utils.voc_bbox_label_names

def caffe_normalize(img):
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    mean = torch.tensor([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1).type_as(img)
    # NOTE: check the
    assert img.max()<=1.0
    img=img*255.0  
    img = (img - mean).float()
    return img

def torch_normalize(img):
    assert img.max()<=1.0
    # img=img/255.0
    img=torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.224])(img)

    return img


def img_normalize(img):
    r"""Image normalize...
    Args:
        img: [PIL.Image]: [h,w,c], `RGB` format in [0,255]
    Return:
        img (tensor[float32])
    """
    return transforms.Compose([
        transforms.ToTensor(),
        caffe_normalize if cfg.use_caffe else torch_normalize
    ])(img)



def TrainTransform(img,boxes,labels):
    r"""Data augmentation transform
    Args:
        img (PIL.Image) : [h,w,c] with `RGB`, and each piexl in [0,255]
        boxes (np.ndarray[int32]): [n,4] with `xyxy`
        labels (np.ndarray[int32]): [n], the labels of the box
    Return:
        img (tensor[float32]): [c,s',s'] with `BGR`, and each pixel in [-128,127]
        boxes (tensor[float32] ): [n',4] with `xyxy`
        labels (tensor[float32] ): [n'], the labels of box after transforms
    """
    if isinstance(labels,np.ndarray):
        labels=torch.tensor(labels).long()
    assert isinstance(labels,torch.Tensor)
    
    if isinstance(boxes,np.ndarray):
        boxes=torch.tensor(boxes).float()
    assert isinstance(boxes,torch.Tensor)

    if cfg.data_aug:
        img = random_distort(img)
        if torch.rand(1) < 0.5:
            img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123, 116, 103))

        img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize_box(
        img, boxes, size=(cfg.intput_wh, cfg.intput_wh),
        random_interpolation=True
    )
    
    if cfg.data_aug:
        img, boxes = random_flip(img, boxes)
    img=img_normalize(img)

    return img, boxes, labels

def TestTransform(img,boxes):
    r"""For the preprocess of test data
    Args:
        img (PIL.Image): 
        boxes (tensor): no use
    Return:
        img (tensor[float32]): [c,h,w], pixel range:[0,1.] 
    """
    if isinstance(boxes,np.ndarray):
        boxes=torch.tensor(boxes).float()
    assert isinstance(boxes,torch.Tensor)
    img, _ = resize_box(
        img, boxes, size=(cfg.intput_wh, cfg.intput_wh),
        random_interpolation=False
    )

    img=img_normalize(img)

    return img


class TrainDataset(Dataset):
    classes=name_list
    def __init__(self):
        #self.cfg=cfg
        self.sdb=VOCBboxDataset(cfg.voc_dir,'trainval')
    
    def __getitem__(self,idx):
        # NOTE: sdb returns the `yxyx`...
        ori_img= self.sdb._get_image(idx) # [h,w,c] 
        
        # change dim order
        # ori_img=ori_img.transpose(1,2,0) # [h,w,c]

        boxes,labels,diffs=self.sdb._get_annotations(idx)
        boxes=boxes.copy()
        boxes=boxes[:,[1,0,3,2]] # change `yxyx` to `xyxy`

        # boxes=torch.tensor(boxes)
        img,boxes,labels=TrainTransform(ori_img,boxes,labels)

        target_,labels_=calc_target_(boxes,labels)
        num_pos=(labels_>0).sum()
        # tqdm.write("Image Id:%d, number of pos:%d"%(idx,num_pos) ,end=',\t ' )

        return img,target_,labels_


    def __len__(self):
        return len(self.sdb)

class TestDataset(Dataset):
    classes=name_list
    def __init__(self, voc_data_dir=cfg.voc_dir, split='test', use_difficult=True):
        self.sdb = VOCBboxDataset(voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img= self.sdb._get_image(idx)
        # ori_img=ori_img.transpose(1,2,0) # [h,w,c]
        boxes,labels,diffs=self.sdb._get_annotations(idx)
        boxes=boxes.copy()
        boxes=boxes[:,[1,0,3,2]] # change `yxyx` to `xyxy`

        img=TestTransform(ori_img,boxes.copy() )
        return img, np.array(ori_img.size),\
            boxes,labels.astype('long'), diffs.astype('int')

    def __len__(self):
        return len(self.sdb)

