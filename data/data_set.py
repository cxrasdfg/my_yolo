# coding=utf-8

import torch
import torchvision
from chainercv.datasets.voc import voc_utils
from torchvision import transforms
from torch.utils.data import Dataset  
import numpy as np

# from net.ssd.net_tool import calc_target_
from tqdm import tqdm

from .vocdataset import VOCDataset
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
        img (tensor[float32]): [c,h,w] in the range[0.0,1.0]
    """
    return transforms.Compose([
        transforms.ToTensor(),
        # caffe_normalize if cfg.use_caffe else torch_normalize
    ])(img)



def TrainTransform(img,boxes,labels,size,data_aug=False):
    r"""Data augmentation transform
    Args:
        img (PIL.Image) : [h,w,c] with `RGB`, and each piexl in [0,255]
        boxes (np.ndarray[int32]): [n,4] with `xyxy`
        labels (np.ndarray[int32]): [n], the labels of the box
        size (tupple): (width,height) of the image
        data_aug (boolean): will enable the data augmentation if True
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
    
    assert size[0]==size[1]
    input_wh=size[0]
    
    if data_aug:
        img = random_distort(img)
        if torch.rand(1) < 0.5:
            img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123, 116, 103))

        img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize_box(
        img, boxes, size=(input_wh, input_wh),
        random_interpolation=True
    )
    
    if data_aug:
        img, boxes = random_flip(img, boxes)
    img=img_normalize(img)

    return img, boxes, labels

def TestTransform(img,boxes,size):
    r"""For the preprocess of test data
    Args:
        img (PIL.Image): [h,w,c]
        boxes (tensor): no use
    Return:
        img (tensor[float32]): [c,h,w], pixel range:[0,1.] 
    """
    if isinstance(boxes,np.ndarray):
        boxes=torch.tensor(boxes).float()
    assert isinstance(boxes,torch.Tensor)
    
    assert size[0]==size[1]
    input_wh=size[0]
    
    img, _ = resize_box(
        img, boxes, size=(input_wh, input_wh),
        random_interpolation=False
    )

    img=img_normalize(img)

    return img

class TrainDataset(Dataset):
    classes=name_list
    def __init__(self,w,h,voc_root='/root/workspace/D/VOC2007_2012',MAX_BOX_NUM=100,data_aug=False):
        r"""
        Args:
            w (int): width of the image
            h (int): height of the image
            voc_root(str): root directory of the voc data(transformed data)
            MAX_BOX_NUM (int): indicates the maximum number of the boxes...
        """
        self.sdb=VOCDataset(voc_root,'train.txt')
        self.target_size=(w,h)
        self.MAX_BOX_NUM=MAX_BOX_NUM
        self.data_aug=data_aug

        if not self.data_aug:
            print('Warning: the data augumentation is not enabled...')

    def __getitem__(self,idx):
        r"""
        Args:
            idx (int): the idx of the sampled data
        Return:
            img (tensor[float32]): [c,h,w], pixel range (0,1)
            
        """
        ori_img,boxes,labels= self.sdb[idx] # [h,w,c] 
        
        boxes=boxes.copy()
       
        img,boxes,labels=TrainTransform(ori_img,boxes,labels,self.target_size,data_aug=self.data_aug)

        real_box_num=len(boxes)
        real_box_num=torch.tensor(real_box_num)

        # boxes with the fixed length...
        # it will cost extra memory but i have no way...
        assert real_box_num<=self.MAX_BOX_NUM
        fixed_boxes=torch.full([self.MAX_BOX_NUM,4],-1).float()
        fixed_boxes[:real_box_num]=boxes

        fixed_labels=torch.full([self.MAX_BOX_NUM],-999).long()
        fixed_labels[:real_box_num]=labels

        return img,fixed_boxes,fixed_labels,real_box_num


    def __len__(self):
        return len(self.sdb)

class TestDataset(Dataset):
    classes=name_list
    def __init__(self,w,h,voc_root='/root/workspace/D/VOC2007_2012',MAX_BOX_NUM=100):
        self.sdb = VOCDataset(voc_root,'test.txt')
        self.target_size=(w,h)
        self.MAX_BOX_NUM=MAX_BOX_NUM
        
    def __getitem__(self, idx):
        ori_img,boxes,labels= self.sdb[idx]
        # ori_img=ori_img.transpose(1,2,0) # [h,w,c]
        boxes=boxes.copy()

        img=TestTransform(ori_img,boxes.copy(),self.target_size)

        real_box_num=len(boxes)
        real_box_num=torch.tensor(real_box_num)

        # boxes with the fixed length...
        # it will cost extra memory but i have no way...
        assert real_box_num<=self.MAX_BOX_NUM
        fixed_boxes=torch.full([self.MAX_BOX_NUM,4],-1).float()
        fixed_boxes[:real_box_num]=boxes

        fixed_labels=torch.full([self.MAX_BOX_NUM],-999).long()
        fixed_labels[:real_box_num]=labels

        fixed_diffs=torch.full([self.MAX_BOX_NUM],0).long()        


        return img, np.array(ori_img.size),\
            fixed_boxes,fixed_labels, fixed_diffs,\
            real_box_num

    def __len__(self):
        return len(self.sdb)

