import numpy as np
import os
import warnings
import xml.etree.ElementTree as ET

from PIL import Image
from torch.utils.data import Dataset

def read_image(path):
     f = Image.open(path)
     img = f.convert('RGB')
     return img

class VOCDataset(Dataset):  
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]    

    def __init__(self,voc_root,list_name):
        r"""
        Args:
            voc_root (str): the path of my voc data 
            list_name (str): name of the list file, e.g. `train.txt`, 'test.txt'
        
        """
        super(VOCDataset,self).__init__()
        self._root=voc_root
        self._list_name=list_name
        self._img_list=[]
        self._gt=[]
        
        # load the list
        with open(self._root+'/'+self._list_name,encoding='utf-8') as list_file:
            buffer=list_file.read()
            buffer=buffer.split('\n')
            for i in buffer:
                temp=i.split(' ')
                assert (len(temp)-1)% 5 ==0
                if temp[0] == '':
                    continue
                self._img_list.append(temp[0])
                del temp[0]
                temp=np.array([int(_) if str.isdigit(_) else float(_)   for _ in temp],dtype='float32')
                self._gt.append(temp.reshape([-1,5]))

        assert len(self._gt) == len(self._img_list)            
        # print(buffer)

    def __getitem__(self,idx):
        r"""
        Args:
            idx(int): index of the sampled data
        Return:
            img (PIL.Image):  `RGB` format, pixel range[0,255]
            boxes (np.ndarray[float32]): `xyxy` format, abstract coordinate
            labels (np.ndarray[int]): from 0 to `cls_num`-1
        """
        img_path=self._root+'/'+ self._img_list[idx] # abosolute path
        boxes=self._gt[idx]  # [N,5].(cls_id,xmin,ymin,xmax,ymax), float type.
        boxes=boxes.copy() # copy to avoid the potential changes.. 
        img=read_image(img_path)
        # h,w,c=img.shape
        # boxes[:,1:]=boxes[:,1:]/np.array([w,h,w,h]) # normalization   

        # img=img[:,:,::-1] # convert bgr to rgb
        # img=img.astype('float32')

        return img,boxes[:,1:],(boxes[:,0]).astype('int')


    def __len__(self):
        return len(self._img_list)