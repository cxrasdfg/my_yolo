# coding=utf-8

import numpy as np
import torch
from torch.autograd import  Variable
import codecs
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from skimage.transform import resize
import cv2 as cv
import time

class DetectionBox(object):
    def __init__(self,x,y,w,h,objectness,cls_prob):
        self.x=x
        self.y=y
        self.w=w
        self.h=h

        self.objectness=objectness
        self.cls_prob=cls_prob
        self.sort_id=None  # just sort it

def IouRate(bb,gt):
    """
    :param bb: N x 4 estimated data, format of coord is [left_x,top_y,w,h]
    :param gt: N x 4 ground truth data
    :return:IOU for each pair samples
    """

    Left=np.maximum(bb[:,0],gt[:,0])

    Right=np.minimum(bb[:,0]+bb[:,2],gt[:,0]+gt[:,2])

    Top=np.maximum(bb[:,1],gt[:,1])

    Bottom=np.minimum(bb[:,1]+bb[:,3],gt[:,1]+gt[:,3])

    Inter=np.maximum(0,Right-Left)*np.maximum(0,Bottom-Top)
    Union=bb[:,2]*bb[:,3]+gt[:,2]*gt[:,3]-Inter

    return Inter/Union



def NP2Variable(v,is_cuda=True):
    if is_cuda:
        return Variable(torch.from_numpy(v).float()).cuda()
    else:
        return Variable(torch.from_numpy(v).float())

def nms_sort(boxes):
    """
    :param boxes: num x (4 + 1 + classes)
    :return:
    """
    assert len(boxes.shape) == 2
    classes= boxes.shape[1] - 5  # subtract (4 + 1)

    # set cls_id for sorting
    for i in range(classes):
        boxes=np.array(sorted(boxes, key=lambda x:x[5+i],reverse=True))
        for j in range(len(boxes)):
            if boxes[j,5+i] ==0:
                continue

            for k in range(j+1,len(boxes)):
                box_j=np.array(boxes[j:j+1,:4])
                box_k=np.array(boxes[k:k+1,:4])
                # change [center_x,center_y,w,h] to [left_x,top_y,w,h]
                box_j[:,:2]-=box_j[:,2:]/2
                box_k[:,:2]-=box_k[:,2:]/2

                iou=IouRate(box_j,box_k)
                if iou[0]>=.45:
                    boxes[k,5+i]=0

    return boxes

def GetNames(name_file):
    file = codecs.open(name_file, 'rb+', encoding='utf-8')
    data = file.readlines()
    data=list(data)
    for i in range(len(data)):
        data[i]=data[i].strip('\n')
    return data

def GetBoxesForShow(boxes,thresh=.5):
    """
    :param boxes: num1 x (4 + 1 + classes)
    :param thresh: num2 x (4 + class_id)
    :return:
    """
    cls_prob=boxes[:,5:]
    idx=cls_prob.argmax(axis=1)

    max_prob=cls_prob[[i for i in range(len(cls_prob))],idx]

    boxes=boxes[:,:6]
    boxes[:,4]=idx  # idx equals to the class
    boxes[:,5]=max_prob
    res=boxes[max_prob>=thresh]

    return res

def GetBoxFromNetOutput(dark_out, thresh=.5):
    """
    :param dark_out: batch x m_anchor_num x (width * height) x (4 + 1 + classes)
    :param thresh:
    :return: batch x (m_anchor_num * width * height) x (4 + 1 + classes)
    """
    batch,m_anchor_num,w_h,coor_prob=dark_out.shape

    dark_out=dark_out.view([batch, m_anchor_num * w_h, coor_prob])

    # change to numpy array
    dark_out=dark_out.cpu().data.numpy()

    res=[]
    for i in range(batch):
        truth= dark_out[i, :, 4]
        boxes=dark_out[i][np.where(truth >= thresh)]

        # change condition probability to joint probability
        boxes[:,5:]*=boxes[:,4:5]

        res.append(boxes)

    return res

def LoadImgForward(img_path,shape,is_cuda=True):
    # img=Image.open(img_path).convert('RGB')
    # img=img.resize(shape)

    # width = img.width
    # height = img.height
    # img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    # img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    # img = img.view(1, 3, height, width)
    # img = img.float().div(255.0)
    #
    # if is_cuda:
    #     img=Variable(img).cuda()
    # else:
    #     img=Variable(img)
    img=plt.imread(img_path)
    img=resize(img,shape)
    img=np.array([img],dtype=np.float32)
    img=np.transpose(img,[0,3,1,2])
    # img/=255
    img=NP2Variable(img)
    return img

def DrawBoxOnImg(img_path,boxes,name_file_path=None):
    im=cv.imread(img_path)
    h=im.shape[0]
    w=im.shape[1]
    class_name=None
    if name_file_path is not None:
        class_name=GetNames(name_file_path)

    for box in boxes:
        bb_for_show = np.array(box[:4])
        # change (center_x,center_y,w,h) to (left_x,top_y,w,h)
        bb_for_show[:2]-=bb_for_show[2:]/2
        bb_for_show*=[w,h,w,h]
        p1 = (int(bb_for_show[0]), int(bb_for_show[1]))
        p2 = (int(bb_for_show[0] + bb_for_show[2]), int(bb_for_show[1] + bb_for_show[3]))
        cv.rectangle(im, p1, p2, (255, 0, 0), 2, 1)
        if class_name is not None:
            # cv.putText(im, "%s" % class_name[int(box[4])], (bb_for_show[0], bb_for_show[1]), cv.FONT_HERSHEY_SIMPLEX, 0.75,
            #        (0, 0, 255), 2)
            im=DrawText(im,"%s" % class_name[int(box[4])],
                        (bb_for_show[0], bb_for_show[1]),
                        (255,0,0))

        print("%s, prob:%.4f" % (class_name[int(box[4])],box[5]))
    cv.imshow('Detection', im)

    if cv.waitKey(0) & 0xff == 27:
        exit()

def DrawText(img_op,text_str,pos,color):
    img_PIL = Image.fromarray(cv.cvtColor(img_op, cv.COLOR_BGR2RGB))

    # ??  ??*.ttc????????? /usr/share/fonts/opentype/noto/ ????locate *.ttc
    font = ImageFont.truetype('NotoSansCJK-Black.ttc', 18)

    draw = ImageDraw.Draw(img_PIL)
    draw.text(pos, text_str, font=font, fill=color)

    img_op= cv.cvtColor(np.asarray(img_PIL), cv.COLOR_RGB2BGR)

    return img_op