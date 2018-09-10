# coding=UTF-8

import torch
from data import TestDataset
from torch.utils.data import DataLoader
from chainercv.evaluations import eval_detection_voc as voc_eval
from config import cfg
import os
import re
from net import SSD as MyNet
from tqdm import tqdm

def get_check_point():
    pat=re.compile("""weights_([\d]+)_([\d]+)""")
    base_dir=cfg.weights_dir
    w_files=os.listdir(base_dir)
    if len(w_files)==0:
        return 0,0,None
    w_files=sorted(w_files,key=lambda elm:int(pat.match(elm)[1]),reverse=True)

    w=w_files[0]
    res=pat.match(w)
    epoch=int(res[1])
    iteration=int(res[2])

    return epoch,iteration,base_dir+w


def eval_net(net=None,num=cfg.eval_number,shuffle=False):
    data_set=TestDataset()
    data_loader=DataLoader(data_set,batch_size=1,shuffle=shuffle,drop_last=False)
    
    is_cuda=cfg.use_cuda
    did=cfg.device_id

    if net is None:
        classes=data_set.classes
        net=MyNet(len(classes)+1)
        _,_,last_time_model=get_check_point()

        if os.path.exists(last_time_model):
            model=torch.load(last_time_model)
            net.load_state_dict(model)
            print("Using the model from the last check point:`%s`"%(last_time_model))
            
            if is_cuda:
                net.cuda(did)
        else:
            raise ValueError("no model existed...")

    net.eval()
   
    upper_bound=num

    gt_bboxes=[]
    gt_labels=[]
    gt_difficults=[]
    pred_bboxes=[]
    pred_classes=[]
    pred_scores=[]

    for i,(img,sr_im_size,gt_box,label,diff) in tqdm(enumerate(data_loader)):
        assert img.shape[0]==1
        if i> upper_bound:
            break

        sr_im_size=sr_im_size.float()
        if is_cuda:
            img=img.cuda(did)
            im_size=sr_im_size.cuda(did)
            
        pred_box,pred_class,pred_prob=net.predict(img,im_size)[0]
        prob_mask=pred_prob>cfg.out_thruth_thresh
        pbox=pred_box[prob_mask ] 
        plabel=pred_class[prob_mask ].long()
        pprob=pred_prob[prob_mask]

        gt_box=gt_box.numpy()

        if len(gt_box)!=0:
            # print(gt_box.shape)
            gt_box=gt_box[:,:,[1,0,3,2]] # change `xyxy` to `yxyx` 
        gt_bboxes += list(gt_box )
        gt_labels += list(label.numpy())
        gt_difficults += list(diff.numpy().astype('bool'))

        pbox=pbox.cpu().detach().numpy()
        if len(pbox)!=0:
            pbox=pbox[:,[1,0,3,2]] # change `xyxy` to `yxyx`
        pred_bboxes+=[pbox]
        pred_classes+=[plabel.cpu().numpy()]
        pred_scores+=[pprob.cpu().detach().numpy()]

        # pred_bboxes+=[np.empty(0) ]
        # pred_classes+=[np.empty(0) ]
        # pred_scores+=[np.empty(0) ]

    res=voc_eval(pred_bboxes,pred_classes,pred_scores,
        gt_bboxes,gt_labels,gt_difficults,use_07_metric=True)
    # print(res)

    # avoid potential error
    net.train()

    return res
