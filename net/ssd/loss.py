# coding=utf-8

import torch
from config import cfg

from tqdm import tqdm

def _smooth_l1_loss(x,gt,sigma):
    sigma2 = sigma ** 2
    diff = (x - gt)
    abs_diff = diff.abs()
    flag = (abs_diff < (1. / sigma2))
    y=torch.where(flag, (sigma2 / 2.) * (diff ** 2), (abs_diff-.5/sigma2))
    return y.sum()

class SSDLoss(torch.nn.Module):
    def __init__(self):
        super(SSDLoss,self).__init__()

    def forward(self,pos_cls,pos_label,neg_cls,out_box,gt_box,_alpha=cfg.alpha):
        # pos_label is already plus by one 
        assert pos_label.min() >0
        num_pos=len(pos_cls)
        num_neg=len(neg_cls)
        n_cls=num_pos+num_neg
        
        if num_pos == 0:
            return 0

        cls_loss=-pos_cls[torch.arange(num_pos).long(),(pos_label).long()].log().sum()\
            -(neg_cls[:,0].log().sum() if len(neg_cls)!=0 else 0)

        tqdm.write('cls_loss: %.5f'%(cls_loss.item()),end=' ||')

        ttt=pos_cls[torch.arange(num_pos ).long(),(pos_label).long()].max()
        _,acc=pos_cls[:,:].max(dim=1)
        acc=acc
        acc=acc.long()
        acc=(acc==pos_label).sum().float()/num_pos
        tqdm.write("class branch: max prob=%.5f, acc=%.5f" \
            %(ttt,acc),end=" || ")

        # smooth l1...
        reg_loss=_smooth_l1_loss(out_box,gt_box,cfg.sigma)
        tqdm.write('reg_loss: %.5f'%(reg_loss.item()) ,end=' || ')
        # loss=cls_loss/n_cls+reg_loss/num_pos*_lambda
        loss=cls_loss+reg_loss*_alpha
        loss/=num_pos

        return loss
