# coding=utf-8
import torch
import numpy as np
from config import cfg
from tqdm import tqdm

from libs import pth_nms as ext_nms
from .vgg16_caffe import caffe_vgg16 as vgg16,L2Norm
from .net_tool import default_boxes,decode_box,ssd_loc_mean,ssd_loc_std
from .loss import SSDLoss


class ConvBN2d(torch.nn.Module):
    def __init__(self,in_,out_,size_,pad_,stride_,bn_,relu_=True):
        super(ConvBN2d,self).__init__()

        self.conv=torch.nn.Conv2d(in_,out_,size_,stride_,padding=pad_)
        self.bn=torch.nn.BatchNorm2d(out_) if bn_ else None
        self.act=torch.nn.ReLU(inplace=True) if relu_ else None
    
    def forward(self,x):
        x=self.conv(x)
        if self.bn:
            x=self.bn(x)
        if self.act:
            x=self.act(x)
        return x

class SSD(torch.nn.Module):
    def __init__(self,class_num):
        super(SSD,self).__init__()
        
        assert class_num==21,"Only support VOC dataset currently..."
        
        # this num should contain the background...
        self.class_num=class_num

        self.conv4,self.conv5,self.conv6,self.conv7=vgg16()
        self.l2_norm=L2Norm(self.conv4[-2].out_channels,cfg.l2norm_scale)

        bn_=cfg.use_batchnorm        
        
        self.conv8=torch.nn.Sequential(*[
            ConvBN2d(1024,256,1,0,1,bn_),
            ConvBN2d(256,512,3,1,2,bn_)
        ])
        
        self.conv9=torch.nn.Sequential(*[
            ConvBN2d(512,128,1,0,1,bn_),
            ConvBN2d(128,256,3,1,2,bn_)
        ])

        self.conv10=torch.nn.Sequential(*[
            ConvBN2d(256,128,1,0,1,bn_),
            ConvBN2d(128,256,3,0,1,bn_)
        ])

        self.conv11=torch.nn.Sequential(*[
            ConvBN2d(256,128,1,0,1,bn_),
            ConvBN2d(128,256,3,0,1,bn_)
        ] )

        ar=cfg.ar
        feat_map=cfg.feat_map
        det_in=cfg.det_in_channels

        assert len(ar) == len(feat_map)
        assert len(ar) == len(det_in)

        num_loc_channels=[len(_)*2*4 for _ in ar]
        num_cls_channels=[len(_)*2*self.class_num for _ in ar]

        conv_loc_layers=[]
        conv_cls_layers=[]

        for input_ch,num_loc_ch,num_cls_ch in \
            zip(det_in,num_loc_channels,num_cls_channels):
            conv_loc_layers+=[ConvBN2d(input_ch,num_loc_ch,3,1,1,bn_,False)]
            conv_cls_layers+=[torch.nn.Sequential(*[
                ConvBN2d(input_ch,num_cls_ch,3,1,1,bn_,False),
                # torch.nn.Softmax(dim=1) #  sily b...
            ])]

        self.conv_loc_layers=torch.nn.Sequential(*conv_loc_layers)
        self.conv_cls_layers=torch.nn.Sequential(*conv_cls_layers)
        
        # use xavier to initialize the newly added layer...
        for k,v in torch.nn.Sequential(*[
            # self.conv6,self.conv7,
            self.conv8,self.conv9,self.conv10,self.conv11,
            self.conv_loc_layers,self.conv_cls_layers
        ]).named_parameters():
            if 'bias' in k:
                # torch.nn.init.normal_(v.data)
                v.data.zero_()
            else:
                # torch.nn.init.xavier_normal_(v.data)
                torch.nn.init.xavier_uniform_(v.data)

            # torch.nn.init.normal_(v.data,0,0.1)
        
        # get default boxes
        self.default_boxes=default_boxes
        
        self.loss_func=SSDLoss()

        self.get_optimizer(lr=cfg.lr,use_adam=cfg.use_adam,weight_decay=cfg.weight_decay)
    
    def _convert_other(self):
        other_w=torch.load('./models/ssd300_mAP_77.43_v2.pth')

        net_state=self.state_dict()
        keys=list(net_state.keys())
        other_keys=list(other_w.keys())

        #  totally one to one
        for _k,_ko in zip(keys,other_keys ):
            net_state[_k]=other_w[_ko]
        

        torch.save(net_state,'%sweights_%d_%d'%(cfg.weights_dir,10000,1000) )

    def get_optimizer(self,lr=1e-3,use_adam=False,weight_decay=0.0005):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        params=[]
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                # if 'bias' in key:
                #     params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                # else:
                #     params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
        if use_adam:
            print("Using Adam optimizer")
            self.optimizer = torch.optim.Adam(params)
        else:
            print("Using SGD optimizer")
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def forward(self,x):
        x_4=self.conv4(x)

        # x_5 and x_6 is not for prediction
        x_5=self.conv5(x_4)

        x_4=self.l2_norm(x_4)

        x_6=self.conv6(x_5)

        x_7=self.conv7(x_6)
        x_8=self.conv8(x_7)
        x_9=self.conv9(x_8)
        x_10=self.conv10(x_9)
        x_11=self.conv11(x_10)

        res=[]
        for x_,loc_l,cls_l in \
            zip(
                (x_4,x_7,x_8,x_9,x_10,x_11),
                self.conv_loc_layers,
                self.conv_cls_layers
            ):
            loc_out= loc_l(x_)
            cls_out=cls_l(x_)

            res.append((loc_out,cls_out))
        
        return res
    
    def convert_features(self,x):
        r"""Convert the feature after forward...
        Args:
            x (list[(tensor,tensor),...]):
        Return:
            locs (tensor[float32]): [b,N',4]
            clss (tensor[float32]): [b,N',cls_num]
        """
        locs=torch.empty(0).type_as(x[0][0])
        clses=locs.clone()
        
        for loc,cls in x:
            # loc[b,c1,h,w], cls[b,c2,h,w]
            b,_,h,w=loc.shape
            loc=loc.view(b,-1,4,h,w) # [b,bnum,4,h,w]
            loc=loc.permute(0,1,3,4,2).contiguous() # [b,bnum,h,w,4]
            loc=loc.view(b,-1,4) # [b,bnum*h*w,4]
            locs=torch.cat([locs,loc],dim=1) # [b,n'+bnum*h*w ,4]

            cls=cls.view(b,-1,self.class_num,h,w) # [b,bnum,cls_num,h,w]
            cls=cls.permute(0,1,3,4,2).contiguous() # [b,bnum,h,w,cls_num]
            cls=cls.view(b,-1,self.class_num) # [b,bnum*h*w,cls_num]
            cls=cls.softmax(dim=2)
            clses=torch.cat([clses,cls],dim=1) # [b,n'+bnum*h*w,cls_num]
            # tqdm.write('h:%d,w:%d' %(h,w) ,end=',\t ')

        return locs,clses
    
    def train_once(self,imgs,target_,labels_):
        r"""Train once
        Args:
            imgs (tensor[float32]): [b,c,h,w]
            target_ (tensor[float32]): [b,tbnum,4]
            labels (tensor[long]): [b,tbnum], idx 0 is the negative sample
        Return:
            loss (float)
        """
        # forward
        x=self(imgs) # list: shape of [6]
        out_locs,out_clses=self.convert_features(x) # [b,tbnum,4] [b,tbum,cls_num]
        
        pos_mask=(labels_>0) # [b,tbnum]
        pos_out_locs=out_locs[pos_mask] #  [n',4]
        pos_out_clses=out_clses[pos_mask] # [n',cls_num]

        # NOTE: label has already plused by one
        pos_gt_labels=labels_[pos_mask] # [n']
        pos_gt_locs=target_[pos_mask] # [n',4]
        

        # HNM(Hard Negative Mining)...
        # by convention, use 0 represents negative scores...
        neg_out_clses=out_clses[:,:,0].clone() # [b,tbnum]
        neg_out_clses[pos_mask]=1e10 
        _,idx=neg_out_clses.sort(dim=1) # [b,tbnum]
        _,idx=idx.sort(dim=1) # [b,tbnum], get the rank...
        # NOTE: ratio of pos and neg is 1:3
        num_neg=cfg.neg_ratio*pos_mask.long().sum(dim=1) # [b]
        rank_mask=idx<num_neg[:,None].expand_as(idx) # [b,tbnum] just select top num_neg negative sample...
        neg_out_clses=out_clses[rank_mask] # [n'',cls_num]

        loss=self.loss_func(
            pos_out_clses,
            pos_gt_labels,
            neg_out_clses,
            pos_out_locs,
            pos_gt_locs,
            cfg.alpha
        )

        # gradient descent
        if isinstance(loss,int) and loss==0:
            return 0
        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.parameters(),10)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def predict(self,imgs,src_shape):
        r"""Predict an image
        Args:
            img (tensor[float32]): [b,3,h,w]
            src_shape (tensor[float32]): [b,2] the width and height of the origin image
        Return:
            res (list[(boxes,classes,confs),...]) 
        """
        ratios= src_shape/\
            torch.tensor(
                imgs.shape[2:][::-1]
            )[None].expand_as(src_shape).type_as(imgs) # [b,2]

        ratios=torch.cat([ratios,ratios],dim=1) # [b,4]

        x=self(imgs)
        locs,clses=self.convert_features(x) 

        res=[]

        mean=ssd_loc_mean[None].expand_as(locs).type_as(locs) # [b,tbum,4]
        std=ssd_loc_std[None].expand_as(locs).type_as(locs) # [b,tbnum,4]
        locs=locs*std+mean

        locs=decode_box(locs,self.default_boxes[None].expand_as(locs))
        for loc,cls,ratio in zip(locs,clses,ratios):
            # loc[tbnum,4], cls [tbnum,cls_num]
            
            # remove neg
            cls=cls[:,1:]
            temp=self.nms(torch.cat([loc,cls],dim=1),cfg.out_nms,cfg.out_nms_filter) # [n',4+cls_num-1]
            temp_scores=temp[:,4:] # scores
            scores,idx=temp_scores.max(dim=1)
            
            # print("shape en?",temp.shape)
            pred_box=temp[:,:4]
            pred_label=idx
            pred_conf=scores

            # NOTE: as the paper says, keep top-N score's boxes per image
            _max,_idx=pred_conf.sort(descending=True)
            _idx=_idx[:cfg.out_box_num_per_im]
            pred_box=pred_box[_idx]
            pred_label=pred_label[_idx]
            pred_conf=pred_conf[_idx]

            # print("shape en?",pred_box.shape)
            pred_box*=ratio[None].expand_as(pred_box)

            res.append((pred_box,pred_label,pred_conf))

        return res


    def nms(self,rois,thresh=.7,filter_=1e-10):
        """
        nms to remove the duplication
        input:
            rois (tensor): [N,4+cls_num], attention that the location format is `xyxy`
            thresh (float): the threshold for nms
        return:
            rois (tensor): [M,4+cls_num], regions after nms 
        """
        cls_num=rois.shape[1]-4
        
        for i in range(cls_num):
            dets=rois[:,[0,1,2,3,i+4]]
            order=ext_nms(dets,thresh=thresh)
            mask=torch.full([len(dets)],1,dtype=torch.uint8)
            mask[order]=0
            del order
            rois[:,i+4][mask]=0
        
        sorted_rois,_=rois[:,4:].max(dim=1)
        rois=rois[sorted_rois>filter_] # [M,4+cls_num] 

        return rois 

    def cuda(self,did=0):
        torch.nn.Module.cuda(self,did)
        if not self.default_boxes.is_cuda:
            self.default_boxes=self.default_boxes.cuda(did)
        return self

    def cpu(self):
        torch.nn.Module.cpu(self)
        if self.default_boxes.is_cuda:
            self.default_boxes=self.default_boxes.cpu()
        return self

    def _print(self):
        print('********\t NET STRUCTURE \t********')
        print(torch.nn.Sequential(*[
            self.conv4,self.conv5,self.conv6,self.conv7,
            self.conv8,self.conv9,self.conv10,self.conv11,
            self.conv_loc_layers,self.conv_cls_layers
        ]))
        print('********\t NET END \t********')
