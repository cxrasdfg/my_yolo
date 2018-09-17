# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from utility import t_meshgrid_2d,t_box_iou,ccwh2xyxy,xyxy2ccwh

class DetectionLayer(torch.nn.Module):
    def __init__(self,
        classes,
        coords,
        rescore,
        side,
        num,
        softmax,
        sqrt,
        jitter,
        object_scale,
        noobject_scale,
        class_scale,
        coord_scale,):
        
        super(DetectionLayer,self).__init__()

        # use list wrap to prevent the recursive calling in module...
        # use this parameter in forward...
        # self.ptr_darknet=[ptr_darknet]

        self.classes,\
        self.coords,\
        self.rescore,\
        self.side,\
        self.num,\
        self.softmax,\
        self.sqrt,\
        self.jitter,\
        self.object_scale,\
        self.noobject_scale,\
        self.class_scale,\
        self.coord_scale=\
        classes,\
        coords,\
        rescore,\
        side,\
        num,\
        softmax,\
        sqrt,\
        jitter,\
        object_scale,\
        noobject_scale,\
        class_scale,\
        coord_scale

        x=torch.linspace(0,self.side-1,self.side) 
        self.x_grid,\
        self.y_grid=t_meshgrid_2d(x,x) # range [0,self.side-1]

    def forward(self,*args):
        r"""
        Args:
            if self.training:
                b_x (tensor[float32]): [b,f], a connected layer is before this layer...
                b_fixed_boxes (tensor[float32]): [b,MAX_BOX_NUM,4]
                b_fixed_labels (tensor[long]): [b,MAX_BOX_NUM]
                b_real_box_num (tensor[long]): [b], number of real boxes in element of batch
                ref_darknet: reference to the darknet, then we can get the gloabl hyperparams...
            else:
                b_x (tensor[float32]): [b,f]
                b_src_img_size (tensor[float32]): [b,2], stores the original image width and height
        Return:
            if self.training:
                loss (tensor[float32]): the loss of it...
            else:
                res (list[(pred_boxes,pred_labels,pred_confs),...])
        """
        if self.training:
            b_x,b_fixed_boxes,\
            b_fixed_labels,\
            b_real_box_num,\
            ref_darknet=args
            # continue?
            # self.side*self.side*(self.num*(self.coords+self.rescore)+self.classes)
            
            # reshape
            b,_=b_x.shape

            # here shape to be [b,side,side,(num*(coords+rescore)+classes)]
            b_x=b_x.view(b,self.side,self.side,(self.num*(self.coords+self.rescore)+self.classes))
            
            img_h=ref_darknet.net_height
            img_w=ref_darknet.net_width
            
            x_grid=self.x_grid.type_as(b_x) # [side,side]=[7,7]
            y_grid=self.y_grid.type_as(b_x)# [side,side]=[7,7]

            # prepare `default box`
            x_grid=x_grid[...,None] # [side,side,1]
            y_grid=y_grid[...,None] # [side,side,1]

            x_grid=x_grid.expand(-1,-1,self.num) # [side,side,num]
            y_grid=y_grid.expand(-1,-1,self.num) # [side,side,num]
            
            x_grid=x_grid[...,None] # [side,side,num,1]
            y_grid=y_grid[...,None] # [side,side,num,1]
            
            default_xy=torch.cat([x_grid,y_grid],dim=3) # [side,side,num,2]

            # NOTE: no normalization, since offset laies on [0,1]
            # normalize to [0.,1.]
            # default_xy[...,0]/=1.*self.side
            # default_xy[...,1]/=1.*self.side    
            
            loss=[]
            for idx_batch,(x, # [7,7,35]
                    fixed_boxes, # [box_num,4]
                    fiexd_labels, # [box_num]
                    real_box_num,
                )\
                in enumerate(zip(b_x,b_fixed_boxes,b_fixed_labels,b_real_box_num)):
                # NOTE: get the real boxes...
                boxes=fixed_boxes[:real_box_num]
                labels=fiexd_labels[:real_box_num]

                # 1. gt_boxes...
                # change to `ccwh`
                ccwh_boxes=xyxy2ccwh(boxes)
                # normalize the box to [0.,1.*side)
                ccwh_boxes[:,[0,2]]=ccwh_boxes[:,[0,2]].clone()/(1.*img_w/self.side)
                ccwh_boxes[:,[1,3]]=ccwh_boxes[:,[1,3]].clone()/(1.*img_h/self.side)
                
                # yolo selects the boxes which gt's center is located in...
                # NOTE: for indexing, axis_y is the first dim...
                grid_idx= ccwh_boxes[:,:2].long()
                x_idx=grid_idx[:,0] # [real_box_num]
                y_idx=grid_idx[:,1] # [real_box_num]

                # normalize width and height to [0.0,1.0]
                ccwh_boxes[:,[0,2]]= ccwh_boxes[:,[0,2]].clone()/(1.0*self.side)
                ccwh_boxes[:,[1,3]]= ccwh_boxes[:,[1,3]].clone()/(1.0*self.side)

                # 2. some transformations on net output
                out_locoff_conf=x[...,:self.num*(self.coords+self.rescore)]
                
                # [side,side,num,(coords+rescore)]
                out_locoff_conf=out_locoff_conf.view(self.side,self.side,self.num,-1) 
                
                out_locoff=out_locoff_conf[...,:self.coords] # [side,side,num,coords]
                out_conf=out_locoff_conf[...,self.coords] # [side,side,num]

                out_cls=x[...,self.num*(self.coords+self.rescore):] # [side,side,classes]
                
                # 3. from output to predict box...
                # out_loc,out_conf,out_cls
                # I think should add sigmoid on location.
                out_loc=out_locoff.sigmoid() # [side,side,num,4]
                out_conf=out_conf.sigmoid() # [side,side,num]

                if self.softmax:
                    out_cls=out_cls.softmax(dim=2) # [side,side,20]
                else:
                    out_cls=out_cls.sigmoid() # [side,side,20]
                
                pred_loc=out_loc.clone() # [side,side,num,4]
                # NOTE: add default_xy, from the offset to the relative coords...
                pred_loc[...,:2]=pred_loc[...,:2].clone()+default_xy
                
                # normalize the x,y in pred_loc to [0,1]
                pred_loc[...,:2]=pred_loc[...,:2].clone()/self.side

                # 4. match the box centered in
                # centered_loc=pred_loc[y_idx,x_idx] # [len(y_idx),num,4]=[real_box_num,num,4]
                # change to `xyxy`
                xyxy_boxes=ccwh2xyxy(ccwh_boxes)

                tt=pred_loc.shape
                pred_loc=pred_loc.view(-1,4)
                
                pred_loc=ccwh2xyxy(pred_loc)

                pred_loc=pred_loc.view(tt)
                # ious=t_box_iou(centered_loc.view(-1,4),xyxy_boxes) # [real_box_num*num,real_box_num]
                # ious=ious.view(real_box_num,self.num,self.coords) # [real_box_num,num,real_box_num]
                
                gt_loc_target=torch.full(out_loc.shape,0).type_as(out_loc) # [side,side,num,4]
                gt_iou_target=torch.full(out_loc.shape[:3],0).type_as(out_loc) # [side,side,num]
                gt_cell_box_assign=torch.full(out_loc.shape[:3],0).long() # [side,side,num]
                gt_cell_asiign=torch.full(out_loc.shape[:2],0).long() # [side,side]    
                gt_cls_assign=torch.full(out_cls.shape,0).type_as(out_cls) # [side,side,classes]

                for gt_loc_xyxy,gt_loc_ccwh,gt_label,cell_i,cell_j in\
                    zip(xyxy_boxes,ccwh_boxes,labels,y_idx,x_idx):
                    cell_boxes=pred_loc[cell_i,cell_j] # [num,4]
                    ious=t_box_iou(cell_boxes,gt_loc_xyxy[None]).view(-1) # [num]
                    max_box_idx=ious.argmax()
                    
                    # NOTE: it may cause one cell box is assigned to multi gt boxes, use flag array to prevent.
                    gt_cell_asiign[cell_i,cell_j]=1
                    gt_cell_box_assign[cell_i,cell_j,max_box_idx]=1
                    # NOTE: No need for plusing by one
                    
                    # tt=torch.full([self.classes],0).type_as(gt_cls_assign)
                    # tt[gt_label]=1.
                    gt_cls_assign[cell_i,cell_j,gt_label]=1.
                    
                    # NOTE: get offset...
                    tt=gt_loc_ccwh.clone()
                    tt[:2]=tt[:2].clone()*self.side-default_xy[cell_i,cell_j,max_box_idx]
                    gt_loc_target[cell_i,cell_j,max_box_idx]=tt
                    # assign iou... i.e. the confidence target...
                    gt_iou_target[cell_i,cell_j,max_box_idx]=ious[max_box_idx] 
                
                # 5.prepare the loss function....
                if self.sqrt:
                    gt_loc_target[...,2:]=gt_loc_target[...,2:].clone().sqrt()
                    # NOTE: using this exp will raise the inplace error, but I have used `clone` to prevent, confused...
                    # out_loc[...,2:]=out_loc[...,2:].clone().sqrt()+0                    
                    sqrt_wh=out_loc[...,2:].clone().sqrt()
                    out_loc=out_loc.clone()
                    out_loc[...,2:]=sqrt_wh
                
                one_obj_i_j_mask=gt_cell_box_assign > 0 # [side,side,num]
                one_obj_i_mask=gt_cell_asiign > 0 # [side,side]

                loss+=[
                    self.coord_scale*( ((out_loc-gt_loc_target)**2).sum(dim=3) [one_obj_i_j_mask]).sum()\
                    +self.object_scale*( ((out_conf-gt_iou_target)**2) [one_obj_i_j_mask] ).sum()\
                    +self.noobject_scale*( ((out_conf-0)**2) [(1-one_obj_i_j_mask)] ).sum()\
                    +self.class_scale*( ((out_cls-gt_cls_assign)**2).sum(dim=2) [one_obj_i_mask]).sum()
                ]

            return sum(loss)/len(loss)

            # print(ref_darknet)
            # exit(0)
        else:
            b_x,b_src_img_size=args
            raise NotImplementedError()
    