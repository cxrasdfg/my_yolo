# coding=utf-8
# author=theppsh

from config import cfg
import numpy as np 
import torch

# hold the rand_seed...
np.random.seed(cfg.rand_seed)
torch.manual_seed(cfg.rand_seed)
torch.cuda.manual_seed(cfg.rand_seed)

from net import Darknet
from data import TrainDataset,\
    eval_net,get_check_point,\
    read_image,TestTransform,\
    draw_bbox as draw_box,show_img
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

def adjust_lr(opt,iters,lrs=cfg.lrs):
    lr=0
    for k,v in lrs.items():
        lr=v
        if iters<int(k):
            break

    for param_group in opt.param_groups:
        
        param_group['lr'] = lr


def train():
    cfg._print()
   
    # im=torch.randn([1,3,300,300])
    # for loc,cls in net(im):
        # print(loc.shape,cls.shape)
    # for k,v in net.named_parameters():
    #     print(k,v,v.shape)

    # boxes=get_default_boxes()
    # print(boxes.shape)
    data_set=TrainDataset(416,416)

    data_loader=DataLoader(
        data_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_worker
    )
    
    # test the dataloader...
    # for b_imgs,b_boxes,b_labels,b_real_box_num in tqdm(data_loader):
        # tqdm.write('%s %s %s'%(str(b_imgs.shape),str(b_imgs.max()),str(b_imgs.min())) )

    exit(0)
    # NOTE: plus one, en?
    net=SSD(len(data_set.classes)+1)
    net._print()
    epoch,iteration,w_path=get_check_point()
    if w_path:
        model=torch.load(w_path)
        net.load_state_dict(model)
        print("Using the model from the last check point:%s"%(w_path) )
        epoch+=1

    net.train()
    is_cuda=cfg.use_cuda
    did=cfg.device_id
    if is_cuda:
        net.cuda(did)

    while epoch<cfg.epochs:
        
        # print('********\t EPOCH %d \t********' % (epoch))
        for i,(imgs,targets,labels) in tqdm(enumerate(data_loader)):
            if is_cuda:
                imgs=imgs.cuda(did)
                targets=targets.cuda(did)
                labels=labels.cuda(did)

            _loss=net.train_once(imgs,targets,labels)
            tqdm.write('Epoch:%d, iter:%d, loss:%.5f'%(epoch,iteration,_loss))

            iteration+=1
            adjust_lr(net.optimizer,iteration,cfg.lrs)

        if epoch % cfg.save_per_epoch ==0:
            torch.save(net.state_dict(),'%sweights_%d_%d'%(cfg.weights_dir,epoch,iteration) )

        if epoch % cfg.eval_per_epoch==0:
            _map= eval_net(net=net,num=100,shuffle=True)['map']
            print("map:",_map)
            
        epoch+=1

    print(eval_net(net=net))

def test_net():
    data_set=TrainDataset()

    classes=data_set.classes
    net=SSD(len(classes)+1)
    _,_,last_time_model=get_check_point()
    # assign directly
    # last_time_model='./weights/weights_21_110242'

    if os.path.exists(last_time_model):
        model=torch.load(last_time_model)
        net.load_state_dict(model)
        print("Using the model from the last check point:`%s`"%(last_time_model))
    else:
        raise ValueError("no model existed...")

    net.eval()
    is_cuda=cfg.use_cuda
    did=cfg.device_id

    img_src=read_image('./data/img/dog.jpg')
    w,h=img_src.size

    img=TestTransform(img_src,torch.tensor([[0,0,1,1]]).float()) # [c,h,w]
    img=img[None]

    if is_cuda:
        net.cuda(did)
        img=img.cuda(did)
    boxes,labels,probs=net.predict(img,torch.tensor([[w,h]]).type_as(img))[0]

    prob_mask=probs>cfg.out_thruth_thresh
    boxes=boxes[prob_mask ] 
    labels=labels[prob_mask ].long()
    probs=probs[prob_mask]

    img_src=np.array(img_src) # [h,w,3] 'RGB'
    # change to 'BGR'
    img_src=img_src[:,:,::-1].copy()

    if len(boxes) !=0:
        draw_box(img_src,boxes,color='pred',
            text_list=[ 
                classes[_]+'[%.3f]'%(__)  for _,__ in zip(labels,probs)
            ]
        )
    show_img(img_src,-1)

if __name__ == '__main__':
    m=Darknet('./net/yolo/cfg/yolo.cfg','./models/extraction.weights')
    m.train()

    m._print()
    # m(torch.randn([1,3,416,416]))
    
    exit(0)

    if len(sys.argv)==1:
        opt='train'
    else:
        opt=sys.argv[1]
    if opt=='train':
        train()
    elif opt=='test':
        test_net()
    elif opt=='eval':
        print(eval_net())
    else:
        raise ValueError('opt shuold be in [`train`,`test`,`eval`]')
    # train()
