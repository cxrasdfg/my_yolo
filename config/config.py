# coding=utf-8

class CFG():
    voc_dir='/root/workspace/D/VOC2007_2012/VOCdevkit/VOC2007'

    device_id=0
    use_cuda=True

    weights_dir='./weights/'

    caffe_model="./models/vgg16_caffe_pretrain.pth"
    use_caffe=True

    loc_mean=[.0,.0,.0,.0]
    loc_std=[.1,.1,.2,.2]

    use_batchnorm=False

    intput_wh=300

    smin=.2
    smax=.9

    # aspect ratio
    ar=[[1,2],[1,2,3],[1,2,3],[1,2,3],[1,2],[1,2]]
    # about the net
    feat_map=[38,19,10,5,3,1] # h equals to w
    # feat_map=[37,18,9,5,3,1] # h equals to w
    det_in_channels=[512,1024,512,256,256,256]
    steps = [8, 16, 32, 64, 100, 300]

    l2norm_scale=20

    rand_seed=1234567

    batch_size=32
    num_worker=8
    
    clip=True
    freeze_top=False
    data_aug=True
    neg_ratio=3.0
    alpha=1.
    sigma=1.
    epochs=416
    save_per_epoch=32
    eval_per_epoch=32
    lr=1e-3
    lrs={'40000':lr,'50000':lr/10.,'60000':lr/100.}
    weight_decay=0.0005
    use_adam=False

    out_thruth_thresh=.5
    out_nms=.45
    out_nms_filter=.01
    out_box_num_per_im=200
    pos_thresh=.5

    eval_number=10000

    def _print(self):
        print('Config')
        print('{')
        for k in self._attr_list():
            print('%s=%s'% (k,getattr(self,k)) )
        print("}")
    @staticmethod

    def _attr_list():
        return [k for k in CFG.__dict__.keys() if not k.startswith('_') ] 
            
    # rpn path
   
cfg=CFG()
