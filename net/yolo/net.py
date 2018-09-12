# coding=utf-8

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .net_tool import *
from .module import Conv2dLocal
from collections import OrderedDict

def ParserCfg(cfg_path):
    # parse the cfg file
    blocks=[]
    block=None
    file=codecs.open(cfg_path, 'rb+', encoding='utf-8')
    data=file.readlines()
    for num,line in enumerate(data):
        line=line.strip()
        if line.startswith('#') or line =='\n' \
                or line == '\r' or line =='':
            continue

        elif line.startswith('[net]'):
            assert block is None
            block=dict()
            block['type']='net'
            continue

        elif line.startswith('[convolutional]'):
            # print(line)
            blocks.append(block)
            block = dict()
            block['type']='convolutional'

        elif line.startswith('[shortcut]'):
            blocks.append(block)
            block= dict()
            block['type']='shortcut'

        elif line.startswith('[yolo]'):
            blocks.append(block)
            block=dict()
            block['type']='yolo'

        elif line.startswith('[route]'):
            blocks.append(block)
            block=dict()
            block['type']='route'

        elif line.startswith('[upsample]'):
            blocks.append(block)
            block=dict()
            block['type']='upsample'

        elif line.startswith('[connected]'):
            blocks.append(block)
            block=dict()
            block['type']='connected'
        
        elif line.startswith('[dropout]'):
            blocks.append(block)
            block=dict()
            block['type']='dropout'
        
        elif line.startwith('[maxpool]'):
            blocks.append(block)
            block={}
            block['type']='maxpool'
        
        elif line.startswith('[local]'):
            blocks.append(block)
            block={}
            block['type']='local'
        elif line.startswith('[detection]'):
            blocks.append(block)
            block={}
            block['type']='detection'

        elif line.find('=') == -1:
            raise  Exception(' File "%s", line:%d\nInvalid expression'%(cfg_path,num+1))

        else:
            lv=line.split('=')
            assert len(lv)==2

            lv,rv=lv[0].strip(),lv[1].strip()

            assert len(lv)!=0 and len(rv)!=0
            block[lv]=rv
    blocks.append(block)
    return blocks


class YoloLayer(torch.nn.Module):

    def __init__(self,mask,anchors,classes,num,jitter,
                 ignore_thresh,truth_thresh,random):
        super(YoloLayer, self).__init__()
        self.mask=mask
        self.anchors=anchors
        self.classes=classes
        self.num=num
        self.jitter=jitter
        self.ignore_thresh=ignore_thresh
        self.truth_thresh=truth_thresh
        self.random=random

        self.size_of_anchor=len(self.anchors)//self.num
        self.masked_anchor=[[self.anchors[i*self.size_of_anchor],
                             self.anchors[i*self.size_of_anchor+1]] for i in self.mask]

    def forward(self, x,net_height,net_width):
        batch,channel,height,width=x.shape

        m_anchor_num=len(self.masked_anchor)

        # transfer to batch * m_anchor_num * (width x height) * (5 + classes)
        x=x.view(
            [batch,channel,width*height]
        ).view(
            [batch,m_anchor_num,(5+self.classes),-1]
        ).transpose(2,3)

        coor_xy=x[:,:,:,:2]
        coor_wh=x[:,:,:,2:4]
        truth=x[:,:,:,4:5]
        class_prob=x[:,:,:,5:]

        coor_xy=F.sigmoid(coor_xy)
        truth=F.sigmoid(truth)
        class_prob=F.sigmoid(class_prob)

        grid_x=Variable(torch.linspace(0,width-1,width).float()).cuda()
        grid_y=Variable(torch.linspace(0,height-1,height).float()).cuda()

        grid_x=grid_x.expand(
            [batch,m_anchor_num,height,width]
        ).contiguous().view(
            [batch,m_anchor_num,-1]
        )

        grid_y=grid_y.expand(
            [batch,m_anchor_num,height,width]
        ).transpose(2,3).contiguous().view(
            [batch,m_anchor_num,-1]
        )

        # add offset
        # coor_xy[:,:,:,0]+=grid_x
        # coor_xy[:,:,:,1]+=grid_y

        grid_xy=torch.cat(
            [grid_x.view(batch, m_anchor_num, -1, 1)
                , grid_y.view(batch, m_anchor_num, -1, 1)
             ]
            , dim=3
        )
        coor_xy+=grid_xy
        # renormalize
        # coor_xy[:,:,:,0]/=width
        # coor_xy[:,:,:,1]/=height
        coor_xy/=NP2Variable(np.array([int(width),int(height)]))

        #anchor
        m_anchor=np.array(self.masked_anchor,dtype=np.float32)

        # normalization
        m_anchor[:,0]/=net_width
        m_anchor[:,1]/=net_height

        m_anchor=NP2Variable(m_anchor)
        m_anchor=m_anchor.expand(
            [batch,m_anchor_num,2]
        ).contiguous().view(
            [batch, m_anchor_num,1, 2]
        ).expand(batch,m_anchor_num,width * height,2)
        coor_wh=torch.exp(coor_wh)*m_anchor

        x=torch.cat([coor_xy,coor_wh,truth,class_prob],dim=3)

        return x


class DetectionLayer(torch.nn.Module):
    def __init__(self,classes,
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
        coord_scale):
        
        super(DetectionLayer,self).__init__()
        
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
        coord_scale
        
    def forward(self,x):
        raise NotImplementedError()

class Conv2D(nn.Module):
    def __init__(self,input_channel,output_channel,
                 size,stride,same_padding,bn,activation):
        super(Conv2D, self).__init__()
        if bn:
            bias=False
        else:
            bias=True
        padding=int(size-1)//2 if same_padding else 0
        self.conv=nn.Conv2d(input_channel,output_channel,
                            size,stride,padding,bias=bias)
        self.bn=nn.BatchNorm2d(output_channel) if bn else None
        if activation=='linear':
            self.act=None
        elif activation=='leaky':
            self.act=nn.LeakyReLU(0.1,inplace=True)
        else:
            raise ValueError('Specify `linear` or `leaky` for activation')

    def forward(self, x):
        x=self.conv(x)
        if self.bn:
            x=self.bn(x)
        if self.act is not None:
            x=self.act(x)
        return x

class Linear(nn.Module):
    def __init__(self,in_features,out_features,act,bn=False):
        super(Linear,self).__init__()
        self.lin=torch.nn.Linear(in_features,out_features,bias=not bn)
        
        self.bn=torch.nn.BatchNorm2d(out_features) if bn else None

        if act == 'linear':
            self.act=None
        elif act=='leaky':
            self.act=nn.LeakyReLU(0.1,inplace=True)
        else:
            raise ValueError('attr::`act` must be `linear` or `leaky`')
    
    def forward(self,x):
        x=self.lin(x)
        if self.bn:
            x=self.bn(x)
        if self.act:
            x=self.act(x)
        return x

class RouteLayer(nn.Module):
    """
    this layer is just a placeholder,
    do not use it in forwarding...
    """
    def __init__(self,route_layers):
        super(RouteLayer, self).__init__()
        self.route_layers=route_layers

    @DeprecationWarning
    def forward(self, x):
        raise ValueError('the function forward in this module has been deprecated')


class ShortcutLayer(nn.Module):
    def __init__(self,from_index,act):
        super(ShortcutLayer, self).__init__()
        self.from_index=from_index
        self.act=act

    def forward(self, x):
        raise ValueError('the function forward in this module has been deprecated')


class UpsampleLayer(nn.Module):
    def __init__(self,stride=2):
        super(UpsampleLayer, self).__init__()
        self.stride=stride

    def forward(self, x):
        batch=x.shape[0]
        channel=x.shape[1]
        height=x.shape[2]
        width=x.shape[3]

        # wrong usage: x.view(batch,channel,1,width,1,height)...????????
        x=x.view(batch,
               channel,
               height,1,
               width,1).expand(batch,
                              channel,
                              height,
                              self.stride,
                              width,
                              self.stride).contiguous().view(batch,
                                                        channel,
                                                        self.stride*height,
                                                        self.stride*width)
        return x

class Darknet(nn.Module):
    def __init__(self,cfgfile,weight_file=None):
        if cfgfile.startswith('yolov3'):
            raise ValueError('only support YoloV3')
        super(Darknet, self).__init__()
        # self.net_property=None
        self.blocks=ParserCfg(cfgfile)
        self.layers=nn.Sequential

        # the net property
        self.batch = None
        self.subdivisions = None
        self.net_width = None
        self.net_height = None
        self.channels = None
        self.momentum = None
        self.decay = None
        self.angle = None
        self.saturation = None
        self.exposure = None
        self.hue = None

        self.learning_rate = None
        self.burn_in = None
        self.max_batches = None
        self.policy = None
        self.steps = None
        self.scales = None

        self.is_cuda=False

        self.create_net()

        if weight_file is not None:
            self.load_trained_weight(weight_file)
        # print(self.layers)

    def create_net(self):
        def cal_hw(im_size,kernel,stride,pad):
            padding=int(kernel-1)//2 if pad else 0
            return (im_size+2*padding-kernel)//stride+1
        
        idx=0
        LOC=[] # Layer Output Channel
        hwlist=[] # stores the height and width of feature map
        hwlist.append((self.net_height,self,net_width))

        layers=[]
        first_connected=True

        for block in self.blocks:
            # print(block)
            bType=block['type']
            if bType=='net':
                idx+=1
                # self.net_property=block
                self.batch = int(block['batch'])
                self.subdivisions = int(block['subdivisions'])
                self.net_width = int(block['width'])
                self.net_height = int(block['height'])

                assert self.net_width == self.net_height

                self.channels = int(block['channels'])
                self.momentum = float(block['momentum'])
                self.decay = float(block['decay'])
                self.angle = float(block['angle'])
                self.saturation = float(block['saturation'])
                self.exposure = float(block['exposure'])
                self.hue = float(block['hue'])

                self.learning_rate = float(block['learning_rate'])
                self.burn_in = int(block['burn_in'])
                self.max_batches = int(block['max_batches'])
                self.policy = block['policy']

                steps=[]
                for t in block['steps'].split(','):
                    steps.append(int(t))
                self.steps = steps

                scales=[]
                for t in block['scales'].split(','):
                    scales.append(float(t))
                self.scales = scales

                LOC.append(int(block['channels']))

            elif bType=='convolutional':
                out_channels=int(block['filters'])
                bn=False
                if block.get('batch_normalize') is not None:
                    if block['batch_normalize'] == '1':
                        bn=True
                _kernel=int(block['size'])
                _stride=int(block['stride'])
                _pad=int(block['pad'])
                layers.append(['Conv%d'%(len(layers)),
                               Conv2D(LOC[-1],out_channels,_kernel,
                       _stride,_pad,bn,block['activation'])])
                
                LOC.append(out_channels)

                # append feature map size
                ch,cw=hwlist[-1]
                ch=cal_hw(ch,_kernel,_stride,_pad)
                cw=cal_hw(cw,_kernel,_stride,_pad)
                hwlist.append((ch,cw))

            elif bType=='connected':
                if first_connected:
                    first_connected=False
                    in_features= hwlist[-1][0]*\
                        hwlist[-1][1]*\
                        out_channels[-1]
                else:
                    in_features=out_channels[-1]
                out_features=int(block['output'])
                bn=False
                if block.get('batch_normalize') is not None:
                    if block['batch_normalize'] == '1':
                        bn=True
                layers.append(['Linear%d'%(len(layers)),Linear(
                    in_features,out_features,act=block['activation'],bn=bn) ])
                LOC.append(out_features)
            
            elif bType=='maxpool':
                _size=int(block['size'])
                _stride=int(block['stride'])
                layers.append(['Maxpool%d'%(len(layers)),torch.nn.MaxPool2d(_size,_stride)])
                LOC.append(LOC[-1])

                ch,cw=hwlist[-1]
                ch=cal_hw(ch,_size,_stride,pad=False)
                cw=cal_hw(cw,_size,_stride,pad=False)
                hwlist.append((ch,cw))

            elif bType=='local':
                _size=int(block['size'])
                _stride=int(block['stride'])
                _pad=int(block['pad'])
                _filters=int(block['filters'])
                _activation=int(block['activation'])
                
                padding=int(_size-1)//2 if _pad else 0
                ch,cw=hwlist[-1]
                layers.append(['Local%d'%len(layers),
                    Conv2dLocal(ch,cw,LOC[-1],_filters,_size,_stride,padding)])

                LOC.append(_filters)

                ch=cal_hw(ch,_size,_stride,_pad)
                cw=cal_hw(cw,_size,_stride,_pad)
                hwlist.append((ch,cw))
            
            elif bType=='detection':
                classes=int(block['classes'])
                coords=int(block['coords'])
                rescore=int(block['rescore'])
                side=int(block['side'])
                num=int(block['num'])
                softmax=int(block['softmax'])
                sqrt=int(block['sqrt'])
                jitter=float(block['jitter'])

                object_scale=int(block['object_scale'])
                noobject_scale=float(block['noobject_scale'])
                class_scale=int(block['class_scale'])
                coord_scale=int(block['coord_scale'])

                assert LOC[-1]==side*side*(num*(coords+rescore)+classes)
                
                layers.append(['Detection%d'%(len(layers)),
                    DetectionLayer(
                        classes,coords,rescore,
                        side,num,softmax,sqrt,jitter,
                        object_scale,noobject_scale,class_scale,coord_scale)])

                LOC.append('detection pos')

            elif bType=='dropout':
                probability=float(block['probability'])
                layers.append(['Dropout%d'%(len(layers)),torch.nn.Dropout(probability)])
                LOC.append(LOC[-1])

            elif bType=='shortcut':
                sFrom=int(block['from'])
                assert sFrom<0
                layers.append(
                    ['Shortcut%d'%(len(layers)),
                    ShortcutLayer(sFrom+len(layers),block['activation'])]
                )

                LOC.append(LOC[sFrom])

            elif bType=='route':
                temp=[]
                channel_sum=0
                for t in block['layers'].split(','):
                        if int(t)<0:
                            temp.append(int(t)+len(layers))
                        else:
                            temp.append(int(t))
                        channel_sum+=LOC[int(t)]
                assert len(temp) != 0
                layers.append(['Route%d'%(len(layers)),RouteLayer(temp)])
                LOC.append(channel_sum)

            elif bType=='upsample':

                layers.append(['Upsample%d'%(len(layers)),UpsampleLayer()])
                LOC.append(LOC[-1])
            
            elif bType=='yolo':
                mask=[]
                for n in block['mask'].split(','):
                    mask.append(int(n))
                anchors=[]
                for n in block['anchors'].split(','):
                    anchors.append(int(n))

                classes=int(block['classes'])
                num=int(block['num'])
                jitter=float(block['jitter'])
                ignore_thresh=float(block['ignore_thresh'])
                truth_thresh=int(block['truth_thresh'])
                random=int(block['random'])

                assert LOC[-1]==len(mask)*(5+classes)

                layers.append(['Yolo%d'%(len(layers)),
                               YoloLayer(mask,anchors,classes,
                                         num,jitter,ignore_thresh,
                                         truth_thresh,random)])
                LOC.append('yolo pos')

        self.layers=nn.Sequential(OrderedDict(layers))

    def forward(self,x):
        output=[]  # save the reference for the output of corresponding layer
        res=[]
        idx=0
        first_connected=True
        for name,module in self.layers.named_children():
            if name.startswith('Conv'):
                # if idx==85:
                #     print(idx)
                x=module(x)
                output.append(x)

            elif name.startwith('Linear'):
                if first_connected:
                    first_connected=False
                    b,_,_,_=x.shape
                    x=x.view(b,-1)
                
                x=module(x)
                output.append(x)
            
            elif name.startwith('Maxpool'):
                x=module(x)
                output.append(x)

            elif name.startswith('Local'):
                x=module(x)
                output.append(x)

            elif name.startwith('Dropout'):
                x=module(x)
                output.append(x)

            elif name.startswith('Shortcut'):
                last_x=output[module.from_index]
                x=x+last_x
                if module.act != 'linear':
                    assert 0
                output.append(x)
            
            elif name.startswith('Route'):
                x=None
                for route_layer in module.route_layers:
                    if x is None:
                        x=output[route_layer]
                    else:
                        route_layer=output[route_layer]
                        assert route_layer.shape[2]==x.shape[2] and \
                            route_layer.shape[3]==x.shape[3]
                        x=torch.cat([x,route_layer],dim=1)
                assert x is not None
                output.append(x)

            elif name.startswith('Upsample'):
                x=module(x)
                output.append(x)

            elif name.startswith('Yolo'):
                x=module(x,self.net_height,self.net_width)
                output.append('yolo output,just for taking a place')
                res.append(x)
            
            elif name.startswith('Detection'):
                x=module(x)
                output.append('detection output, take a place')
                res.append(x)
                
            else:
                raise ValueError('unrecognized layer...')

            idx+=1
        # for i,val in enumerate(output):
        #     torch.save(val,'./mine/%d'%(i))
        if len(res) == 0:
            res=output[-1]
        return res

    def load_trained_weight(self,file_name):
        file=open(file_name,"rb+")
        header=np.fromfile(file,dtype=np.int32,count=5)
        del header

        buffer=np.fromfile(file,dtype=np.float32)
        start=0
        for name,layer in self.layers.named_children():
            if isinstance(layer,Conv2D):
                if layer.bn is not None:
                    num_b = layer.bn.bias.numel()
                    layer.bn.bias.data.copy_(torch.from_numpy(buffer[start:start + num_b]))
                    start = start + num_b
                    layer.bn.weight.data.copy_(torch.from_numpy(buffer[start:start + num_b]))
                    start = start + num_b
                    layer.bn.running_mean.copy_(torch.from_numpy(buffer[start:start + num_b]))
                    start = start + num_b
                    layer.bn.running_var.copy_(torch.from_numpy(buffer[start:start + num_b]))
                    start = start + num_b
                else:
                    num_b=layer.conv.bias.numel()
                    layer.conv.bias.data.copy_(torch.from_numpy(buffer[start:start + num_b]))
                    start = start + num_b

                num_w = layer.conv.weight.numel()
                layer.conv.weight.data.copy_(torch.from_numpy(buffer[start:start + num_w]))
                start = start + num_w

            elif isinstance(layer,Linear):
                if layer.bn is not None:
                    num_b=layer.bn.bias.numel()
                    layer.bn.bias.data.copy_(torch.from_numpy(buffer[start:start+num_b]))
                    start=start + num_b
                    
                    num_w=layer.lin.weight.numel()
                    layer.lin.weight.data.copy_(torch.from_numpy(buffer[start:start+num_w]))
                    start = start + num_w

                    layer.bn.weight.data.copy_(torch.from_numpy(buffer[start:start + num_b]))
                    start = start + num_b

                    layer.bn.running_mean.copy_(torch.from_numpy(buffer[start:start + num_b]))
                    start = start + num_b
                    
                    layer.bn.running_var.copy_(torch.from_numpy(buffer[start:start + num_b]))
                    start = start + num_b
                    
                else:   
                    num_b=layer.lin.bias.numel()
                    layer.lin.bias.data.copy_(torch.from_numpy(buffer[start:start+num_b]))
                    start = start +num_b

                    num_w=layer.lin.weight.numel()
                    layer.lin.weight.data.copy_(torch.from_numpy(buffer[start:start+num_w]))
                    start = start + num_w

            else:
                pass


if __name__ == '__main__':
    print('This is the ugly implementation of yolov3...')
    img_path = './data/dog.jpg'
    bs = Darknet('cfg/yolov3.cfg', 'yolov3.weights')
    bs.eval()
    bs.cuda()
    test_x = LoadImgForward(img_path, (bs.net_height, bs.net_width))

    test_y = bs(test_x)

    con_y = None
    for y in test_y:
        if con_y is None:
            con_y = y
        else:
            con_y = torch.cat([con_y, y], dim=2)

    final_batch_boxes = []
    batch_boxes = GetBoxFromNetOutput(con_y)  # batch x (m_anchor_num * width * height) x (4 + 1 + classes)
    for boxes in batch_boxes:
        temp = nms_sort(boxes)  # num1 x (4 + 1 + classes)
        temp = GetBoxesForShow(temp)  # num2 x (4 + class_id)
        final_batch_boxes.append(temp)

        DrawBoxOnImg(img_path, temp, './data/coco.names')