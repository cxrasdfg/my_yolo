# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from .net_tool import *
from .layer import Conv2dLocalLayer,DetectionLayer,\
    LinearLayer,RouteLayer,ShortcutLayer,\
    UpsampleLayer,YoloLayer,Conv2dLayer

from .parser import ParserCfg
from collections import OrderedDict

def cal_hw(im_size,kernel,stride,pad):
            padding=int(kernel-1)//2 if pad else 0
            return (im_size+2*padding-kernel)//stride+1

class Darknet(nn.Module):
    def __init__(self,cfgfile,weight_file=None):
        # if not cfgfile.startswith('yolov3'):
            # raise ValueError('only support YoloV3')
        
        if 'yolo' not in cfgfile:
            raise ValueError('only support yolo...')

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

        # get optimizer...
        self.get_optimizer()

    def create_net(self):
        idx=0
        LOC=[] # Layer Output Channel
        hwlist=[] # stores the height and width of feature map

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
                self.angle = float(block['angle']) if 'angle' in block else None
                self.saturation = float(block['saturation'])
                self.exposure = float(block['exposure'])
                self.hue = float(block['hue'])

                self.learning_rate = float(block['learning_rate'])
                self.burn_in = int(block['burn_in']) if 'burn_in' in block else None
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

                hwlist.append((self.net_height,self.net_width) )
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
                               Conv2dLayer(LOC[-1],out_channels,_kernel,
                       _stride,_pad,bn,block['activation'])])
                
                LOC.append(out_channels)

                # append feature map size
                ch,cw=hwlist[-1]
                # print(block)

                ch=cal_hw(ch,_kernel,_stride,_pad)
                cw=cal_hw(cw,_kernel,_stride,_pad)
                # print("%dx%d"%(ch,cw))
                hwlist.append((ch,cw))

            elif bType=='connected':
                if first_connected:
                    first_connected=False
                    in_features= hwlist[-1][0]*\
                        hwlist[-1][1]*\
                        LOC[-1]
                else:
                    in_features=LOC[-1]
                out_features=int(block['output'])
                bn=False
                if block.get('batch_normalize') is not None:
                    if block['batch_normalize'] == '1':
                        bn=True
                layers.append(['Linear%d'%(len(layers)),LinearLayer(
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
                _activation=block['activation']
                
                padding=int(_size-1)//2 if _pad else 0
                ch,cw=hwlist[-1]
                layers.append(['Local%d'%len(layers),
                    Conv2dLocalLayer(ch,cw,LOC[-1],_filters,_size,_stride,padding,act=_activation)])

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
    
    def get_optimizer(self):
        params=[]
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                params += [{'params': [value], 'lr': self.learning_rate, 'weight_decay': self.decay}]
        
        print("Using SGD optimizer")
        self.optimizer = torch.optim.SGD(params, momentum=self.momentum)
        return self.optimizer
    
    def opt_step(self,loss):
        r"""Use the loss to backward the net, 
        then the optimizer will update the weights
        which requires grad...
        Args:
            loss (list[tensor]): list of the tensor
        Return:
            loss (tensor[float32]): value of current loss...
        """
        assert isinstance(loss,list)
        
        loss=sum(loss)/len(loss)
        
        # grad descend
        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.parameters(),10)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def forward(self,*args):
        if self.training:
            x,b_fixed_boxes,\
            b_fixed_labels,\
            b_real_box_num=args
        else:
            x,b_src_img_size=args
            
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

            elif name.startswith('Linear'):
                if first_connected:
                    first_connected=False
                    b,_,_,_=x.shape
                    x=x.view(b,-1)
                
                x=module(x)
                output.append(x)
            
            elif name.startswith('Maxpool'):
                x=module(x)
                output.append(x)

            elif name.startswith('Local'):
                x=module(x)
                output.append(x)

            elif name.startswith('Dropout'):
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
                # print(output)
                # for _ in output:
                    # print(_.shape)
                if self.training:
                    x=module(x,b_fixed_boxes,\
                        b_fixed_labels,\
                        b_real_box_num,self)
                else:
                    x=module(x,b_src_img_size)
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
        r"""This method will use the darknet pretrained weights,
        the weight file must be the darknet binary weight file...
        Args:
            file_name: name of the darknet weight file
        """
        file=open(file_name,"rb+")
        major=np.fromfile(file,dtype=np.int32,count=1)
        minor=np.fromfile(file,dtype=np.int32,count=1)
        revision=np.fromfile(file,dtype=np.int32,count=1)

        if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
            seen=np.fromfile(file,dtype=np.int64,count=1)
        else:
            seen=np.fromfile(file,dtype=np.int32,count=1)

        transpose=(major > 1000) or (minor > 1000)


        buffer=np.fromfile(file,dtype=np.float32)
        start=0
        loaded_count=0
        try:
            for name,layer in self.layers.named_children():
                if isinstance(layer,Conv2dLayer):
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
                    # NOTE: does the memory is stacked by this way?
                    layer.conv.weight.data.view(-1).copy_(torch.from_numpy(buffer[start:start + num_w]))
                    start = start + num_w
                    loaded_count+=1

                elif isinstance(layer,LinearLayer):
                    if layer.bn is not None:
                        num_b=layer.bn.bias.numel()
                        layer.bn.bias.data.copy_(torch.from_numpy(buffer[start:start+num_b]))
                        start=start + num_b
                        
                        num_w=layer.lin.weight.numel()
                        layer.lin.weight.data.view(-1).copy_(torch.from_numpy(buffer[start:start+num_w]))
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
                        layer.lin.weight.data.view(-1).copy_(torch.from_numpy(buffer[start:start+num_w]))
                        start = start + num_w
                    
                    loaded_count+=1

                else:
                    pass
        except RuntimeError:
            pass
        print("load pretrained weight: load %d layers" %(loaded_count) )
    
    def _print(self):
        nh=self.net_height
        nw=self.net_width
        nc=self.channels

        nf=1e100
        layer_out=[]  # layer output
        print("layer     filters    size              input                output")
        for counter,(name,layer) in enumerate(self.layers.named_children()):
            if name.startswith('Conv'):
                out_channels=layer.conv.out_channels
                kernel=layer.conv.kernel_size
                stride=layer.conv.stride
                assert stride[0] == stride[1]
                stride=stride[0]
                
                assert nc==layer.conv.in_channels
                new_nh=cal_hw(nh,kernel[0] ,stride,layer.conv.padding[0]!=0)
                new_nw=cal_hw(nw,kernel[1],stride,layer.conv.padding[0]!=0)
                print(\
                    "%5d conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d"
                    %(counter,out_channels,kernel[0],kernel[1],stride,nw,nh,nc,new_nw,new_nh,out_channels)
                    )
                
                nh=new_nh
                nw=new_nw
                nc=out_channels

                layer_out.append((nw,nh,nc))

            elif name.startswith('Linear'):
                
                print(\
                    "%5d connected                            %4d  ->  %4d"
                    %(counter,layer.lin.in_features,layer.lin.out_features)
                )
                nf=layer.lin.out_features

                layer_out.append(nf)
            
            elif name.startswith('Maxpool'):
                kernel=layer.kernel_size
                stride=layer.stride
                
                new_nh=cal_hw(nh,kernel,stride,layer.padding!=0)
                new_nw=cal_hw(nw,kernel,stride,layer.padding!=0)
                print(\
                    "%5d max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d"
                    %(counter,kernel,kernel,stride,nw,nh,nc,new_nw,new_nh,nc)
                )
                nh=new_nh
                nw=new_nw
                
                layer_out.append((nw,nh,nc))

            elif name.startswith('Local'):
                assert nc==layer.loc.in_channels
                out_channels=layer.loc.out_channels

                kernel=layer.loc.kernel_size
                stride=layer.loc.stride
                assert stride[0] == stride[1]
                stride=stride[0]

                new_nh=cal_hw(nh,kernel[0],stride,layer.loc.padding[0]!=0)
                new_nw=cal_hw(nw,kernel[1],stride,layer.loc.padding[0]!=0)
                print(\
                    "%5d Local Layer: %d x %d x %d image, %d filters -> %d x %d x %d image"
                    %(counter,nw,nh,nc,out_channels,new_nw,new_nh,out_channels)
                )

                nh=new_nh
                nw=new_nw
                nc=out_channels

                layer_out.append((nw,nh,nc))

            elif name.startswith('Dropout'):
                
                last_=layer_out[-1]
                if len(last_)==3:
                    nf=last_[0]*last_[1]*last_[2]
                elif len(last_)==1:
                    nf=last_
                else:
                    assert 0
                print(\
                    "%5d dropout       p = %.2f               %4d  ->  %4d"
                    %(counter,layer.p,nf,nf)
                )

                layer_out.append(nf)


            elif name.startswith('Shortcut'):
                
                nw,nh,nc=layer_out[-1]
                print(\
                    "%5d res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d"
                    %(counter,layer.from_index,nw,nh,nc,nw,nh,nc)
                )
                layer_out.append((nw,nh,nc))


            elif name.startswith('Route'):
                this_c=None
                this_h=None
                this_w=None
                print("%5d route "%(counter),end='')
                for idx in layer.route_layers:
                    if this_c is None:
                        this_w=layer_out[idx][0]  
                        this_h=layer_out[idx][1]  
                        this_c=layer_out[idx][2] 
                    else:
                        assert this_w==layer_out[idx][0]  
                        assert this_h==layer_out[idx][1]
                        this_c+=layer_out[idx][2] 
                    print(idx,end="")
                print()
                
                nw=this_w
                nh=this_h
                nc=this_c
        
                layer_out.append((nw,nh,nc))

            elif name.startswith('Upsample'):
                stride=layer.stride
                new_nh=new_nh*stride
                new_nw=new_nw*stride

                print("%5d upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d"\
                    %(counter,stride,nw,nh,nc,new_nw,new_nh,nc))
                
                nh=new_nh
                nw=new_nw
                
                layer_out.append((nw,nh,nc))

            elif name.startswith('Yolo'):
                print('%5d yolo'%(counter))
                layer_out.append('hold a place')
            
            elif name.startswith('Detection'):
                print('%5d Detection Layer'%(counter))
                layer_out.append('hold a place')
            


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