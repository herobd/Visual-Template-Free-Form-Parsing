import torch
from torch import nn
from base import BaseModel
import math
import json
import numpy as np
from .net_builder import make_layers





class YoloBoxDetector(nn.Module): #BaseModel
    def __init__(self, config): # predCount, base_0, base_1):
        #super(YoloBoxDetector, self).__init__(config)
        super(YoloBoxDetector, self).__init__()
        self.forPairing=False
        self.config = config
        self.rotation = config['rotation'] if 'rotation' in config else True
        self.numBBTypes = config['number_of_box_types']
        self.numBBParams = 6 #conf,x-off,y-off,h-scale,w-scale,rot-off
        self.numLineParams = 5 #conf,x-off,y-off,h-scale,rot

        self.predPointCount = config['number_of_point_types'] if 'number_of_point_types' in config else 0
        self.predPixelCount = config['number_of_pixel_types'] if 'number_of_pixel_types' in config else 0
        self.predLineCount = config['number_of_line_types'] if 'number_of_line_types' in config else 0

        with open(config['anchors_file']) as f:
            self.anchors = json.loads(f.read()) #array of objects {rot,height,width}
        if self.rotation:
            self.meanH=48.0046359128/2
        else:
            self.meanH=62.1242376857/2
        self.numAnchors = len(self.anchors)
        if self.predLineCount>0:
            print('Warning, using hardcoded mean H (yolo_box_detector)')

        in_ch = 3 if 'color' not in config or config['color'] else 1
        norm = config['norm_type'] if "norm_type" in config else None
        if norm is None:
            print('Warning: YoloBoxDetector has no normalization!')
        dilation = config['dilation'] if 'dilation' in config else 1
        dropout = config['dropout'] if 'dropout' in config else None
        #self.cnn, self.scale = vgg.vgg11_custOut(self.predLineCount*5+self.predPointCount*3,batch_norm=batch_norm, weight_norm=weight_norm)
        self.numOutBB = (self.numBBTypes+self.numBBParams)*self.numAnchors
        self.numOutLine = (self.numBBTypes+self.numLineParams)*self.predLineCount
        self.numOutPoint = self.predPointCount*3

        if 'down_layers_cfg' in config:
            layers_cfg = config['down_layers_cfg']
        else:
            layers_cfg=[in_ch,64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]

        self.net_down_modules, down_last_channels = make_layers(layers_cfg, dilation,norm,dropout=dropout)
        self.final_features=None 
        self.last_channels=down_last_channels
        self.net_down_modules.append(nn.Conv2d(down_last_channels, self.numOutBB+self.numOutLine+self.numOutPoint, kernel_size=1))
        self._hack_down = nn.Sequential(*self.net_down_modules)
        scaleX=1
        scaleY=1
        for a in layers_cfg:
            if a=='M' or (type(a) is str and a[0]=='D'):
                scaleX*=2
                scaleY*=2
            elif type(a) is str and a[0]=='U':
                scaleX/=2
                scaleY/=2
            elif type(a) is str and a[0:4]=='long': #long pool
                scaleX*=3
                scaleY*=2
        self.scale=(scaleX,scaleY)

        if self.predPixelCount>0:
            if 'up_layers_cfg' in config:
                up_layers_cfg =  config['up_layers_cfg']
            else:
                up_layers_cfg=[512, 'U+512', 256, 'U+256', 128, 'U+128', 64, 'U+64']
            self.net_up_modules, up_last_channels = make_layers(up_layers_cfg, 1, norm,dropout=dropout)
            self.net_up_modules.append(nn.Conv2d(up_last_channels, self.predPixelCount, kernel_size=1))
            self._hack_up = nn.Sequential(*self.net_up_modules)

        #self.base_0 = config['base_0']
        #self.base_1 = config['base_1']
        if 'DEBUG' in config:
            self.setDEBUG()

    def forward(self, img):
        #import pdb; pdb.set_trace()
        y = self._hack_down(img)
        if self.forPairing:
            return y[:,:(self.numBBParams+self.numBBTypes)*self.numAnchors,:,:]
        #levels=[img]
        #for module in self.net_down_modules:
        #    levels.append(module(levels[-1]))
        #y=levels[-1]


        #priors_0 = Variable(torch.arange(0,y.size(2)).type_as(img.data), requires_grad=False)[None,:,None]
        priors_0 = torch.arange(0,y.size(2)).type_as(img.data)[None,:,None]
        priors_0 = (priors_0 + 0.5) * self.scale[1] #self.base_0
        priors_0 = priors_0.expand(y.size(0), priors_0.size(1), y.size(3))
        priors_0 = priors_0[:,None,:,:].to(img.device)

        #priors_1 = Variable(torch.arange(0,y.size(3)).type_as(img.data), requires_grad=False)[None,None,:]
        priors_1 = torch.arange(0,y.size(3)).type_as(img.data)[None,None,:]
        priors_1 = (priors_1 + 0.5) * self.scale[0] #elf.base_1
        priors_1 = priors_1.expand(y.size(0), y.size(2), priors_1.size(2))
        priors_1 = priors_1[:,None,:,:].to(img.device)

        anchor = self.anchors
        pred_boxes=[]
        pred_offsets=[] #we seperate anchor predictions here. And compute actual bounding boxes
        for i in range(self.numAnchors):

            offset = i*(self.numBBParams+self.numBBTypes)
            if self.rotation:
                rot_dif = (math.pi/2)*torch.tanh(y[:,3+offset:4+offset,:,:])
            else:
                rot_dif = torch.zeros_like(y[:,3+offset:4+offset,:,:])

            stackedPred = [
                torch.sigmoid(y[:,0+offset:1+offset,:,:]),                #0. confidence
                torch.tanh(y[:,1+offset:2+offset,:,:])*self.scale[0] + priors_1,        #1. x-center
                torch.tanh(y[:,2+offset:3+offset,:,:])*self.scale[1] + priors_0,        #2. y-center
                rot_dif + anchor[i]['rot'],      #3. rotation (radians)
                torch.exp(y[:,4+offset:5+offset,:,:]) * anchor[i]['height'], #4. height (half), I don't think this needs scaled
                torch.exp(y[:,5+offset:6+offset,:,:]) * anchor[i]['width'],  #5. width (half)   as we scale the anchors in training
            ]

            #stackedOffsets = [
            #        y[:,0+offset:1+offset,:,:],
            #        y[:,1+offset:2+offset,:,:],
            #        y[:,2+offset:3+offset,:,:],
            #        y[:,4+offset:5+offset,:,:],
            #        y[:,4+offset:5+offset,:,:]
            #]
            #if self.rotation:
            #    stackedOffsets.append( rot_dif )

            for j in range(self.numBBTypes):
                stackedPred.append(y[:,6+j+offset:7+j+offset,:,:])         #x. class prediction
                #stackedOffsets.append(y[:,6+j+offset:7+j+offset,:,:])         #x. class prediction
            pred_boxes.append(torch.cat(stackedPred, dim=1))
            #pred_offsets.append(torch.cat(stackedOffsets, dim=1))
            pred_offsets.append(y[:,offset:offset+self.numBBParams+self.numBBTypes,:,:])

        if len(pred_boxes)>0:
            bbPredictions = torch.stack(pred_boxes, dim=1)
            offsetPredictions = torch.stack(pred_offsets, dim=1)
            
            bbPredictions = bbPredictions.transpose(2,4).contiguous()#from [batch, anchors, channel, rows, cols] to [batch, anchros, cols, rows, channels]
            bbPredictions = bbPredictions.view(bbPredictions.size(0),bbPredictions.size(1),-1,bbPredictions.size(4))#flatten to [batch, anchors, instances, channel]
            #avg_conf_per_anchor = bbPredictions[:,:,:,0].mean(dim=0).mean(dim=1)
            bbPredictions = bbPredictions.view(bbPredictions.size(0),-1,bbPredictions.size(3)) #[batch, instances+anchors, channel]

            offsetPredictions = offsetPredictions.permute(0,1,3,4,2).contiguous()
        else:
            bbPredictions=None
            offsetPredictions=None

        linePreds=[]
        offsetLinePreds=[]
        for i in range(self.predLineCount):
            offset = i*(self.numLineParams+self.numBBTypes) + self.numAnchors*(self.numBBParams+self.numBBTypes)
            stackedPred=[
                torch.sigmoid(y[:,0+offset:1+offset,:,:]),                          #confidence
                torch.tanh(y[:,1+offset:2+offset,:,:])*self.scale[0] + priors_1,       #x-center
                torch.tanh(y[:,2+offset:3+offset,:,:])*self.scale[1] + priors_0,       #y-center
                (math.pi)*torch.tanh(y[:,3+offset:4+offset,:,:]),                 #rotation (radians)
                torch.exp(y[:,4+offset:5+offset,:,:])*self.meanH                    #scale (half-height),
                
            ]
            for j in range(self.numBBTypes):
                stackedPred.append(y[:,5+j+offset:6+j+offset,:,:])         #x. class prediction

            predictions = torch.cat(stackedPred, dim=1)
            predictions = predictions.transpose(1,3).contiguous()#from [batch, channel, rows, cols] to [batch, cols, rows, channels]
            predictions = predictions.view(predictions.size(0),-1,predictions.size(3))#flatten to [batch, instances, channel]
            linePreds.append(predictions)

            offsets = y[:,offset:offset+self.numLineParams+self.numBBTypes,:,:]
            offsets = offsets.permute(0,2,3,1).contiguous()
            offsetLinePreds.append(offsets)

        pointPreds=[]
        for i in range(self.predPointCount):
            offset = i*3 + self.numAnchors*(self.numBBParams+self.numBBTypes)
            predictions = torch.cat([
                torch.sigmoid(y[:,0+offset:1+offset,:,:]),    #confidence
                y[:,1+offset:2+offset,:,:] + priors_1,        #x
                y[:,2+offset:3+offset,:,:] + priors_0         #y
            ], dim=1)
            
            predictions = predictions.transpose(1,3).contiguous()#from [batch, channel, rows, cols] to [batch, cols, rows, channels]
            predictions = predictions.view(predictions.size(0),-1,3)#flatten to [batch, instances, channel]
            pointPreds.append(predictions)

        pixelPreds=None
        if self.predPixelCount>0:
            y2=levels[-2]
            p=-3
            for module in self.net_up_modules[:-1]:
                #print('uping {} , {}'.format(y2.size(), levels[p].size()))
                y2 = module(y2,levels[p])
                p-=1
            pixelPreds = self.net_up_modules[-1](y2)
            



        return bbPredictions, offsetPredictions, linePreds, offsetLinePreds, pointPreds, pixelPreds #, avg_conf_per_anchor

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Trainable parameters: {}'.format(params))
        print(self)

    def setForPairing(self):
        self.forPairing=True
        def save_final(module,input,output):
            self.final_features=output
        self.net_down_modules[-2].register_forward_hook(save_final)
    def setForGraphPairing(self,beginningOfLast=False,featuresFromHere=-1):
        def save_final(module,input,output):
            self.final_features=output
        if beginningOfLast:
            self.net_down_modules[-2][0].register_forward_hook(save_final) #after max pool
            self.last_channels= self.last_channels//2 #HACK
        else:
            if type( self.net_down_modules[-2][featuresFromHere]) == torch.nn.modules.activation.ReLU:
                self.net_down_modules[-2][featuresFromHere].register_forward_hook(save_final)
            else:
                print('Layer {} of the final conv block was specified, but it is not a ReLU layer. Did you choose the right layer?'.format(featuresFromHere))
                exit()
    def setDEBUG(self):
        #self.debug=[None]*5
        #for i in range(0,1):
        #    def save_layer(module,input,output):
        #        self.debug[i]=output.cpu()
        #    self.net_down_modules[i].register_forward_hook(save_layer)

        def save_layer0(module,input,output):
            self.debug0=output.cpu()
        self.net_down_modules[0].register_forward_hook(save_layer0)
        def save_layer1(module,input,output):
            self.debug1=output.cpu()
        self.net_down_modules[1].register_forward_hook(save_layer1)
        def save_layer2(module,input,output):
            self.debug2=output.cpu()
        self.net_down_modules[2].register_forward_hook(save_layer2)
        def save_layer3(module,input,output):
            self.debug3=output.cpu()
        self.net_down_modules[3].register_forward_hook(save_layer3)
        def save_layer4(module,input,output):
            self.debug4=output.cpu()
        self.net_down_modules[4].register_forward_hook(save_layer4)
