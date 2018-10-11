import torch
from torch import nn
from base import BaseModel
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math
import json


def offsetFunc(netPred): #this changes the offset prediction from the network
    #YOLOv2,3 use sigmoid on activation to prevent predicting outside of cell impossible
    #But this probably makes it hard to predict on the edges?
    #so just
    return nn.tanh(netPred)
    # we offset by 0.5, so the center of a cell is when netPred is 0
    # tanh allows it to predict all the way to the center of its neighbor cell,
    # but this only occurs when netPred is +/- inf

def rotFunc(netPred):
    return math.pi/2 * netPred

def make_layers(cfg, batch_norm=False, instance_norm=False, use_weight_norm=False):
    modules = []
    in_channels = [cfg[0]]
    
    layers=[]
    layerCodes=[]
    for i,v in enumerate(cfg[1:]):
        if v == 'M':
            modules.append(nn.Sequential(*layers))
            layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            layerCodes = [v]
        elif type(v)==str and v[:2] == 'U+':
            if len(layers)>0:
                if type(layerCodes[0])==str and layerCodes[0][:2]=='U+':
                    layers[0].addConv(nn.Sequential(*layers[1:]))
                    modules.append(layers[0])
                else:
                    modules.append(nn.Sequential(*layers))
            layers = [up(in_channels[-1])]
            layerCodes = [v]

            in_channels.append(int(v[2:])+in_channels[-1])
        else:
            conv2d = nn.Conv2d(in_channels[-1], v, kernel_size=3, padding=1)
            #if i == len(cfg)-1:
            #    layers += [conv2d]
            #    break
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            elif instance_norm:
                layers += [conv2d, nn.InstanceNorm2d(v), nn.ReLU(inplace=True)]
            elif use_weight_norm:
                layers += [weight_norm(conv2d), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            layerCodes.append(v)
            in_channels.append(v)
    if len(layers)>0:
        if type(layerCodes[0])==str and layerCodes[0][:2]=='U+':
            layers[0].addConv(nn.Sequential(*layers[1:]))
            modules.append(layers[0])
        else:
            modules.append(nn.Sequential(*layers))
    return modules, in_channels[-1] #nn.Sequential(*layers)

class up(nn.Module):
    def __init__(self, in_ch, bilinear=True):
        super(up, self).__init__()
        self.outSize=in_ch
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            #self.up = nn.functional.interpolate
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

    def addConv(self,conv):
        self.conv=conv

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, math.ceil(diffX / 2),
                        diffY // 2, math.ceil(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class YoloBoxDetector(BaseModel):
    def __init__(self, config): # predCount, base_0, base_1):
        super(YoloBoxDetector, self).__init__(config)
        self.numBBTypes = config['number_of_box_types']
        self.numBBParams = 6 #conf,x-off,y-off,h-scale,w-scale,rot-off
        with open(config['anchors_file']) as f:
            self.anchors = json.loads(f.read()) #array of objects {rot,height,width}
        #TODO Rescale anchors?
        self.numAnchors = len(self.anchors)
        self.predPointCount = config['number_of_point_types']
        self.predPixelCount = config['number_of_pixel_types']
        in_ch = 3 if 'color' not in config or config['color'] else 1
        if "norm_type" in config:
            batch_norm=config["norm_type"]=='batch_norm'
            instance_norm=config["norm_type"]=='instance_norm'
            weight_norm=config["norm_type"]=='weight_norm'
        else:
            batch_norm=False
            instance_norm=False
            weight_norm=False
        if not (batch_norm or instance_norm or weight_norm):
            print('Warning: YoloBoxDetector has no normalization!')
        #self.cnn, self.scale = vgg.vgg11_custOut(self.predLineCount*5+self.predPointCount*3,batch_norm=batch_norm, weight_norm=weight_norm)
        self.numOutBB = (self.numBBTypes+self.numBBParams)*self.numAnchors
        self.numOutPoint = self.predPointCount*3

        if 'down_layers_cfg' in config:
            layers_cfg = config['down_layers_cfg']
        else:
            layers_cfg=[in_ch,64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]

        self.net_down_modules, down_last_channels = make_layers(layers_cfg, batch_norm, instance_norm, weight_norm)
        self.net_down_modules.append(nn.Conv2d(down_last_channels, self.numOutBB+self.numOutPoint, kernel_size=1, padding=1))
        self._hack_down = nn.Sequential(*self.net_down_modules)
        self.scale=1
        for a in layers_cfg:
            if a=='M':
                self.scale*=2

        if self.predPixelCount>0:
            if 'up_layers_cfg' in config:
                up_layers_cfg =  config['up_layers_cfg']
            else:
                up_layers_cfg=[512, 'U+512', 256, 'U+256', 128, 'U+128', 64, 'U+64']
            self.net_up_modules, up_last_channels = make_layers(up_layers_cfg, batch_norm, weight_norm)
            self.net_up_modules.append(nn.Conv2d(up_last_channels, self.predPixelCount, kernel_size=1, padding=1))
            self._hack_up = nn.Sequential(*self.net_up_modules)

        #self.base_0 = config['base_0']
        #self.base_1 = config['base_1']

    def forward(self, img):
        #y = self.cnn(img)
        levels=[img]
        for module in self.net_down_modules:
            levels.append(module(levels[-1]))
        y=levels[-1]

        #priors_0 = Variable(torch.arange(0,y.size(2)).type_as(img.data), requires_grad=False)[None,:,None]
        priors_0 = torch.arange(0,y.size(2)).type_as(img.data)[None,:,None]
        priors_0 = (priors_0 + 0.5) * self.scale #self.base_0
        priors_0 = priors_0.expand(y.size(0), priors_0.size(1), y.size(3))
        priors_0 = priors_0[:,None,:,:]

        #priors_1 = Variable(torch.arange(0,y.size(3)).type_as(img.data), requires_grad=False)[None,None,:]
        priors_1 = torch.arange(0,y.size(3)).type_as(img.data)[None,None,:]
        priors_1 = (priors_1 + 0.5) * self.scale #elf.base_1
        priors_1 = priors_1.expand(y.size(0), y.size(2), priors_1.size(2))
        priors_1 = priors_1[:,None,:,:]

        anchor = self.anchors
        stackedPred=[]
        for i in range(self.numAnchors):

            offset = i*(self.numBBParams+self.numBBTypes)

            stackedPred += [
                torch.sigmoid(y[:,0+offset:1+offset,:,:]),                #0. confidence
                torch.tanh(y[:,1+offset:2+offset,:,:])*self.scale + priors_1,        #1. x-center
                torch.tanh(y[:,2+offset:3+offset,:,:])*self.scale + priors_0,        #2. y-center
                (math.pi/2)*torch.tanh(y[:,3+offset:4+offset,:,:]) + anchor[i]['rot'],      #3. rotation (radians)
                torch.exp(y[:,4+offset:5+offset,:,:]) * anchor[i]['height'], #4. height (half), I don't think this needs scaled
                torch.exp(y[:,5+offset:6+offset,:,:]) * anchor[i]['width'],  #5. width (half)  
            ]

            for j in range(self.numBBTypes):
                stackedPred.append(y[:,6+j+offset:7+j+offset,:,:])         #x. class prediction

        bbPredictions = torch.cat(stackedPred, dim=1)
        
        bbPredictions = bbPredictions.transpose(1,3).contiguous()#from [batch, channel, rows, cols] to [batch, cols, rows, channels]
        bbPredictions = bbPredictions.view(bbPredictions.size(0),-1,bbPredictions.size(3))#flatten to [batch, instances, channel]

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
            



        return bbPredictions, pointPreds, pixelPreds

