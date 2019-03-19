
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math
import numpy as np
from .coordconv import CoordConv

def primeFactors(n):
    ret = [1]
    # Print the number of two's that divide n 
    while n % 2 == 0:
        if len(ret)==0:
            ret.append(2)
        n = n / 2

    # n must be odd at this point 
    # so a skip of 2 ( i = i + 2) can be used 
    for i in range(3,int(math.sqrt(n))+1,2):

        # while i divides n , print i ad divide n 
        while n % i== 0:
            ret.append(i)
            n = n / i

    # Condition if n is a prime 
    # number greater than 2 
    if n > 2:
        ret.append(n)
    return ret

def getGroupSize(channels,goalSize=None):
    if goalSize is None:
        if channels>=32:
            goalSize=8
        else:
            goalSize=4
    if channels%goalSize==0:
        return goalSize
    factors=primeFactors(channels)
    bestDist=9999
    for f in factors:
        if abs(f-goalSize)<=bestDist: #favor larger
            bestDist=abs(f-goalSize)
            bestGroup=f
    return int(bestGroup)

class ncReLU(nn.Module):
    def __init__(self):
        super(ncReLU, self).__init__()
        self.r = nn.ReLU(inplace=False)
    def forward(self,input):
        return torch.cat([self.r(input), -self.r(-input)], 1)

#ResNet block based on:
 #No projection in the residual network https://link.springer.com/content/pdf/10.1007%2Fs10586-017-1389-z.pdf
class ResBlock(nn.Module):
    def __init__(self,in_ch,out_ch,dilation=1,norm='',downsample=False, dropout=None, secondKernel=3):
        super(ResBlock, self).__init__()
        layers=[]
        skipFirstReLU=False
        if in_ch!=out_ch:
            assert(out_ch==2*in_ch)
            layers.append(ncReLU())
            skipFirstReLU=True
        if downsample:
            layers.append(nn.AvgPool2d(2)) #could be learned, but this allows a better identity?
        if len(layers)>0:
            self.transform = nn.Sequential(*layers)
        else:
            self.transform = lambda x: x

        layers=[]
        if not skipFirstReLU:
            #I'm not sure if this is the best thing
            #there should be a way to normalize (mask after normalization?)
            if norm=='batch_norm':
                layers.append(nn.BatchNorm2d(out_ch))
            if norm=='instance_norm':
                layers.append(nn.InstanceNorm2d(out_ch))
            if norm=='group_norm':
                layers.append(nn.GroupNorm(getGroupSize(out_ch),out_ch))
            layers.append(nn.ReLU(inplace=True)) 
        conv1=nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
        if norm=='weight_norm' or skipFirstReLU: #or just use this normalization?
            layers.append(weight_norm(conv1))
        else:
            layers.append(conv1)


        if norm=='batch_norm':
            layers.append(nn.BatchNorm2d(out_ch))
        if norm=='instance_norm':
            layers.append(nn.InstanceNorm2d(out_ch))
        if norm=='group_norm':
            layers.append(nn.GroupNorm(getGroupSize(out_ch),out_ch))
        if dropout is not None:
            if dropout==True or dropout=='2d':
                layers.append(nn.Dropout2d(p=0.1,inplace=True))
            elif dropout=='normal':
                layers.append(nn.Dropout2d(p=0.1,inplace=True))
        layers.append(nn.ReLU(inplace=True)) 
        assert(secondKernel%2 == 1)
        conv2=nn.Conv2d(out_ch, out_ch, kernel_size=secondKernel, padding=(secondKernel-1)//2)
        if norm=='weight_norm':
            layers.append(weight_norm(conv2))
        else:
            layers.append(conv2)

        self.side = nn.Sequential(*layers)

    def forward(self,x):
        x=self.transform(x)
        return x+self.side(x)

class GeneralRes(nn.Module):
    def __init__(self,ms,in_ch,out_ch,norm,dropout):
        layers=[]
        skipFirstReLU=False
        if in_ch!=out_ch:
            assert(out_ch==2*in_ch)
            layers.append(ncReLU())
            skipFirstReLU=True
        if downsample:
            layers.append(nn.AvgPool2d(2)) #could be learned, but this allows a better identity?
        if len(layers)>0:
            self.transform = nn.Sequential(*layers)
        else:
            self.transform = lambda x: x

        layers=[]
        if not skipFirstReLU:
            #I'm not sure if this is the best thing
            #there should be a way to normalize (mask after normalization?)
            if norm=='batch_norm':
                layers.append(nn.BatchNorm2d(out_ch))
            if norm=='instance_norm':
                layers.append(nn.InstanceNorm2d(out_ch))
            if norm=='group_norm':
                layers.append(nn.GroupNorm(getGroupSize(out_ch),out_ch))
            layers.append(nn.ReLU(inplace=True)) 
        if norm=='weight_norm' or skipFirstReLU: #or just use this normalization?
            layers.append(weight_norm(ms[0]))
        else:
            layers.append(ms[0])

        for m in ms[1:]:
            if norm=='batch_norm':
                layers.append(nn.BatchNorm2d(out_ch))
            if norm=='instance_norm':
                layers.append(nn.InstanceNorm2d(out_ch))
            if norm=='group_norm':
                layers.append(nn.GroupNorm(getGroupSize(out_ch),out_ch))
            if dropout is not None:
                if dropout==True or dropout=='2d':
                    layers.append(nn.Dropout2d(p=0.1,inplace=True))
                elif dropout=='normal':
                    layers.append(nn.Dropout2d(p=0.1,inplace=True))
            layers.append(nn.ReLU(inplace=True)) 
            assert(secondKernel%2 == 1)
            if norm=='weight_norm':
                layers.append(weight_norm(m))
            else:
                layers.append(m)

        self.side = nn.Sequential(*layers)

    def forward(self,x):
        x=self.transform(x)
        return x+self.side(x)



def convReLU(in_ch,out_ch,norm,dilation=1,kernel=3,dropout=None,depthwise=False,coordconv=None):
    if type(dilation) is int:
        dilation=(dilation,dilation)
    if type(kernel) is int:
        kernel=(kernel,kernel)
    padding = ( dilation[0]*(kernel[0]//2), dilation[1]*(kernel[1]//2) )
    groups = in_ch if depthwise else 1
    if coordconv is not None:
        conv2d = CoordConv(in_ch,out_ch, kernel_size=kernel, padding=padding,dilation=dilation,groups=groups,features=coordconv)
    else:
        conv2d = nn.Conv2d(in_ch,out_ch, kernel_size=kernel, padding=padding,dilation=dilation,groups=groups)
    #if i == len(cfg)-1:
    #    layers += [conv2d]
    #    break
    if norm=='weight_norm':
        layers = [weight_norm(conv2d)]
    else:
        layers = [conv2d]
    if norm=='batch_norm':
        layers.append(nn.BatchNorm2d(out_ch))
    elif norm=='instance_norm':
        layers.append(nn.InstanceNorm2d(out_ch))
    elif norm=='group_norm':
        layers.append(nn.GroupNorm(getGroupSize(out_ch),out_ch))
    if dropout is not None:
        if dropout==True or dropout=='2d':
            layers.append(nn.Dropout2d(p=0.1,inplace=True))
        elif dropout=='normal':
            layers.append(nn.Dropout(p=0.1,inplace=True))
        elif type(dropout)==float:
            layers.append(nn.Dropout2d(p=dropout,inplace=True))
    layers += [nn.ReLU(inplace=True)]
    return layers

def fcReLU(in_ch,out_ch,norm,dropout=None,relu=True):
    fc = nn.Linear(in_ch,out_ch)
    if norm=='weight_norm':
        layers = [weight_norm(fc)]
    else:
        layers = [fc]
    if norm=='batch_norm':
        layers.append(nn.BatchNorm1d(out_ch))
    elif norm=='instance_norm':
        layers.append(nn.InstanceNorm1d(out_ch))
    elif norm=='group_norm':
        layers.append(nn.GroupNorm(getGroupSize(out_ch),out_ch))
    if dropout is not None:
        if dropout != False:
            layers.append(nn.Dropout(p=0.1,inplace=True))
    if relu:
        layers += [nn.ReLU(inplace=True)]
    return layers

def make_layers(cfg, dilation=1, norm=None, dropout=None):
    modules = []
    in_channels = [cfg[0]]
    
    layers=[]
    layerCodes=[]
    for i,v in enumerate(cfg[1:]):
        if v == 'M':
            modules.append(nn.Sequential(*layers))
            layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            layerCodes = [v]
        elif type(v)==str and v[0:4] == 'long':
            modules.append(nn.Sequential(*layers))
            layers = [nn.MaxPool2d(kernel_size=(2,3), stride=(2,3))]
            layerCodes = [v]
        elif type(v)==str and v == 'ReLU':
            layers.append( nn.ReLU(inplace=True) )
            layerCodes.append(v)
        elif type(v)==str and v[:4]=='drop':
            if len(v)>6 and v[4:7]=='out':
                ind=8
            else:
                ind=4
            if len(v)>ind:
                amount = float(v[ind:])
            else:
                amount = 0.5
            layers.append(torch.nn.Dropout2d(p=amount, inplace=True))
            layerCodes.append(v)
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
        elif type(v)==str and v[0] == 'R':
            outCh=int(v[1:])
            layers.append(ResBlock(in_channels[-1],outCh,dilation,norm,dropout=dropout))
            layerCodes.append(v)
            in_channels.append(outCh)
        elif type(v)==str and v[0] == 'C': 
            div = v.find('-')
            if div==-1:
                kernel_size=5
                outCh=int(v[1:])
            else:
                kernel_size=int(v[1:div])
                outCh=int(v[1+div:])
            conv2d = nn.Conv2d(in_channels[-1], outCh, kernel_size=kernel_size, padding=(kernel_size-1)//2)
            #if i == len(cfg)-1:
            #    layers += [conv2d]
            #    break
            layers.append(conv2d)
            layerCodes.append(v)
            in_channels.append(outCh)
        elif type(v)==str and v[0] == 'D':
            outCh=int(v[1:]) #down sampling ResNet layer
            layers.append(ResBlock(in_channels[-1],outCh,dilation,norm,downsample=True,dropout=dropout))
            layerCodes.append(v)
            in_channels.append(outCh)
        elif type(v)==str and v[0] == 'U':
            outCh=int(v[1:]) #up sampling layer, linear
            layers.append(nn.ConvTranspose2d(in_channels[-1], outCh, kernel_size=2, stride=2, bias=False))
            layerCodes.append(v)
            in_channels.append(outCh)
        elif type(v)==str and v[0] == 'W': #dilated conv later
            outCh=int(v[1:])
            layers += convReLU(in_channels[-1],outCh,norm,dilation,dropout=dropout)
            layerCodes.append(v)
            in_channels.append(outCh)
        elif type(v)==str and v[0] == 'k': #conv later with custom kernel size
            div = v.find('-')
            kernel_size=int(v[1:div])
            outCh=int(v[div+1:])
            layers += convReLU(in_channels[-1],outCh,norm,kernel=kernel_size,dropout=dropout)
            layerCodes.append(v)
            in_channels.append(outCh)
        elif type(v)==str and v[:2] == 'cc': #CoordConv 'ccTYPE-k#,d#,hd-CHS'
            div = v.find('-')
            div2 = v.find('-',div+1)
            typ = v[2:div]
            kernel_size=[3,3]
            dilate=1
            if div2!=-1:
                for param in v[div+1:div2].split(','):
                    if param[0]=='k':
                        kernel_size=int(param[1:])
                        kernel_size=[kernel_size,kernel_size]
                    elif param[0]=='d':
                        dilate=int(param[1:])
                    elif param[:2]=='h':
                        kernel_size[0]=1
                    elif param[:2]=='v':
                        kernel_size[1]=1
                    else:
                        print("unknown subparameter {} in {}".format(param,v))
                        exit()
            else:
                div2=div
            outCh=int(v[div2+1:])
            layers += convReLU(in_channels[-1],outCh,norm,dilate,kernel=kernel_size,dropout=dropout,coordconv=typ)
            layerCodes.append(v)
            in_channels.append(outCh)
        elif type(v)==str and v[0] == 'd': #3x3 conv layer with custom dilation
            if v[:3] == 'dil':
                ind=4
            else:
                ind=1
            div = v.find('-')
            div0 = v.find(',')
            if div0==-1:
                dilate=int(v[ind:div])
            else:
                assert(div0<div)
                dilate=( int(v[ind:div0]), int(v[div0+1:div]) )
            outCh=int(v[div+1:])
            layers += convReLU(in_channels[-1],outCh,norm,dilate,dropout=dropout)
            layerCodes.append(v)
            in_channels.append(outCh)
        elif type(v)==str and v[:2] == 'hd': #horz 1x3 conv layer with custom dilation
            if v[:4] == 'hdil':
                ind=5
            else:
                ind=2
            div = v.find('-')
            dilate=int(v[ind:div])
            outCh=int(v[div+1:])
            layers += convReLU(in_channels[-1],outCh,norm,dilate,kernel=(1,3),dropout=dropout)
            layerCodes.append(v)
            in_channels.append(outCh)
        elif type(v)==str and v[:2] == 'vd': #vert 3x1 conv layer with custom dilation
            if v[:4] == 'vdil':
                ind=5
            else:
                ind=2
            div = v.find('-')
            dilate=int(v[ind:div])
            outCh=int(v[div+1:])
            layers += convReLU(in_channels[-1],outCh,norm,dilate,kernel=(3,1),dropout=dropout)
            layerCodes.append(outCh)
            in_channels.append(outCh)
        elif type(v)==str and v[0] == 'B': #ResNet layer with custom dilation
            div = v.find('-')
            dilate=int(v[1:div])
            outCh=int(v[div+1:])
            layers.append(ResBlock(in_channels[-1],outCh,dilate,norm,dropout=dropout))
            layerCodes.append(v)
            in_channels.append(outCh)
        elif type(v)==str and v[:4] == 'wave': #Res layer that is dilated in first conv and second conv is 1x1
            div = v.find('-')
            dilate=v[4:div]
            div0 = dilate.find(',')
            if div0==-1:
                dilate=int(dilate)
            else:
                assert(div0<div)
                dilate=( int(dilate[:div0]), int(dilate[div0+1:]) )
            outCh=int(v[div+1:])
            layers.append(ResBlock(in_channels[-1],outCh,dilate,norm,dropout=dropout,secondKernel=1))
            layerCodes.append(v)
            in_channels.append(outCh)
        elif type(v)==str and v[:2] == 'FC': #fully connected layer
            if v[2:4]=='nR':
                div= 4
                relu=False
                norm=None
                dropout=None
            else:
                div = 2
                relu=True
            outCh=int(v[div:])
            layers += fcReLU(in_channels[-1],outCh,norm,dropout=dropout,relu=relu)
            layerCodes.append(v)
            in_channels.append(outCh)
        #elif type(v)==str and len(v)>9 and v[:10] == 'global-avg':
        #    modules.append(nn.Sequential(*layers))
        #    layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
        #    layerCodes = [v]
        elif type(v)==str and v[:3] == 'sep': #depth-wise seperable conv
            outCh=int(v[3:])
            layers += convReLU(in_channels[-1],in_channels[-1],norm,kernel=(3,3),dropout=dropout,depthwise=True)
            layers += convReLU(in_channels[-1],outCh,norm,kernel=(1,1),dropout=dropout,depthwise=False)
            layerCodes.append(outCh)
            in_channels.append(outCh)
            
        elif type(v)==str:
            print('Error reading net cfg, unknown layer: '+v)
            exit(1)
        else:
            layers += convReLU(in_channels[-1],v,norm,dropout=dropout)
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
