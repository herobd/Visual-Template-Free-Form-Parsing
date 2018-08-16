import torch
from torch import nn
from base import BaseModel


def make_layers(cfg, batch_norm=False, weight_norm=False):
    modules = []
    in_channels = cfg[0]
    
    layers=[]
    for i,v in enumerate(cfg[1:]):
        if v == 'M':
            moduels.append(nn.Sequential(*layers))
            layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'U+':
            moduels.append(nn.Sequential(*layers))
            layers = [up(in_channels[-2])]
            in_channels.append(in_channels[-2]+in_channels[-1])
        else:
            conv2d = nn.Conv2d(in_channels[-1], v, kernel_size=3, padding=1)
            if i == len(cfg)-1:
                layers += [conv2d]
                break
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            elif weight_norm:
                layers += [weight_norm(conv2d), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels.append(v)
    if len(layers)>0:
        moduels.append(nn.Sequential(*layers))
    return modules #nn.Sequential(*layers)

class up(nn.Module):
    def __init__(self, in_ch, bilinear=True):
        super(up, self).__init__()
        self.outSize=in_ch
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            #self.up = nn.functional.interpolate
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, math.ceil(diffX / 2),
                        diffY // 2, math.ceil(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        return x


class Detector(BaseModel):
    def __init__(self, config): # predCount, base_0, base_1):
        super(Detector, self).__init__(config)
        self.predLineCount = config['number_of_line_types']
        self.predPointCount = config['number_of_point_types']
        self.predPixelCount = config['number_of_pixel_types']
        if "norm_type" in config:
            batch_norm=config["norm_type"]=='batch_norm'
            weight_norm=config["norm_type"]=='weight_norm'
        else:
            batch_norm=False
            weight_norm=False
        #self.cnn, self.scale = vgg.vgg11_custOut(self.predLineCount*5+self.predPointCount*3,batch_norm=batch_norm, weight_norm=weight_norm)
        self.numOutEnd = self.predLineCount*5+self.predPointCount*3
        if 'down_layers_cfg' in config:
            layers_cfg = config['down_layers_cfg']+[self.numOutEnd]
        else:
            layers_cfg=a[[3],64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, self.numOutEnd]
        if self.predPixelCount>0:
            if 'up_layers_cfg' in config:
                up_layers_cfg =  config['up_layers_cfg']+[self.predPixelCount]
            else:
                up_layers_cfg=[[512,512], 'U+', 256, 'U+', 128, 'U+', 64, 'U+', self.predPixelCount]

        self.net_down_modules = make_layers(layers_cfg, batch_norm, weight_norm)
        self.scale=1
        for a in layers_cfg:
            if a=='M':
                self.scale*=2

        self.net_up_modules = make_layers(up_layers_cfg, batch_norm, weight_norm)

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

        linePreds=[]
        for i in range(self.predLineCount):
            offset = i*5
            predictions = torch.cat([
                torch.sigmoid(y[:,0+offset:1+offset,:,:]),    #confidence
                y[:,1+offset:2+offset,:,:] + priors_1,        #x-center
                y[:,2+offset:3+offset,:,:] + priors_0,        #y-center
                y[:,3+offset:4+offset,:,:],                   #rotation (radians)
                y[:,4+offset:5+offset,:,:]                    #scale (half-height?)
            ], dim=1)
            
            predictions = predictions.transpose(1,3).contiguous()#from [batch, channel, rows, cols] to [batch, cols, rows, channels]
            predictions = predictions.view(predictions.size(0),-1,5)#flatten to [batch, instances, channel]
            linePreds.append(predictions)

        pointPreds=[]
        for i in range(self.predPointCount):
            offset = i*3
            predictions = torch.cat([
                torch.sigmoid(y[:,0+offset:1+offset,:,:]),    #confidence
                y[:,1+offset:2+offset,:,:] + priors_1,        #x
                y[:,2+offset:3+offset,:,:] + priors_0         #y
            ], dim=1)
            
            predictions = predictions.transpose(1,3).contiguous()#from [batch, channel, rows, cols] to [batch, cols, rows, channels]
            predictions = predictions.view(predictions.size(0),-1,5)#flatten to [batch, instances, channel]
            pointPreds.append(predictions)

        pixelPreds=None
        if self.predPixelCount>0:
            y2=y
            p=-2
            for module in self.net_up_modules[:-1]:
                y2 = module(y2,levels[p])
                p-=1
            pixelPreds = self.net_up_modules[-1](y2)
            



        return linePreds, pointPreds, pixelPreds

