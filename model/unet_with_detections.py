# from https://github.com/milesial/Pytorch-UNet
from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.weight_norm import weight_norm
from .yolo_box_detector import make_layers

class UNetWithDetections(BaseModel):
    def __init__(self, config):
        super(UNetWithDetections, self).__init__(config)

        #def build(self,detect_ch, detect_scale,only_features=False):
        detect_ch = config['detect_ch']
        detect_scale = config['detect_scale']
        self.only_features=False#only_features
        if self.config is None or 'skip_last_sigmoid' not in self.config:
            skip_last_sigmoid=True
        else:
            skip_last_sigmoid=self.config['skip_last_sigmoid']
        if self.config is None or 'norm_type' not in self.config:
            norm_type=None
            print('No norm set for UNetWithDetections')
        else:
            norm_type=self.config['norm_type']

        detect_ch_after_up = self.config['up_sample_ch'] if 'up_sample_ch' in self.config else 128
        if 'up_sample_relu' not in self.config or self.config['up_sample_relu']:
            self.up_sample = nn.Sequential(
                    nn.ConvTranspose2d(detect_ch,detect_ch_after_up,kernel_size=detect_scale//2,stride=detect_scale//2),
                    nn.ReLU(inplace=True)
                    )
        else:   
            self.up_sample = nn.ConvTranspose2d(detect_ch,detect_ch_after_up,kernel_size=detect_scale//2,stride=detect_scale//2)

        inc_cfg = self.config['inc_cfg'] if 'inc_cfg' in self.config else [64]
        down1_cfg = self.config['down1_cfg'] if 'down1_cfg' in self.config else [128,128]
        down2_cfg = self.config['down2_cfg'] if 'down2_cfg' in self.config else [256,256]
        down3_cfg = self.config['down3_cfg'] if 'down3_cfg' in self.config else [512,512]
        down4_cfg = self.config['down4_cfg'] if 'down4_cfg' in self.config else [512,512]

        self.inc, inc_ch = make_layers([self.config['n_channels']]+inc_cfg,  norm=norm_type)
        self.inc=nn.Sequential(*self.inc)
        self.down1 = down(inc_ch+detect_ch_after_up, down1_cfg,  norm_type)
        self.down2 = down(self.down1.outSize, down2_cfg,  norm_type)
        self.down3 = down(self.down2.outSize, down3_cfg,  norm_type)
        self.down4 = down(self.down3.outSize, down4_cfg,  norm_type)
        if not self.only_features:
            up1_cfg = self.config['up1_cfg'] if 'up1_cfg' in self.config else [256,256]
            up2_cfg = self.config['up2_cfg'] if 'up2_cfg' in self.config else [128,128]
            up3_cfg = self.config['up3_cfg'] if 'up3_cfg' in self.config else [64,64]

            self.up1 = up(self.down4.outSize+self.down3.outSize, up1_cfg,  norm_type)
            self.up2 = up(self.up1.outSize+self.down2.outSize, up2_cfg,  norm_type)
            self.up3 = up(self.up2.outSize+self.down1.outSize, up3_cfg,  norm_type)
            #self.up3 = up(self.up2.outSize+self.inc.outSize, 64,  norm_type, dilation_positions)
            self.outc = outconv(self.up3.outSize, 1, skip_last_sigmoid)

    #def unfreeze(self): 
    #    for param in self.detector.parameters(): 
    #        param.requires_grad=param.will_use_grad  
    #    self.detector_frozen=False


    def forward(self, x, detection_features):
        #padH=(self.detector.scale-(image.size(2)%self.detector.scale))%self.detector.scale
        #padW=(self.detector.scale-(image.size(3)%self.detector.scale))%self.detector.scale
        #if padH!=0 or padW!=0:
        #    padder = torch.nn.ZeroPad2d((0,padW,0,padH))
        #    image = padder(image)
        #    queryMask = padder(queryMask)
        #if not self.training and self.storedImageName is not None and imageName==self.storedImageName:
        #    offsetPredictionsD=self.storedOffsetPredictionsD
        #    final_features=self.storedFinal_features
        #else:
        #    save=not self.training
        #    self.storedOffsetPredictionsD=None
        #    self.storedFinal_features=None
        #    if self.detector_frozen:
        #        #self.detector.eval()
        #        #with torch.no_grad():
        #        #batch size set to one to accomidate
        #        offsetPredictionsD = self.detector(image)
        #    else:
        #        offsetPredictionsD = self.detector(image)
        #    final_features=self.detector.final_features

        #    if save:
        #        self.storedOffsetPredictionsD=offsetPredictionsD
        #        self.storedFinal_features=final_features
        #        self.storedImageName=imageName
        up_features = self.up_sample(detection_features)
        x = self.inc(x)
        #x1 = torch.cat([x1,up_features],dim=1)
        x2 = self.down1(x,up_features)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.only_features:
            return x5
        x = self.up1(x5, x4)
        x5=None
        x4=None
        torch.cuda.empty_cache()
        x = self.up2(x, x3)
        x3=None
        torch.cuda.empty_cache()
        x = self.up3(x, x2)
        #x = self.up3(x, x1)
        x = self.outc(x)
        return x.view(x.shape[0],x.shape[-2],x.shape[-1])




class down(nn.Module):
    def __init__(self, in_ch, cfg, normType):
        super(down, self).__init__()
        self.conv, self.outSize = make_layers([in_ch]+cfg,norm=normType)
        self.conv=nn.Sequential(*self.conv)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x,append=None):
        x = self.pool(x)
        if append is not None:
            #padH = x.size(-2)-append.size(-2)
            #padW = x.size(-1)-append.size(-1)
            #if padH!=0 or padW!=0:
            #    padder = torch.nn.ZeroPad2d((0,padW,0,padH))
            #    append = padder(append)
            x = torch.cat([x,append],dim=1)
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, cfg,  normType, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            #self.up = nn.functional.interpolate
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv, self.outSize = make_layers([in_ch]+cfg,norm=normType)
        self.conv=nn.Sequential(*self.conv)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        #print(x1.shape)
        #print(x2.shape)
        #print((diffX,diffY))
        x1 = F.pad(x1, (diffX // 2, math.ceil(diffX / 2),
                        diffY // 2, math.ceil(diffY / 2)))
        #x1 =nn.ReplicationPad2d((diffX // 2, math.ceil(diffX / 2),
        #                          diffY // 2, math.ceil(diffY / 2)))(x1)
        #print(x1.shape)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, skip_last_sigmoid):
        super(outconv, self).__init__()
        self.outSize=out_ch
        if skip_last_sigmoid:
            self.conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1),
                    nn.Sigmoid()
                )

    def forward(self, x):
        x = self.conv(x)
        return x
