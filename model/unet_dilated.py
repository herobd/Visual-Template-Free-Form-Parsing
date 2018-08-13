# from https://github.com/milesial/Pytorch-UNet
from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.weight_norm import weight_norm

class UNetDilated(BaseModel):
    def __init__(self, config):
        super(UNetDilated, self).__init__(config)
        if config is None or 'skip_last_sigmoid' not in config:
            skip_last_sigmoid=False
        else:
            skip_last_sigmoid=config['skip_last_sigmoid']
        if config is None or 'norm_type' not in config:
            norm_type=None
        else:
            norm_type=config['norm_type']
        
        if config is None or 'dilation_positions' not in config:
            dilation_positions=0 #no dilation
        else:
            dilation_positions=config['dilation_positions']


        self.inc = inconv(config['n_channels'], 64,  norm_type, dilation_positions)
        self.down1 = down(self.inc.outSize, 128,  norm_type, dilation_positions)
        self.down2 = down(self.down1.outSize, 256,  norm_type, dilation_positions)
        self.down3 = down(self.down2.outSize, 512,  norm_type, dilation_positions)
        #self.down4 = down(512, 512,  norm_type, dilation_positions)
        #self.up1 = up(1024, 256,  norm_type, dilation_positions)
        self.up1 = up(self.down3.outSize+self.down2.outSize, 256,  norm_type, dilation_positions)
        self.up2 = up(self.up1.outSize+self.down1.outSize, 128,  norm_type, dilation_positions)
        self.up3 = up(self.up2.outSize+self.inc.outSize, 64,  norm_type, dilation_positions)
        self.outc = outconv(self.up3.outSize, 1, skip_last_sigmoid)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x.view(x.shape[0],x.shape[-2],x.shape[-1])



class triple_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, normType, dilationPositions):
        super(triple_conv, self).__init__()
        self.outSize=out_ch
        dilation1 = 3 if 1 in dilationPositions else 1
        dilation2 = 3 if 2 in dilationPositions else 1
        dilation3 = 3 if 3 in dilationPositions else 1
        #if in_ch//2 == out_ch or out_ch//2 == in_ch:
        mid_ch=out_ch
        #else:
        #    mid_ch = min(in_ch,out_ch)*2

        if normType is None or normType=='none':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 3, padding=dilation1, dilation=dilation1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, mid_ch, 3, padding=dilation2, dilation=dilation2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, out_ch, 3, padding=dilation3, dilation=dilation3),
                nn.ReLU(inplace=True)
            )
        elif normType=='batchNorm':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 3, padding=dilation1, dilation=dilation1),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, mid_ch, 3, padding=dilation2, dilation=dilation2),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, out_ch, 3, padding=dilation3, dilation=dilation3),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        elif normType=='weightNorm':
            self.conv = nn.Sequential(
                weight_norm(nn.Conv2d(in_ch, mid_ch, 3, padding=dilation1, dilation=dilation1)),
                nn.ReLU(inplace=True),
                weight_norm(nn.Conv2d(mid_ch, mid_ch, 3, padding=dilation2, dilation=dilation2)),
                nn.ReLU(inplace=True),
                weight_norm(nn.Conv2d(mid_ch, out_ch, 3, padding=dilation3, dilation=dilation3)),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch,  normType, dilationPositions):
        super(inconv, self).__init__()
        self.outSize=out_ch
        self.conv = triple_conv(in_ch, out_ch,  normType, dilationPositions)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch,  normType, dilationPositions):
        super(down, self).__init__()
        self.outSize=out_ch
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            triple_conv(in_ch, out_ch,  normType, dilationPositions)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch,  normType, dilationPositions, bilinear=True):
        super(up, self).__init__()
        self.outSize=out_ch

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            #self.up = nn.functional.interpolate
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = triple_conv(in_ch, out_ch,  normType, dilationPositions)

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
