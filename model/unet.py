# from https://github.com/milesial/Pytorch-UNet
from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.weight_norm import weight_norm

class UNet(BaseModel):
    def __init__(self, config):
        super(UNet, self).__init__(config)
        if config is None or 'skip_last_sigmoid' not in config:
            skip_last_sigmoid=False
        else:
            skip_last_sigmoid=config['skip_last_sigmoid']
        if config is None or 'norm_type' not in config:
            norm_type=None
        else:
            norm_type=config['norm_type']

        self.inc = inconv(config['n_channels'], 64, norm_type)
        self.down1 = down(64, 128, norm_type)
        self.down2 = down(128, 256, norm_type)
        self.down3 = down(256, 512, norm_type)
        self.down4 = down(512, 512, norm_type)
        self.up1 = up(1024, 256, norm_type)
        self.up2 = up(512, 128, norm_type)
        self.up3 = up(256, 64, norm_type)
        self.up4 = up(128, 64, norm_type)
        self.outc = outconv(64, 1, skip_last_sigmoid)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x.view(x.shape[0],x.shape[-2],x.shape[-1])



class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, normType):
        super(double_conv, self).__init__()
        if normType is None:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        elif normType=='batchNorm':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        elif normType=='weightNorm':
            self.conv = nn.Sequential(
                weight_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
                nn.ReLU(inplace=True),
                weight_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, normType):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, normType)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, normType):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, normType)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, normType, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, normType)

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
