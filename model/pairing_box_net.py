import torch
from torch import nn
from base import BaseModel
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math
import json
from .net_builder import make_layers

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



class PairingBoxNet(nn.Module):
    def __init__(self, config,detector_config,detect_ch,detect_scale): # predCount, base_0, base_1):
        super(PairingBoxNet, self).__init__()
        self.multConf = config['mult_conf'] if 'mult_conf' in config else False
        self.rotation = detector_config['rotation'] if 'rotation' in detector_config else True
        self.numBBTypes = detector_config['number_of_box_types']
        self.numBBParams = 6 #conf,x-off,y-off,rot-off,h-scale,w-scale
        with open(detector_config['anchors_file']) as f:
            self.anchors = json.loads(f.read()) #array of objects {rot,height,width}
        self.numAnchors = len(self.anchors)
        #self.predPointCount = config['number_of_point_types']
        #self.predPixelCount = config['number_of_pixel_types']
        self.numOutBB = (self.numBBTypes+self.numBBParams)*self.numAnchors
        #self.numOutPoint = self.predPointCount*3
        maskSize = 1+(config['num_dist_masks'] if 'num_dist_masks' in config else 0) #+1 for query mask
        im_ch = maskSize+( 3 if 'color' not in detector_config or detector_config['color'] else 1 )
        norm = config['norm_type'] if "norm_type" in config else None
        if norm is None:
            print('Warning: PairingBoxNet has no normalization!')
        dilation = config['dilation'] if 'dilation' in config else 1

        if 'down1_layers_cfg' in config:
            layers_cfg_down1 = config['down1_layers_cfg']
        else:
            layers_cfg_down1=[32, 'M']

        #if type(layers_cfg_down1[0])==str or layers_cfg_down1[0]>4:
        layers_cfg_down1 = [im_ch]+layers_cfg_down1
        down1_modules, down1_last_ch = make_layers(layers_cfg_down1, dilation,norm)
        self.net_down1 = nn.Sequential(*down1_modules)
        down1scale=1
        down1scaleX=1
        down1scaleY=1
        for a in layers_cfg_down1:
            if a=='M' or (type(a) is str and a[0]=='D'):
                down1scaleX*=2
                down1scaleY*=2
            elif type(a) is str and a[0]=='U':
                down1scaleX/=2
                down1scaleY/=2
            elif type(a) is str and a[0:4]=='long': #long pool
                down1scaleX*=3
                down1scaleY*=2
        down1scale=[down1scaleX,down1scaleY]

        detect_ch_after_up = config['up_sample_ch'] if 'up_sample_ch' in config else 256

        up_sample_size = (detect_scale[0]//down1scale[0], detect_scale[1]//down1scale[1])

        #if 'up_sample_relu' not in config or config['up_sample_relu']:
        self.up_sample = nn.Sequential(
                    nn.ConvTranspose2d(detect_ch,detect_ch_after_up,kernel_size=up_sample_size,stride=up_sample_size),
                    nn.ReLU(inplace=True)
                    )
        #else:
        #    self.up_sample = nn.ConvTranspose2d(detect_ch,detect_ch_after_up,kernel_size=detect_scale,stride=detect_scale)

        if 'down2_layers_cfg' in config:
            layers_cfg_down2 = config['down2_layers_cfg']
        else:
            layers_cfg_down2=[128,'M',256, "M", 512, "M", 1024]

        layers_cfg_down2 = [down1_last_ch+detect_ch_after_up]+layers_cfg_down2
        down2_modules, down2_last_ch = make_layers(layers_cfg_down2, dilation,norm)
        self.net_down2 = nn.Sequential(*down2_modules)
        self.scale=down1scale
        for a in layers_cfg_down2:
            if a=='M' or (type(a) is str and a[0]=='D'):
                self.scale[0]*=2
                self.scale[1]*=2
            elif type(a) is str and a[0]=='U':
                self.scale[0]/=2
                self.scale[1]/=2
            elif type(a) is str and a[0:4]=='long': #long pool
                self.scale[0]*=3
                self.scale[1]*=2
        assert(self.scale[0] == detect_scale[0] and self.scale[1] == detect_scale[1])
        
        if 'final_layers_cfg' in config:
            layers_cfg_final = config['final_layers_cfg']
        else:
            layers_cfg_final=[1024]
        layers_cfg_final = [down2_last_ch+self.numOutBB]+layers_cfg_final
        final_modules, final_last_ch = make_layers(layers_cfg_final, dilation,norm)
        final_modules.append( nn.Conv2d(final_last_ch, self.numOutBB, kernel_size=1) )
        self.final = nn.Sequential(*final_modules)

        #if self.predPixelCount>0:
        #    if 'up_layers_cfg' in config:
        #        up_layers_cfg =  config['up_layers_cfg']
        #    else:
        #        up_layers_cfg=[512, 'U+512', 256, 'U+256', 128, 'U+128', 64, 'U+64']
        #    self.net_up_modules, up_last_channels = make_layers(up_layers_cfg, 1, norm)
        #    self.net_up_modules.append(nn.Conv2d(up_last_channels, self.predPixelCount, kernel_size=1))
        #    self._hack_up = nn.Sequential(*self.net_up_modules)


    def forward(self, img, mask, detector_features, detected_boxes):
        #import pdb; pdb.set_trace()
        #y = self.cnn(img)
        up_features = self.up_sample(detector_features)
        #input = torch.cat([img,mask,up_features],dim=1)
        #at_box_res = self.net_down(input)
        input = torch.cat([img,mask],dim=1)
        down1 = self.net_down1(input)
        input = torch.cat([down1,up_features],dim=1)
        at_box_res = self.net_down2(input)
        #for i in range(self.numAnchors):
        #    offset = i*(self.numBBParams+self.numBBTypes)
        #    detected_boxes[:,offset,:,:]=0
        with_detections = torch.cat([at_box_res,detected_boxes],dim=1)
        pred = self.final(with_detections)
 
        #This is done in anchor loop to exclude conf
        #pred+=detected_boxes



        #priors_0 = Variable(torch.arange(0,y.size(2)).type_as(img.data), requires_grad=False)[None,:,None]
        priors_0 = torch.arange(0,pred.size(2)).type_as(img.data)[None,:,None].to(img.device)
        priors_0 = (priors_0 + 0.5) * self.scale[1] #self.base_0
        priors_0 = priors_0.expand(pred.size(0), priors_0.size(1), pred.size(3))
        priors_0 = priors_0[:,None,:,:]

        #priors_1 = Variable(torch.arange(0,y.size(3)).type_as(img.data), requires_grad=False)[None,None,:]
        priors_1 = torch.arange(0,pred.size(3)).type_as(img.data)[None,None,:].to(img.device)
        priors_1 = (priors_1 + 0.5) * self.scale[0] #elf.base_1
        priors_1 = priors_1.expand(pred.size(0), pred.size(2), priors_1.size(2))
        priors_1 = priors_1[:,None,:,:]

        anchor = self.anchors
        pred_boxes=[]
        pred_offsets=[]
        for i in range(self.numAnchors):

            offset = i*(self.numBBParams+self.numBBTypes)
            pred[:,1+offset:self.numBBParams+self.numBBTypes+offset,:,:] += detected_boxes[:,1+offset:self.numBBParams+self.numBBTypes+offset,:,:]
            if self.multConf:
                pred[:,offset,:,:] *= detected_boxes[:,offset,:,:]
            if self.rotation:
                rot_dif = (math.pi/2)*torch.tanh(pred[:,3+offset:4+offset,:,:])
            else:
                rot_dif = torch.zeros_like(pred[:,3+offset:4+offset,:,:])

            stackedPred = [
                torch.sigmoid(pred[:,0+offset:1+offset,:,:]),                #0. confidence
                torch.tanh(pred[:,1+offset:2+offset,:,:])*self.scale[0] + priors_1,        #1. x-center
                torch.tanh(pred[:,2+offset:3+offset,:,:])*self.scale[1] + priors_0,        #2. y-center
                rot_dif + anchor[i]['rot'],      #3. rotation (radians)
                torch.exp(pred[:,4+offset:5+offset,:,:]) * anchor[i]['height'], #4. height (half), I don't think this needs scaled
                torch.exp(pred[:,5+offset:6+offset,:,:]) * anchor[i]['width'],  #5. width (half)  
            ]


            for j in range(self.numBBTypes):
                stackedPred.append(pred[:,6+j+offset:7+j+offset,:,:])         #x. class prediction
                #stackedOffsets.append(y[:,6+j+offset:7+j+offset,:,:])         #x. class prediction
            pred_boxes.append(torch.cat(stackedPred, dim=1))
            pred_offsets.append(pred[:,offset:offset+self.numBBParams+self.numBBTypes,:,:])

        bbPredictions = torch.stack(pred_boxes, dim=1)
        offsetPredictions = torch.stack(pred_offsets, dim=1)
        
        bbPredictions = bbPredictions.transpose(2,4).contiguous()#from [batch, anchors, channel, rows, cols] to [batch, anchros, cols, rows, channels]
        bbPredictions = bbPredictions.view(bbPredictions.size(0),bbPredictions.size(1),-1,bbPredictions.size(4))#flatten to [batch, anchors, instances, channel]
        #avg_conf_per_anchor = bbPredictions[:,:,:,0].mean(dim=0).mean(dim=1)
        bbPredictions = bbPredictions.view(bbPredictions.size(0),-1,bbPredictions.size(3)) #[batch, instances+anchors, channel]

        offsetPredictions = offsetPredictions.permute(0,1,3,4,2).contiguous() #to [batch, anchor, row, col, channels]

            



        return bbPredictions, offsetPredictions #, avg_conf_per_anchor

