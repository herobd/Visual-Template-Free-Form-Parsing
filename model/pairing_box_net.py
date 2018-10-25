import torch
from torch import nn
from base import BaseModel
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math
import json
from model.yolo_box_detector import make_layers


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
    def __init__(self, config): # predCount, base_0, base_1):
        super(PairingBoxNet, self).__init__()
        self.rotation = config['rotation'] if 'rotation' in config else True
        self.numBBTypes = config['number_of_box_types']
        self.numBBParams = 6 #conf,x-off,y-off,rot-off,h-scale,w-scale
        with open(config['anchors_file']) as f:
            self.anchors = json.loads(f.read()) #array of objects {rot,height,width}
        self.numAnchors = len(self.anchors)
        #self.predPointCount = config['number_of_point_types']
        #self.predPixelCount = config['number_of_pixel_types']
        self.numOutBB = (self.numBBTypes+self.numBBParams)*self.numAnchors
        #self.numOutPoint = self.predPointCount*3
        im_ch = 1+( 3 if 'color' not in config or config['color'] else 1 ) #+1 for query mask
        norm = config['norm_type'] if "norm_type" in config else None
        if norm is None:
            print('Warning: PairingBoxNet has no normalization!')
        dilation = config['dilation'] if 'dilation' in config else 1


        if 'up_sample_relu' not in config or config['up_sample_relu']:
            self.up_sample = nn.Sequential(
                    nn.ConvTranspose2d(detect_ch,detect_ch,kernel_size=detect_scale,stride=detect_scale),
                    nn.ReLU(inplace=True)
                    )
        else:
            self.up_sample = nn.ConvTranspose2d(detect_ch,detect_ch,kernel_size=detect_scale,stride=detect_scale)

        if 'down_layers_cfg' in config:
            layers_cfg_down = config['down_layers_cfg']
        else:
            layers_cfg_down=[im_ch,64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 1024, 1024]

        if layers_cfg_down[0]>4:
            layers_cfg_down = [im_ch]+layers_cfg_down

        down_modules, down_last_ch = make_layers(layers_cfg_down, dilation,norm)
        self.net_down = nn.Sequential(*down_modules)
        self.scale=1
        for a in layers_cfg:
            if a=='M' or (type(a) is str and a[0]=='D'):
                self.scale*=2
            elif type(a) is str and a[0]=='U':
                self.scale/=2
        assert(self.scale == detect_scale)
        
        if 'final_layers_cfg' in config:
            layers_cfg_final = config['final_layers_cfg']
        else:
            layers_cfg_final=['R1024']
        layers_cfg_final = [down_last_ch+self.numOutBB]+layers_cfg_final
        final_modules, final_last_ch = make_layers(layers_cfg_final, dilation,norm)
        final_modules.append( nn.Conv2d(final_last_ch, self.numOutBB, kernel_size=1) )
        self.final = nn.nn.Sequential(*final_modules)

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
        input = np.cat([image,mask,up_features],dim=1)
        at_box_res = self.net_down(input)
        with_detections = np.cat([at_box_res,detected_boxes],dim=1)
        pred = self.final(with_detections)
        
        pred+=detected_boxes
        #pred[:,:,:,:,1:]+=detected_boxes[:,:,:,:,1:]
        #pred[:,:,:,:,0:1]*=detected_boxes[:,:,:,:,0:1]



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
        pred_boxes=[]
        pred_offsets=[]
        for i in range(self.numAnchors):

            offset = i*(self.numBBParams+self.numBBTypes)
            if self.rotation:
                rot_dif = (math.pi/2)*torch.tanh(pred[:,3+offset:4+offset,:,:])
            else:
                rot_dif = torch.zeros_like(y[:,3+offset:4+offset,:,:])

            stackedPred = [
                torch.sigmoid(pred[:,0+offset:1+offset,:,:]),                #0. confidence
                torch.tanh(pred[:,1+offset:2+offset,:,:])*self.scale + priors_1,        #1. x-center
                torch.tanh(pred[:,2+offset:3+offset,:,:])*self.scale + priors_0,        #2. y-center
                rot_dif + anchor[i]['rot'],      #3. rotation (radians)
                torch.exp(pred[:,4+offset:5+offset,:,:]) * anchor[i]['height'], #4. height (half), I don't think this needs scaled
                torch.exp(pred[:,5+offset:6+offset,:,:]) * anchor[i]['width'],  #5. width (half)  
            ]


            for j in range(self.numBBTypes):
                stackedPred.append(y[:,6+j+offset:7+j+offset,:,:])         #x. class prediction
                #stackedOffsets.append(y[:,6+j+offset:7+j+offset,:,:])         #x. class prediction
            pred_boxes.append(torch.cat(stackedPred, dim=1))
            pred_offsets.append(y[:,offset:offset+self.numBBParams+self.numBBTypes,:,:])

        bbPredictions = torch.stack(pred_boxes, dim=1)
        offsetPredictions = torch.stack(pred_offsets, dim=1)
        
        bbPredictions = bbPredictions.transpose(2,4).contiguous()#from [batch, anchors, channel, rows, cols] to [batch, anchros, cols, rows, channels]
        bbPredictions = bbPredictions.view(bbPredictions.size(0),bbPredictions.size(1),-1,bbPredictions.size(4))#flatten to [batch, anchors, instances, channel]
        #avg_conf_per_anchor = bbPredictions[:,:,:,0].mean(dim=0).mean(dim=1)
        bbPredictions = bbPredictions.view(bbPredictions.size(0),-1,bbPredictions.size(3)) #[batch, instances+anchors, channel]

        offsetPredictions = offsetPredictions.permute(0,1,3,4,2).contiguous() #to [batch, anchor, row, col, channels]

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
            



        return bbPredictions, offsetPredictions, pointPreds, pixelPreds #, avg_conf_per_anchor

