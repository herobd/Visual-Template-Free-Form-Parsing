from base import BaseModel
import torch
from .pairing_box_net import PairingBoxNet #, YoloBoxDetector
from model import *
from model.yolo_loss import inv_tanh, get_closest_anchor_iou, get_closest_anchor_dist
import json
from datasets.forms_box_detect import FormsBoxDetect
import math


class PairingBoxFromGT(BaseModel):
    def __init__(self, config):
        super(PairingBoxFromGT, self).__init__(config)
        #if (not config['net_features']) if 'net_features' in config else True:
        if 'detector_checkpoint' not in config:
            with open(config['anchors_file']) as f:
                numAnchors=len(json.load(f))
            numBBParams=6 #conf,x-off,y-off,rot-off,h-scale,w-scale
            numBBTypes=config['number_of_box_types']
            #config['up_sample_ch'] = (numBBParams+numBBTypes)*numAnchors
            self.detect_scale=16
            self.pairer = PairingBoxNet(
                    config,
                    config,
                    numAnchors*(numBBParams+numBBTypes),
                    self.detect_scale)
            self.detector=None
        else:
            checkpoint = torch.load(config['detector_checkpoint'])
            detector_config = config['detector_config'] if 'detector_config' in config else checkpoint['config']['model']
            if 'state_dict' in checkpoint:
                self.detector = eval(checkpoint['config']['arch'])(detector_config)
                self.detector.load_state_dict(checkpoint['state_dict'])
            else:
                self.detector = checkpoint['model']
            self.detector.setForPairing()
            for param in self.detector.parameters():
                param.will_use_grad=param.requires_grad
                param.requires_grad=False
            self.detector_frozen=True

            self.pairer = PairingBoxNet(
                    config['pairer_config'],
                    detector_config,
                    self.detector.last_channels,
                    self.detector.scale)
            self.storedImageName=None
            self.detect_scale=self.detector.scale

        self.numBBTypes = self.pairer.numBBTypes
        self.numBBParams = self.pairer.numBBParams
        self.scale = self.pairer.scale
        self.anchors = self.pairer.anchors
        self.rotation = self.pairer.rotation

        if self.rotation:
            o_r = torch.FloatTensor([a['rot'] for a in self.anchors])
            o_h = torch.FloatTensor([a['height'] for a in self.anchors])
            o_w = torch.FloatTensor([a['width'] for a in self.anchors])
            cos_rot = torch.cos(o_r)
            sin_rot = torch.sin(o_r)
            p_left_x =  -cos_rot*o_w
            p_left_y =  sin_rot*o_w
            p_right_x = cos_rot*o_w
            p_right_y = -sin_rot*o_w
            p_top_x =   -sin_rot*o_h
            p_top_y =   -cos_rot*o_h
            p_bot_x =   sin_rot*o_h
            p_bot_y =   cos_rot*o_h
            anchor_points=torch.stack([p_left_x,p_left_y,p_right_x,p_right_y,p_top_x,p_top_y,p_bot_x,p_bot_y], dim=1)
            anchor_hws= (o_h+o_w)/2.0
            self.preparedAnchors=(anchor_points,anchor_hws)
            self.get_closest_anchor = lambda r,h,w: get_closest_anchor_dist(self.preparedAnchors,r,h,w)
        else:
            self.preparedAnchors=[[a['width'], a['height']] for a in self.anchors]
            self.get_closest_anchor = lambda r,h,w: get_closest_anchor_iou(self.preparedAnchors,h,w)
                

        self.dataset_train = FormsBoxDetect(dirPath=config['data_loader_detect']['data_dir'],split='train', config=config['data_loader_detect'])
        del config['data_loader_detect']['crop_params']
        self.dataset_valid = FormsBoxDetect(dirPath=config['data_loader_detect']['data_dir'],split='valid', config=config['data_loader_detect'])

        #create a look up
        self.dataLookUp={}
        for index,instance in enumerate(self.dataset_train):
            self.dataLookUp[instance['imgName']] = (self.dataset_train,index)
        for index,instance in enumerate(self.dataset_valid):
            self.dataLookUp[instance['imgName']] = (self.dataset_valid,index)

        self.no_final_features = config['no_final_features'] if 'no_final_features' in config else False
 
        

    def forward(self, image, queryMask,imageName,scale,cropPoint):
        padH=(self.detect_scale-(image.size(2)%self.detect_scale))%self.detect_scale
        padW=(self.detect_scale-(image.size(3)%self.detect_scale))%self.detect_scale
        if padH!=0 or padW!=0:
            padder = torch.nn.ZeroPad2d((0,padW,0,padH))
            image = padder(image)
            queryMask = padder(queryMask)
        offsets=[]
        for b in range(len(imageName)):
            d= self.dataLookUp[imageName[b]]
            instance = d[0].getitem(d[1],scale[b],cropPoint[b])
            offset = self.create_offsets(instance,image.size(2),image.size(3))
            offsets.append(offset)
        offsets = torch.cat(offsets,dim=0)
        offsets = offsets.to(image.device)
        if self.detector is None:
            final_features = offsets
        else:
            if not self.training and self.storedImageName is not None and imageName==self.storedImageName:
                offsetPredictionsD=self.storedOffsetPredictionsD
                final_features=self.storedFinal_features
            else:
                save=not self.training
                self.storedOffsetPredictionsD=None
                self.storedFinal_features=None
                self.storedImageName=None
                offsetPredictionsD = self.detector(image)
                final_features=self.detector.final_features
                self.detector.final_features=None

                if save:
                    self.storedOffsetPredictionsD=offsetPredictionsD
                    self.storedFinal_features=final_features
                    self.storedImageName=imageName
                    #print('size {}'.format(image.size()))
            if final_features is None:
                import pdb;pdb.set_trace()
        if self.no_final_features:
            final_features[...]=0
        bbPredictions, offsetPredictions = self.pairer( image,
                                                        queryMask,
                                                        final_features, 
                                                        offsets)

        return bbPredictions, offsetPredictions


    def create_offsets(self,gt,imgH,imgW):
        #create what the detector network would have produced for the gt
        #imgH = gt['img'].size(2) # we take the padded version
        #imgW = gt['img'].size(3)
        H=int(imgH/self.detect_scale)
        W=int(imgW/self.detect_scale)
        offset = torch.empty((1,(self.numBBParams+self.numBBTypes)*len(self.anchors),H,W)).normal_(mean=0,std=0.1)
        #set all conf to zero
        for a in range(len(self.anchors)):
            offset[:,a*(self.numBBParams+self.numBBTypes),:,:]=0

        for i in range(gt['bb_gt'].size(1)):
            bb=gt['bb_gt'][0,i]
            a,_ = self.get_closest_anchor(bb[2],bb[3],bb[4])
            index=(self.numBBParams+self.numBBTypes)*a
            cellX = int(bb[0].item()//self.scale)
            cellY = int(bb[1].item()//self.scale)
            priorX = cellX*self.scale + 0.5
            priorY = cellY*self.scale + 0.5
            offset[0,index+0,cellY,cellX]=1 #conf to one
            offset[0,index+1,cellY,cellX]= inv_tanh((bb[0]-priorX)/self.scale) #x
            offset[0,index+2,cellY,cellX]= inv_tanh((bb[1]-priorY)/self.scale) #y
            offset[0,index+3,cellY,cellX]= inv_tanh( (bb[2]-self.anchors[a]['rot'])/(math.pi/2) )
            offset[0,index+4,cellY,cellX]= torch.log(bb[3]/self.anchors[a]['height'])
            offset[0,index+5,cellY,cellX]= torch.log(bb[4]/self.anchors[a]['width'])
            offset[0,index+6:(index+6+self.numBBTypes),cellY,cellX]= bb[13:13+self.numBBTypes] #class
        return offset
