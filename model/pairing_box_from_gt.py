from base import BaseModel
import torch
from .pairing_box_net import PairingBoxNet #, YoloBoxDetector
from model import *


class PairingBoxFromGT(BaseModel):
    def __init__(self, config):
        super(PairingBoxFromGT, self).__init__(config)
        with open(detector_config['anchors_file']) as f:
            numAnchors=len(json.load(f))
        numBBParams=6 #conf,x-off,y-off,rot-off,h-scale,w-scale
        numBBTypes=config['number_of_box_types']

        config['up_sample_ch'] = (numBBParams+numBBTypes)*numAnchors

        self.pairer = PairingBoxNet(
                config,
                config,
                numAnchors*(numBBParams+numBBTypes),
                config['detector_scale'])

        self.numBBTypes = self.pairer.numBBTypes
        self.numBBParams = self.pairer.numBBParams
        self.scale = self.pairer.scale
        self.anchors = self.pairer.anchors
        self.rotation = self.pairer.rotation

        self.preparedAnchors=[]
        for anchor in self.anchors:
            dgsdfg

        self.dataset_train = FormsBoxDetect(dirPath=config['data_loader_detect']['data_dir'],split='train', config=config['data_loader_detect'])
        self.dataset_valid = FormsBoxDetect(dirPath=config['data_loader_detect']['data_dir'],split='valid', config=config['data_loader_detect'])

        #create a look up
        dataLookUp={}
        for index,instance in enumerate(dataset_train):
            dataLookUp[instance['imgName']] = (self.dataset_train,index)
        for index,instance in enumerate(dataset_valid):
            dataLookUp[instance['imgName']] = (self.dataset_valid,index)
 
        

    def forward(self, image, queryMask,imageName,scale,cropX=None,cropY=None):
        for b in len(imagename):
            d= dataLookup[imageName[b]]
            instance = d[0].getitem(d[1],scale,cropX,cropY)
            offset = self.create_offsets(instance)
            offsets.append(offset)
        offsets = torch.cat(offsets,dim=0)
        bbPredictions, offsetPredictions = self.pairer( image,
                                                        queryMask,
                                                        offsets, 
                                                        offsets)

        return bbPredictions, offsetPredictions


    def create_offsets(self,gt):
        #create what the detector network would have produced for the gt
        H=imgH/self.scale
        W=imgW/self.scale
        offset = torch.normal((1,(self.numBBParams+self.numBBTypes)*len(self.anchors),H,W),mu=0,sigma=0.1)
        #set all conf to zero
        for a in range(len(self.anchors)):
            offset[:,a*(self.numBBParams+self.numBBTypes),:,:]=0

        priors_0 = torch.arange(0,y.size(2)).type_as(img.data)[None,:,None]
        priors_0 = (priors_0 + 0.5) * self.scale #self.base_0
        priors_0 = priors_0.expand(y.size(0), priors_0.size(1), y.size(3))
        priors_0 = priors_0[:,None,:,:]#.to(img.device)

        priors_1 = torch.arange(0,y.size(3)).type_as(img.data)[None,None,:]
        priors_1 = (priors_1 + 0.5) * self.scale #elf.base_1
        priors_1 = priors_1.expand(y.size(0), y.size(2), priors_1.size(2))
        priors_1 = priors_1[:,None,:,:]#.to(img.device)

        for bb in bb_gt:
            if self.rotation
                a,_ = get_closest_anchor_iou(self.prepredAnchors,bb[3],bb[4])
            else:
                a,_ = get_closest_anchor_dist(self.prepredAnchors,bb[2],bb[3],bb[4])
            cellX = bb[0]//self.scale
            cellY = bb[1]//self.scale
            offset[0,0,cellY,cellX]=1 #conf to one
            offset[0,1,cellY,cellX]= inv_tanh((bb[0]-priors_1)/self.scale) #x
            offset[0,2,cellY,cellX]= inv_tanh((bb[1]-priors_0)/self.scale) #y
            offset[0,3,cellY,cellX]= inv_tanh( (bb[2]-self.anchor[a]['rot'])/(math.pi/2) )
            offset[0,4,cellY,cellX]= torch.log(bb[3]/anchor[a]['height'])
            offset[0,5,cellY,cellX]= torch.log(bb[4]/anchor[a]['width'])
            offset[0,6:,cellY,cellX]= bb[5:] #class
        return offset
