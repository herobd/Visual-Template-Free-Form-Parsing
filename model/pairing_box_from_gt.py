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

        self.dataset_train = FormsBoxDetect(dirPath=config['data_loader_detect']['data_dir'],split='train', config=config['data_loader_detect'])
        self.dataset_valid = FormsBoxDetect(dirPath=config['data_loader_detect']['data_dir'],split='valid', config=config['data_loader_detect'])

        #create a look up
        dataLookUp={}
        for index,instance in enumerate(dataset_train):
            dataLookUp[instance['imgName']] = (self.dataset_train,index)
        for index,instance in enumerate(dataset_valid):
            dataLookUp[instance['imgName']] = (self.dataset_valid,index)
 
        

    def forward(self, image, queryMask,imageName,scale,cropX=None,cropY=None):
        d= dataLookup[imageName]
        instance = d[0].getitem(d[1],scale,cropX,cropY)
        offsets = self.create_offsets(instance)
        bbPredictions, offsetPredictions = self.pairer( image,
                                                        queryMask,
                                                        offsets, 
                                                        offsets)

        return bbPredictions, offsetPredictions
