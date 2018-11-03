from base import BaseModel
import torch
from .pairing_box_net import PairingBoxNet #, YoloBoxDetector
from model import *


class PairingBoxFull(BaseModel):
    def __init__(self, config):
        super(PairingBoxFull, self).__init__(config)

        checkpoint = torch.load(config['detector_checkpoint'])
        detector_config = config['detector_config'] if 'detector_config' in config else checkpoint['config']['model']
        if 'state_dict' in checkpoint:
            self.detector = eval(checkpoint['config']['arch'])(detector_config)
            self.detector.load_state_dict(checkpoint['state_dict'])
        else:
            self.detector = checkpoint['model']
        self.detector.forPairing=True
        for param in self.detector.parameters(): 
            param.will_use_grad=param.requires_grad 
            param.requires_grad=False 
        self.detector_frozen=True

        self.pairer = PairingBoxNet(
                config['pairer_config'],
                detector_config,
                self.detector.last_channels,
                self.detector.scale)

        self.numBBTypes = self.pairer.numBBTypes
        self.numBBParams = self.pairer.numBBParams
        self.scale = self.pairer.scale
        self.anchors = self.pairer.anchors
        self.rotation = self.pairer.rotation

        self.storedImageName=None

 
    def unfreeze(self): 
        for param in self.detector.parameters(): 
            param.requires_grad=param.will_use_grad 
        self.detector_frozen=False
        

    def forward(self, image, queryMask,imageName=None):
        #print(image.size())
        #pad so that upsampling from model features works
        padH=(self.detector.scale-(image.size(2)%self.detector.scale))%self.detector.scale
        padW=(self.detector.scale-(image.size(3)%self.detector.scale))%self.detector.scale
        if padH!=0 or padW!=0:
            padder = torch.nn.ZeroPad2d((0,padW,0,padH))
            image = padder(image)
            queryMask = padder(queryMask)
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
        bbPredictions, offsetPredictions = self.pairer( image,
                                                        queryMask,
                                                        final_features, 
                                                        offsetPredictionsD)

        return bbPredictions, offsetPredictions
