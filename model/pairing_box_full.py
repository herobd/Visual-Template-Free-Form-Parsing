from base import BaseModel
import torch
from .pairing_box_net import PairingBoxNet #, YoloBoxDetector


class PairingBoxFull(BaseModel):
    def __init__(self, config):
        super(PairingBoxFull, self).__init__(config)

        checkpoint = torch.load(config['detector_checkpoint'])
        detector_config = config['detector_config'] if 'detector_config' in config else checkpoint['config']['model']
        if 'detector_arch' in config:
            self.detector = eval(config['detector_arch'])(detector_config)
            self.detector.load_state_dict(checkpoint['state_dict'])
        else:
            self.detector = checkpoint['model']
        self.detector_frozen=True
        for param in self.detector.parameters():
            param.will_use_grad=param.requires_grad
            param.requires_grad=False

        self.pairer = PairingBoxNet(config['pairer_config'],detector_config,self.detector.last_channels)

    def unfreeze(self):
        for param in self.detector.parameters():
            param.requires_grad=param.will_use_grad
        self.detector_frozen=False

    def forward(self, image, queryMask):
        if self.detector_frozen:
            self.detector.eval()
            with torch.no_grad():
                bbPredictions, offsetPredictions, pointPreds, pixelPreds = self.detector(image)
        else:
            bbPredictions, offsetPredictions, pointPreds, pixelPreds = self.detector(image)

        bbPredictions, offsetPredictions, pointPreds, pixelPreds = self.pairer( image,
                                                                                queryMask,
                                                                                self.detector.final_features, 
                                                                                offsetPredictions)

        return bbPredictions, offsetPredictions
