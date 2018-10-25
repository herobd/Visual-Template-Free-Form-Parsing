from base import BaseModel
import torch
from model import PairingBoxNet, YoloBoxDetector


class PairingBoxFull(BaseModel):
    def __init__(self, config):
        super(PairingBoxFull, self).__init__(config)

        checkpoint = torch.load(config['detector_checkpoint'])
        detector_config = config['detector_config'] if 'detector_config' in config else checkpoint['config']
        if 'detector_type' in config:
            self.detector = eval(config['detector_type'])(detector_config)
            self.detector.load_state_dict(checkpoint['state_dict'])
        else:
            self.detector = checkpoint['model']
        self.detector_frozen=True

        self.pairer = PairingBoxNet(config['pairer_config'])

    def forward(self, image, queryMask):
        if self.detector_frozen:
            self.detector.eval()
            with torch.no_grad():
                bbPredictions, offsetPredictions, pointPreds, pixelPreds = self.detector(image)
        else:
            bbPredictions, offsetPredictions, pointPreds, pixelPreds = self.detector(image)

        bbPredictions, offsetPredictions, pointPreds, pixelPreds = self.pairer(image,queryMask,features, offsetPredictions)

        return bbPredictions, offsetPredictions
