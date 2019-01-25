from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.weight_norm import weight_norm

class SimpleNN(BaseModel):
    def __init__(self, config):
        super(SimpleNN, self).__init__(config)

        featSize = config['feat_size'] if 'feat_size' in config else 10
        numLayers = config['num_layers'] if 'num_layers' in config else 2
        hiddenSize = config['hidden_size'] if 'hidden_size' in config else 1024
        outSize = config['out_size'] if 'out_size' in config else 1

        layers= []
        for i in range(numLayers):
            if i==0:
                inSize=featSize
            else:
                inSize=hiddenSize
            layers += [
                nn.Linear(inSize,hiddenSize),
                nn.BatchNorm1d(hiddenSize),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4)
                ]
        layers.append(nn.Linear(hiddenSize,outSize))
        self.layers=nn.Sequential(*layers)

    def forward(self,input):
        return self.layers(input)
