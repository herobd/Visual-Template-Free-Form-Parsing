"""
    Copyright 2019 Brian Davis
    Visual-Template-free-Form-Parsting is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Visual-Template-free-Form-Parsting is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Visual-Template-free-Form-Parsting.  If not, see <https://www.gnu.org/licenses/>.
"""
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

        reverse = config['reverse_activation'] if 'reverse_activation' in config else False #for resnet stuff
        norm = config['norm'] if 'norm' in config else 'batch_norm'
        dropout = float(config['dropout']) if 'dropout' in config else 0.4

        if numLayers==0:
            assert(featSize==hiddenSize)

        layers= []
        for i in range(numLayers):
            if i==0:
                inSize=featSize
            else:
                inSize=hiddenSize

            if not reverse:
                layers.append(nn.Linear(inSize,hiddenSize))
            if norm=='batch_norm':
                layers.append(nn.BatchNorm1d(hiddenSize))
            elif norm=='group_norm':
                layers.append(nn.GroupNorm(getGroupSize(hiddenSize),hiddenSize))
            if not reverse or i!=0:
                layers += [
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                    ]
            if reverse:
                layers.append(nn.Linear(inSize,hiddenSize))
        if outSize>0:
            layers.append(nn.Linear(hiddenSize,outSize))
        self.layers=nn.Sequential(*layers)

    def forward(self,input):
        return self.layers(input)
