import torch
from torch import nn
#from base import BaseModel
import torch.nn.functional as F
#from torch.nn.utils.weight_norm import weight_norm
import math
import json
from .net_builder import make_layers

#This assumes the classification of edges was done by the pairing_graph modules featurizer

class BinaryPairReal(nn.Module):
    def __init__(self, config): # predCount, base_0, base_1):
        super(BinaryPairReal, self).__init__()
        numBBOut = config['bb_out'] if 'bb_out' in config else 0
        numRelOut = config['rel_out'] if 'rel_out' in config else 1
        assert(numBBOut==0)

        in_ch=config['in_channels']

        norm = config['norm'] if 'norm' in config else 'group_norm'
        dropout = config['dropout'] if 'dropout' in config else True

        layer_desc = config['layers'] if 'layers' in config else ['FC256','FC256','FC256']
        layer_desc = [in_ch]+layer_desc+['FCnR{}'.format(numRelOut)]
        layers, last_ch_relC = make_layers(layer_desc,norm=norm,dropout=dropout)
        self.layers = nn.Sequential(*layers)

        #This is written to by the PairingGraph object (which holds this one)
        self.numShapeFeats = config['num_shape_feats'] if 'num_shape_feats' in config else 16

        

        if 'shape_layers' in config:
            layer_desc = config['shape_layers']
            layer_desc = [self.numShapeFeats]+layer_desc+['FCnR{}'.format(numRelOut)]
            layers, last_ch_relC = make_layers(layer_desc,norm=norm,dropout=dropout)
            self.shape_layers = nn.Sequential(*layers)
        else:
            self.shape_layers=None



    def forward(self, node_features, adjacencyMatrix, numBBs):
        res = self.layers(node_features)
        if self.shape_layers is not None:
            res2 = self.shape_layers(node_features[:,-self.numShapeFeats:])
            res = (res+res2)/2
        return None,res


