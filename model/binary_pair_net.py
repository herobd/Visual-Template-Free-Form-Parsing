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
import torch
from torch import nn
#from base import BaseModel
import torch.nn.functional as F
#from torch.nn.utils.weight_norm import weight_norm
import math
import json
#from .net_builder import make_layers

#This assumes the classification of edges was done by the pairing_graph modules featurizer

class BinaryPairNet(nn.Module):
    def __init__(self, config): # predCount, base_0, base_1):
        super(BinaryPairNet, self).__init__()
        raise NotImplemented('Changes have broken this class, use BinaryPairReal')


    def forward(self, node_features, adjacencyMatrix, numBBs):
        #expects edge_features as batch currently

        #adj = torch.spmm(self.weight,edge_features) + self.bias 
        


        #return
        return None,node_features


