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


    def forward(self, node_features, adjacencyMatrix, edge_features):
        #expects edge_features as batch currently

        #adj = torch.spmm(self.weight,edge_features) + self.bias 
        


        #return
        return None,edge_features


