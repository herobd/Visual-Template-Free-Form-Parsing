#from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
import math

import torch

from torch.nn.parameter import Parameter
from torch import nn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() #does this do parameter init?

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvWithAct(nn.Module):
        def __init__(self,in_ch,out_ch,norm='split_norm',dropout=0.4):
            super(GraphConvWithAct, self).__init__()
            self.graph_conv=GraphConvolution(in_ch,out_ch)
            self.split_normBB=None
            act_layers = []
            if norm=='batch_norm':
                act_layers.append(nn.BatchNorm1d(out_ch)) #essentially all the nodes compose a batch. There aren't enough some times
            elif norm=='group_norm':
                act_layers.append(nn.GroupNorm(4,out_ch))
            elif norm=='split_norm':
                #act_layers.append(nn.InstanceNorm1d(out_ch))
                self.split_normBB = nn.GroupNorm(4,out_ch)
                self.split_normRel = nn.GroupNorm(4,out_ch)
            if type(dropout) is float:
                act_layers.append(nn.Dropout(p=dropout,inplace=True))
            elif dropout:
                act_layers.append(nn.Dropout(p=0.3,inplace=True))
            act_layers.append(nn.ReLU(inplace=True))
            self.act_layers = nn.Sequential(*act_layers)

        def forward(self,node_features,adjacencyMatrix,numBBs):
            if self.split_normBB is not None:
                bb = self.split_normBB(node_features[:numBBs])
                rel = self.split_normRel(node_features[numBBs:])
                node_featuresX = torch.cat((bb,rel),dim=0)
            else:
                node_featuresX = node_features
            node_featuresX = self.act_layers(node_featuresX)
            node_featuresX = self.graph_conv(node_featuresX,adjacencyMatrix)
            return node_featuresX

class GraphResConv(nn.Module):
    """
    Two graph conv residual layer
    """

    def __init__(self, num_features,norm='group_norm',dropout=0.1):
        super(GraphResConv, self).__init__()
        self.side1=GraphConvWithAct(num_features,num_features,norm,dropout)
        self.side2=GraphConvWithAct(num_features,num_features,norm,dropout)
            

    def forward(self,node_features,adjacencyMatrix,numBBs):
        side = self.side1(node_features,adjacencyMatrix,numBBs)
        side = self.side2(side,adjacencyMatrix,numBBs)
        return node_features+side
