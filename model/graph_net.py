import torch
from torch import nn
import torch.nn.functional as F
import math
import json
from .graphconvolution import GraphConvolution

#This assumes the classification of edges was done by the pairing_graph modules featurizer

class GraphNet(nn.Module):
    def __init__(self, config): # predCount, base_0, base_1):
        super(GraphNet, self).__init__()

        class GraphConvWithAct(nn.Module):
            def __init__(self,in_ch,out_ch,config):
                super(GraphConvWithAct, self).__init__()
                self.graph_conv=GraphConvolution(in_ch,out_ch)
                self.split_normBB=None
                act_layers = []
                if 'norm' in config:
                    if config['norm']=='batch_norm':
                        act_layers.append(nn.BatchNorm1d(out_ch)) #essentially all the nodes compose a batch
                    elif config['norm']=='split_norm':
                        #act_layers.append(nn.InstanceNorm1d(out_ch))
                        self.split_normBB = nn.BatchNorm1d(out_ch)
                        self.split_normRel = nn.BatchNorm1d(out_ch)
                if 'dropout' in config:
                    if type(config['dropout']) is float:
                        act_layers.append(nn.Dropout(p=config['dropout'],inplace=True))
                    elif config['dropout']:
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
        
        #how many times to re-apply graph conv layers
        self.repetitions = config['repetitions'] if 'repetitions' in config else 1 

        layer_desc = config['layers'] if 'layers' in config else [256,256,256]
        prevCh=config['in_channels']
        if self.repetitions>1:
            for n in layer_desc:
                assert(n==prevCh)
        self.layers=nn.ModuleList()
        for ch in layer_desc:
            self.layers.append( GraphConvWithAct(prevCh,ch,config) )
            prevCh=ch
        numBBOut = config['bb_out'] if 'bb_out' in config else 0
        numRelOut = config['rel_out'] if 'rel_out' in config else 1

        if numBBOut>0:
            self.bb_out=nn.Linear(prevCh,numBBOut)
        else:
            self.bb_out=lambda x:  None
        if numRelOut>0:
            self.rel_out=nn.Linear(prevCh,numRelOut)
        else:
            self.rel_out=lambda x:  None

        self.split_normBB=None
        act_layers = []
        if 'norm' in config:
            if config['norm']=='batch_norm':
                act_layers.append(nn.BatchNorm1d(prevCh)) #essentially all the nodes compose a batch
            elif config['norm']=='split_norm':
                #act_layers.append(nn.InstanceNorm1d(prevCh))
                self.split_normBB = nn.BatchNorm1d(prevCh)
                self.split_normRel = nn.BatchNorm1d(prevCh)
        if 'dropout' in config:
            if type(config['dropout']) is float:
                act_layers.append(nn.Dropout(p=config['dropout'],inplace=True))
            elif config['dropout']:
                act_layers.append(nn.Dropout(p=0.3,inplace=True))
        act_layers.append(nn.ReLU(inplace=True))
        self.act_layers = nn.Sequential(*act_layers)


    def forward(self, node_features, adjacencyMatrix, numBBs):
        #it is assumed these features are not activated
        node_featuresX = node_features
        for i in range(self.repetitions):
            side=node_featuresX
            for graph_conv in self.layers:
                side = graph_conv(side,adjacencyMatrix,numBBs)
            node_featuresX=side+node_featuresX
        #the graph conv layers are residual, so activation is applied here
        if self.split_normBB is not None:
            bb = self.split_normBB(node_featuresX[:numBBs])
            rel = self.split_normRel(node_featuresX[numBBs:])
            node_featuresX = torch.cat((bb,rel),dim=0)
        node_featuresX = self.act_layers(node_featuresX)

        bb_features = node_featuresX[:numBBs]
        rel_features = node_featuresX[numBBs:]
        return self.bb_out(bb_features), self.rel_out(rel_features)
    
        #return None, node_features[:,0:1]
        




