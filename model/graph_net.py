import torch
from torch import nn
import torch.nn.functional as F
import math
import json
from .graphconvolution import GraphResConv, GraphConvWithAct, GraphTransformerBlock
from .net_builder import make_layers, getGroupSize
import numpy as np
import logging

#This assumes the classification of edges was done by the pairing_graph modules featurizer

class GraphNet(nn.Module):
    def __init__(self, config): # predCount, base_0, base_1):
        super(GraphNet, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        
        self.useRes = config['use_res'] if 'use_res' in config else 'loop'
        #how many times to re-apply graph conv layers
        self.repetitions = config['repetitions'] if 'repetitions' in config else 1 

        self.split_normBB=None
        act_layers = []

        prevCh=config['in_channels']
        if 'layers' in config:
            layer_desc = config['layers'] 
            if self.repetitions>1:
                for n in layer_desc:
                    assert(n==prevCh)
            self.layers=nn.ModuleList()
            for ch in layer_desc:
                if type(self.useRes)==str and 'layer' in self.useRes:
                    assert(prevCh==ch)
                    self.layers.append( GraphResConv(prevCh,config['norm'],config['dropout']) )
                else:
                    self.layers.append( GraphConvWithAct(prevCh,ch,config['norm'],config['dropout']) )
                prevCh=ch
            if 'norm' in config:
                if config['norm']=='batch_norm':
                    act_layers.append(nn.BatchNorm1d(prevCh)) #essentially all the nodes compose a batch
                elif config['norm']=='group_norm':
                    act_layers.append(nn.GroupNorm(getGroupSize(prevCh),prevCh)) 
                elif config['norm']=='split_norm':
                    #act_layers.append(nn.InstanceNorm1d(prevCh))
                    self.split_normBB = nn.GroupNorm(getGroupSize(prevCh),prevCh)
                    self.split_normRel = nn.GroupNorm(getGroupSize(prevCh),prevCh)
            if 'dropout' in config:
                if type(config['dropout']) is float:
                    act_layers.append(nn.Dropout(p=config['dropout'],inplace=True))
                elif config['dropout']:
                    act_layers.append(nn.Dropout(p=0.3,inplace=True))
        else:
            self.layers=None
            #Transformers!
            num_feats = config['in_channels']#config['trans_features']
            num_layers = config['num_trans']
            num_heads = config['num_heads']
            num_ffnn_layers = config['num_ffnn_layers'] if 'num_ffnn_layers' in config else 2
            num_ffnn_feats = config['num_ffnn_feats'] if 'num_ffnn_feats' in config else num_feats
            layers = [ GraphTransformerBlock(num_feats,num_heads,num_ffnn_layers,num_ffnn_feats) for i in range(num_layers)]
            self.transformers = nn.Sequential(*layers)
            act_layers.append(nn.Dropout(p=0.05,inplace=True))

            if 'encoder' in config:
                if type(config['encoder']) is list:
                    encoder_layers = config['encoder'] + ['FCnR{}'.format(num_feats)]
                    layers,_ = make_layers(encoder_layers)
                    self.encoder_fc = nn.Sequential(*layers)
                    self.encoder=None
                else:
                    num_encode_layers = config['encoder'] if type(config['encoder']) is int else num_layers
                    layers = [ GraphTransformerBlock(num_feats,num_heads,num_ffnn_layers,num_ffnn_feats) for i in range(num_encode_layers)]
                    self.encoder = nn.Sequential(*layers)
                    self.encoder_fc = None
            else:
                self.encoder = None
                self.encoder_fc = None
        if 'random_reps' in config:
            self.randomReps=True
            if type(config['random_reps']) is int:
                self.minReps = 0
                self.maxReps = config['random_reps']
            elif type(config['random_reps']) is list:
                assert(len(config['random_reps'])==2)
                self.minReps = config['random_reps'][0]
                self.maxReps = config['random_reps'][1]
            else:
                print('Bad random_reps: {}'.format(config['random_reps']))
                exit()

        else:
            self.randomReps=False
        act_layers.append(nn.ReLU(inplace=True))
            

        self.act_layers = nn.Sequential(*act_layers)

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



    def forward(self, node_features, adjacencyMatrix, numBBs):

        if self.randomReps:
            if self.training:
                repetitions=np.random.randint(self.minReps,self.maxReps+1)
            else:
                repetitions=self.maxReps
        else:
            repetitions=self.repetitions
        #it is assumed these features are not activated
        node_featuresX = node_features
        if self.layers is None:
            adjacencyMatrix = adjacencyMatrix[0].to_dense()
        if self.encoder is not None:
            node_featuresX,_,_=self.encoder((node_featuresX,adjacencyMatrix,numBBs))
        elif self.encoder_fc is not None:
            node_featuresX=self.encoder_fc(node_featuresX)
        for i in range(repetitions):
            if self.layers is not None:
                side=node_featuresX
                for graph_conv in self.layers:
                    side = graph_conv(side,adjacencyMatrix,numBBs)
                if type(self.useRes)==str and 'loop' in self.useRes:
                    node_featuresX=side+node_featuresX
                else:
                    node_featuresX=side
            else:
                node_featuresX,_,_ = self.transformers((node_featuresX,adjacencyMatrix,numBBs))
                #import pdb;pdb.set_trace()
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
        



    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)
