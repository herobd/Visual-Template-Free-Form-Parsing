import torch
from torch import nn
import torch.nn.functional as F
import math
import json
from .net_builder import make_layers, getGroupSize
from .graphconvolution import MultiHeadedAttention
import numpy as np
import logging

#This assumes the inputs are not activated
class MetaGraphAttentionLayer(nn.Module):
    def __init__(self, ch,heads=4,dropout=0.1,norm='group',useRes=True,useGlobal=False,hidden_ch=None,agg_thinker='cat'): 
        super(MetaGraphAttentionLayer, self).__init__()
        
        self.thinker=agg_thinker
        if hidden_ch is None:
            hidden_ch=ch
        self.res=useRes

        edge_in=3
        if self.thinker=='cat':
            node_in=2
        else:
            node_in=1
        act=[[] for i in range(5)]
        dropN=[]
        self.actN=[]
        if useGlobal:
            self.actN_u=[]
        if 'group' in norm:
            #for i in range(5):
                #act[i].append(nn.GroupNorm(getGroupSize(hidden_ch),hidded_ch))
            #global    
            act[0].append(nn.GroupNorm(getGroupSize(3*ch,24),3*ch))
            act[1].append(nn.GroupNorm(getGroupSize(hidden_ch),hidded_ch))
            #edge
            self.actN.append(nn.GroupNorm(getGroupSize(ch),ch))
            if useGlobal:
                self.actN_u.append(nn.GroupNorm(getGroupSize(ch),ch))
            act[2].append(nn.GroupNorm(getGroupSize(hidden_ch),hidded_ch))
            #node
            act[3].append(nn.GroupNorm(getGroupSize(node_in*ch,node_in*8),node_in*ch))
            act[4].append(nn.GroupNorm(getGroupSize(hidden_ch),hidded_ch))
            #out
            act[5].append(nn.GroupNorm(getGroupSize(ch),ch))
            act[6].append(nn.GroupNorm(getGroupSize(ch),ch))
        elif 'batch' in norm:
            #for i in range(5):
            #    act[i].append(nn.BatchNorm1d(ch))
            #global
            act[0].append(nn.BatchNorm1d(3*ch))
            act[1].append(nn.BatchNorm1d(hidden_ch))
            #edge
            self.actN.append(nn.BatchNorm1d(ch))
            if useGlobal:
                self.actN_u.append(nn.BatchNorm1d(ch))
            act[2].append(nn.BatchNorm1d(hidden_ch))
            #node
            act[3].append(nn.BatchNorm1d(node_in*ch))
            act[4].append(nn.BatchNorm1d(hidden_ch))
            #out
            act[5].append(nn.BatchNorm1d(ch))
            act[6].append(nn.BatchNorm1d(ch))
        elif 'instance' in norm:
            #for i in range(5):
            #    act[i].append(nn.InstanceNorm1d(ch))
            #global
            act[0].append(nn.InstanceNorm1d(3*ch))
            act[1].append(nn.InstanceNorm1d(hidden_ch))
            #edge
            self.actN.append(nn.InstanceNorm1d(ch))
            if useGlobal:
                self.actN_u.append(nn.InstanceNorm1d(ch))
            act[2].append(nn.InstanceNorm1d(hidden_ch))
            #node
            act[3].append(nn.InstanceNorm1d(node_in*ch))
            act[4].append(nn.InstanceNorm1d(hidden_ch))
            #out
            act[5].append(nn.InstanceNorm1d(ch))
            act[6].append(nn.InstanceNorm1d(ch))
        else:
            raise NotImplemented('Unknown norm: {}'.format(norm))
        if dropout is not None:
            if type(dropout) is float:
                da=dropout
            else:
                da=0.1
            for i in range(7):
                act[i].append((nn.Dropout(p=da,inplace=True))
            dropN.append((nn.Dropout(p=da,inplace=True))
        for i in range(7):
            act[i].append(nn.ReLU(inplace=True))
        self.useGlobal=useGlobal
        if useGlobal:
            edge_in +=1
            node_in +=1
            self.global_mlp = nn.Sequential(*(act[0]),nn.Linear(ch*3, hidden_ch), *(act[1]), nn.Linear(hidden_ch, ch))

        self.edge_mlp = nn.Sequential(*(act[2]),nn.Linear(ch*edge_in, hidden_ch), *(act[3]), nn.Linear(hidden_ch, ch))
        self.node_mlp = nn.Sequential(*dropN,nn.Linear(ch*node_in, hidden_ch), *(act[4]), nn.Linear(hidden_ch, ch))
        self.mhAtt = MultiHeadedAttention(heads,ch)


        def edge_model(source, target, edge_attr, u):
            # source, target: [E, F_x], where E is the number of edges.
            # edge_attr: [E, F_e]
            # u: [B, F_u], where B is the number of graphs.
            if u is not None:
                assert(u.size(0)==1)
                us = u.expand(source.size(0),u.size(1))
                out = torch.cat([source, target, edge_attr,us], dim=1)
            else:
                out = torch.cat([source, target, edge_attr], dim=1)
            out = self.edge_mlp(out)
            if self.res:
                out+=edge_attr
            return out

        def node_model(x, edge_index, edge_attr, u):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            row, col = edge_index
            eRange = torch.arange(col.size(0))
            mask = torch.zeros(x.size(0), edge_attr.size(0))
            mask[col,eRange]=1
            #Add batch dimension
            x_b = x[None,...]
            edge_attr_b = edge_attr[None,...]
            g = self.mhAtt(x_b,edge_attr_b,edge_attr_b,mask) 
            #above uses unnormalized, unactivated features.

            xa = self.actN(x)
            if u is not None:
                assert(u.size(0)==1)
                us = u.expand(source.size(0),u.size(1))
                us = self.actN_u(us)
                if self.thinker=='cat':
                    out= self.node_mlp(torch.cat((xa,g,us),dim=1))
                elif self.thinker=='add':
                    g+=xa
                    out= self.node_mlp(torch.cat((g,us),dim=1))
            else:
                if self.thinker=='cat':
                    out= self.node_mlp(torch.cat((xa,g),dim=1))
                elif self.thinker=='add':
                    out= self.node_mlp(g+xa)
            if self.res:
                out+=x
            return out

        def global_model(x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            if self.useGlobal:
                if batch is None:
                    out = torch.cat([u, torch.mean(x,dim=0),torch.mean(edge_attr,dim=0)],dim=1)
                else:
                    raise NotImplemented('batching not implemented for scatter_mean of edge_attr')
                    out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
                out = self.global_mlp(out)
                if self.res:
                    out+=u
                return out
            else:
                return None

        self.layer = MetaLayer(edge_model, node_model, global_model)

    def forward(self, input): 
        node_features, edge_indexes, edge_features, u_features = input
        return self.layer(node_features, edge_indexes, edge_features, u_features)

class MetaGraphNet(nn.Module):
    def __init__(self, config): # predCount, base_0, base_1):
        super(MetaGraphNet, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        
        self.useRepRes = config['use_reptition_res'] if 'use_repetition_res' in config else False
        #how many times to re-apply main layers
        self.repetitions = config['repetitions'] if 'repetitions' in config else 1 
        self.randomReps = False

        ch = config['in_channels']
        layerType = 'attention'
        layerCount = config['num_layers'] if 'num_layers' in config else 3
        norm = config['norm'] if 'norm' in config else 'group'
        
        if layerType=='attention':
            layers = [MetaGraphAttentionLayer(ch,heads=4,dropout=0.1,norm=norm,useRes=True,useGlobal=False,hidden_ch=None,agg_thinker='cat') for i in range(layerCount)]
            self.main_layers = nn.Sequential(*layers)
        else:
            print('Unknown layer type: {}'.format(layerType))
            exit()

        if 'num_input_layers' in config and config['num_input_layers']>0:
            if layerType=='attention':
                layers = [MetaGraphAttentionLayer(ch,heads=4,dropout=0.1,norm=norm,useRes=True,useGlobal=False,hidden_ch=None,agg_thinker='cat') for i in range(layerCount)]
                self.input_layers = nn.Sequential(*layers)
        else:
            self.input_layers = None

        if numNodeOut>0:
            self.node_out_layers=nn.Sequential(*act[5],nn.Linear(ch,numNodeOut))
        else:
            self.node_out_layers=lambda x:  None
        if numEdgeOut>0:
            self.edge_out_layers=nn.Sequential(*act[6],nn.Linear(ch,numEdgeOut))
        else:
            self.edge_out_layers=lambda x:  None



    def forward(self, input):
        node_features, edge_indexes, edge_features, u_features = input
        if self.randomReps:
            if self.training:
                repetitions=np.random.randint(self.minReps,self.maxReps+1)
            else:
                repetitions=self.maxReps
        else:
            repetitions=self.repetitions

        #if self.useRes:
            #node_featuresA = self.act_layers(node_features)
            #edge_featuresA = self.act_layers(edge_features)
            #if u_features is not None:
            #    u_featuresA = self.act_layers(u_features)
            #else:
            #    u_featuresA = None

        if self.input_layers is not None:
            node_features, edge_indexes, edge_features, u_features = self.input_layers(node_features, edge_indexes, edge_features, u_features)
    
        out_nodes = []
        out_edges = [] #for holding each repititions outputs, so we can backprop on all of them

        for i in range(repititions):
            node_featuresT, edge_indexesT, edge_featuresT, u_featuresT = self.main_layers(node_features, edge_indexes, edge_features, u_features)
            if self.useRepRes:
                node_features+=node_featuresT
                edge_features+=edge_featuresT
                if u_features is not None:
                    u_features+=u_featuresT
                else:
                    u_features=u_featuresT
            else:
                node_features=node_featuresT
                edge_features=edge_featuresT
                u_features=u_featuresT
            
            node_out = self.node_out_layers(node_features)
            edge_out = self.edge_out_layers(edge_features)

            out_nodes.append(node_out)
            out_edges.append(edge_out)

        return out_nodes, out_edges



    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)
