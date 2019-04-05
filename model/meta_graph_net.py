import torch
from torch import nn
import torch.nn.functional as F
import math
import json
from .net_builder import make_layers, getGroupSize
from .graphconvolution import MultiHeadedAttention
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_mean
import numpy as np
import logging

class MetaGraphFCEncoderLayer(nn.Module):
    def __init__(self, feat,edgeFeat,ch):
        super(MetaGraphFCEncoderLayer, self).__init__()

        self.node_layer = nn.Linear(feat,ch)
        if type(edgeFeat) is int and edgeFeat>0:
            self.edge_layer = nn.Linear(edgeFeat,ch)
            self.hasEdgeInfo=True
        else:
            self.edge_layer = nn.Linear(feat*2,ch)
            self.hasEdgeInfo=False

    def forward(self, input): 
        node_features, edge_indexes, edge_features, u_features = input
        node_featuresN = self.node_layer(node_features)
        if not self.hasEdgeInfo:
            edge_features = node_features[edge_indexes].permute(1,0,2).reshape(edge_indexes.size(1),-1)
        edge_features = self.edge_layer(edge_features)


        return node_featuresN, edge_indexes, edge_features, u_features

class MetaGraphMeanLayer(torch.nn.Module):
    def __init__(self, ch,useGlobal):
        super(MetaGraphMeanLayer, self).__init__()

        self.edge_mlp = nn.Sequential(nn.Linear(3*ch, ch), nn.ReLU(), nn.Linear(ch, ch))
        self.node_mlp = nn.Sequential(nn.Linear(2*ch, ch), nn.ReLU(), nn.Linear(ch, ch))
        if useGlobal:
            self.global_mlp = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Linear(ch, ch))

        def edge_model(source, target, edge_attr, u):
            # source, target: [E, F_x], where E is the number of edges.
            # edge_attr: [E, F_e]
            # u: [B, F_u], where B is the number of graphs.
            out = torch.cat([source, target, edge_attr], dim=1)
            return self.edge_mlp(out)

        def node_model(x, edge_index, edge_attr, u):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            row, col = edge_index
            out = torch.cat([x[col], edge_attr], dim=1)
            out = self.node_mlp(out)
            return scatter_mean(out, row, dim=0, dim_size=x.size(0))

        def global_model(x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            if u is None:
                return None
            out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
            return self.global_mlp(out)

        self.op = MetaLayer(edge_model, node_model, global_model)

    def forward(self, input):
        x, edge_index, edge_attr, u = input
        batch=None
        x, edge_attr, u = self.op(x, edge_index, edge_attr, u, batch)
        return x, edge_index, edge_attr, u


#This assumes the inputs are not activated
class MetaGraphSelectiveLayer(nn.Module):
    def __init__(self, ch,heads=4,dropout=0.1,norm='group',useRes=True,useGlobal=False,hidden_ch=None,agg_thinker='cat',edge_decider=None): 
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
        act=[[] for i in range(7)]
        dropN=[]
        self.actN=[]
        if useGlobal:
            self.actN_u=[]
        if 'group' in norm:
            #for i in range(5):
                #act[i].append(nn.GroupNorm(getGroupSize(hidden_ch),hidden_ch))
            #global    
            act[0].append(nn.GroupNorm(getGroupSize(3*ch,24),3*ch))
            act[1].append(nn.GroupNorm(getGroupSize(hidden_ch),hidden_ch))
            #edge
            act[2].append(nn.GroupNorm(getGroupSize(edge_in*ch,edge_in*8),edge_in*ch))
            act[3].append(nn.GroupNorm(getGroupSize(hidden_ch),hidden_ch))
            #node
            self.actN.append(nn.GroupNorm(getGroupSize(ch),ch))
            if useGlobal:
                self.actN_u.append(nn.GroupNorm(getGroupSize(ch),ch))
            act[4].append(nn.GroupNorm(getGroupSize(hidden_ch),hidden_ch))
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
            act[2].append(nn.BatchNorm1d(edge_in*ch))
            act[3].append(nn.BatchNorm1d(hidden_ch))
            #node
            self.actN.append(nn.BatchNorm1d(ch))
            if useGlobal:
                self.actN_u.append(nn.BatchNorm1d(ch))
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
            act[2].append(nn.InstanceNorm1d(edge_in*ch))
            act[3].append(nn.InstanceNorm1d(hidden_ch))
            #node
            self.actN.append(nn.InstanceNorm1d(ch))
            if useGlobal:
                self.actN_u.append(nn.InstanceNorm1d(ch))
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
                act[i].append(nn.Dropout(p=da,inplace=True))
            dropN.append(nn.Dropout(p=da,inplace=True))
        for i in range(7):
            act[i].append(nn.ReLU(inplace=True))
        self.actN=nn.Sequential(*self.actN)
        if useGlobal:
            self.actN_u=nn.Sequential(*self.actN_u)

        self.useGlobal=useGlobal
        if useGlobal:
            edge_in +=1
            node_in +=1
            self.global_mlp = nn.Sequential(*(act[0]),nn.Linear(ch*3, hidden_ch), *(act[1]), nn.Linear(hidden_ch, ch))

        self.edge_mlp = nn.Sequential(*(act[2]),nn.Linear(ch*edge_in, hidden_ch), *(act[3]), nn.Linear(hidden_ch, ch))
        if edge_decider is None:
            self.edge_decider = nn.Sequential(*(act[6]),nn.Linear(ch, 1), nn.Sigmoid())
        else:
            self.edge_decider = nn.Sequential(edge_decider, SharpSigmoid(0.2))
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
            decision = self.edge_decider(out)
            out *= decision
            if self.res:
                out+=edge_attr
            return out

        def node_model(x, edge_index, edge_attr, u):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            row, col = edge_index
            out = torch.cat([x[col], edge_attr], dim=1)
            out = self.node_mlp(out)
            out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
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
        node_features,edge_features,u_features = self.layer(node_features, edge_indexes, edge_features, u_features)
        return node_features, edge_indexes, edge_features, u_features

#1/(1+(e^(-(x+1)/0.5)))
class SharpSigmoid(nn.Module):
    def __init__(self,center,sharp=0.5):
        super(SharpSigmoid, self).__init__()
        self.c=center
        self.sharp=sharp
    def forward(self,input):
        return 1/(1+torch.exp(-(input+self.c)/self.sharp))


#This assumes the inputs are not activated
class MetaGraphAttentionLayer(nn.Module):
    def __init__(self, ch,heads=4,dropout=0.1,norm='group',useRes=True,useGlobal=False,hidden_ch=None,agg_thinker='cat',soft_prune_edges=False,edge_decider=None): 
        super(MetaGraphAttentionLayer, self).__init__()
        
        self.thinker=agg_thinker
        self.soft_prune_edges=soft_prune_edges
        if hidden_ch is None:
            hidden_ch=ch
        self.res=useRes

        edge_in=3
        if self.thinker=='cat':
            node_in=2
        else:
            node_in=1
        act=[[] for i in range(7)]
        dropN=[]
        self.actN=[]
        if useGlobal:
            self.actN_u=[]
        if 'group' in norm:
            #for i in range(5):
                #act[i].append(nn.GroupNorm(getGroupSize(hidden_ch),hidden_ch))
            #global    
            act[0].append(nn.GroupNorm(getGroupSize(3*ch,24),3*ch))
            act[1].append(nn.GroupNorm(getGroupSize(hidden_ch),hidden_ch))
            #edge
            act[2].append(nn.GroupNorm(getGroupSize(edge_in*ch,edge_in*8),edge_in*ch))
            act[3].append(nn.GroupNorm(getGroupSize(hidden_ch),hidden_ch))
            #node
            self.actN.append(nn.GroupNorm(getGroupSize(ch),ch))
            if useGlobal:
                self.actN_u.append(nn.GroupNorm(getGroupSize(ch),ch))
            act[4].append(nn.GroupNorm(getGroupSize(hidden_ch),hidden_ch))
            #decider
            act[5].append(nn.GroupNorm(getGroupSize(ch),ch))
        elif 'batch' in norm:
            #for i in range(5):
            #    act[i].append(nn.BatchNorm1d(ch))
            #global
            act[0].append(nn.BatchNorm1d(3*ch))
            act[1].append(nn.BatchNorm1d(hidden_ch))
            #edge
            act[2].append(nn.BatchNorm1d(edge_in*ch))
            act[3].append(nn.BatchNorm1d(hidden_ch))
            #node
            self.actN.append(nn.BatchNorm1d(ch))
            if useGlobal:
                self.actN_u.append(nn.BatchNorm1d(ch))
            act[4].append(nn.BatchNorm1d(hidden_ch))
            #decider
            act[5].append(nn.BatchNorm1d(ch))
        elif 'instance' in norm:
            #for i in range(5):
            #    act[i].append(nn.InstanceNorm1d(ch))
            #global
            act[0].append(nn.InstanceNorm1d(3*ch))
            act[1].append(nn.InstanceNorm1d(hidden_ch))
            #edge
            act[2].append(nn.InstanceNorm1d(edge_in*ch))
            act[3].append(nn.InstanceNorm1d(hidden_ch))
            #node
            self.actN.append(nn.InstanceNorm1d(ch))
            if useGlobal:
                self.actN_u.append(nn.InstanceNorm1d(ch))
            act[4].append(nn.InstanceNorm1d(hidden_ch))
            #decider
            act[5].append(nn.InstanceNorm1d(ch))
        else:
            raise NotImplemented('Unknown norm: {}'.format(norm))
        if dropout is not None:
            if type(dropout) is float:
                da=dropout
            else:
                da=0.1
            for i in range(7):
                act[i].append(nn.Dropout(p=da,inplace=True))
            dropN.append(nn.Dropout(p=da,inplace=True))
        for i in range(7):
            act[i].append(nn.ReLU(inplace=True))
        self.actN=nn.Sequential(*self.actN)
        if useGlobal:
            self.actN_u=nn.Sequential(*self.actN_u)

        self.useGlobal=useGlobal
        if useGlobal:
            edge_in +=1
            node_in +=1
            self.global_mlp = nn.Sequential(*(act[0]),nn.Linear(ch*3, hidden_ch), *(act[1]), nn.Linear(hidden_ch, ch))

        self.edge_mlp = nn.Sequential(*(act[2]),nn.Linear(ch*edge_in, hidden_ch), *(act[3]), nn.Linear(hidden_ch, ch))
        
        if self.soft_prune_edges:
            if edge_decider is None:
                self.edge_decider = nn.Sequential(*(act[5]),nn.Linear(ch, 1), nn.Sigmoid())
                # we shift the mean up to bias keeping edges (should help begining of training
                self.edge_decider[len(act[5])].bias = nn.Parameter(self.edge_decider[len(act[5])].bias.data + 2.0/self.edge_decider[len(act[5])].bias.size(0))
            else:
                # we shouldn't need that bias here since it's already getting trained
                self.edge_decider = nn.Sequential(edge_decider, SharpSigmoid(-1))
        else:
            self.edge_decider = None
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
            if self.soft_prune_edges:
                pruneDecision = self.edge_decider(out)
                #print(pruneDecision)
                out *= self.soft_prune_edges
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
            mask = mask.to(x.device)
            #Add batch dimension
            x_b = x[None,...]
            edge_attr_b = edge_attr[None,...]
            g = self.mhAtt(x_b,edge_attr_b,edge_attr_b,mask) 
            #above uses unnormalized, unactivated features.
            g = g[0] #discard batch dim

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
        node_features,edge_features,u_features = self.layer(node_features, edge_indexes, edge_features, u_features)
        return node_features, edge_indexes, edge_features, u_features

class MetaGraphNet(nn.Module):
    def __init__(self, config): # predCount, base_0, base_1):
        super(MetaGraphNet, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        
        self.useRepRes = config['use_reptition_res'] if 'use_repetition_res' in config else False
        #how many times to re-apply main layers
        self.repetitions = config['repetitions'] if 'repetitions' in config else 1 
        self.randomReps = False

        ch = config['in_channels']
        layerType = config['layer_type'] if 'layer_type' in config else 'attention'
        layerCount = config['num_layers'] if 'num_layers' in config else 3
        numNodeOut = config['node_out']
        numEdgeOut = config['edge_out']
        norm = config['norm'] if 'norm' in config else 'group'
        dropout = config['dropout'] if 'dropout' in config else 0.1
        hasEdgeInfo = config['input_edge'] if 'input_edge' in config else True

        self.trackAtt=False

        actN=[]
        actE=[]
        if 'group' in norm:
            actN.append(nn.GroupNorm(getGroupSize(ch),ch))
            actE.append(nn.GroupNorm(getGroupSize(ch),ch))
        elif norm:
            raise NotImplemented('Havent implemented other norms ({}) in MetaGraphNet'.format(norm))
        if dropout:
            actN.append(nn.Dropout(p=dropout,inplace=True))
            actE.append(nn.Dropout(p=dropout,inplace=True))
        actN.append(nn.ReLU(inplace=True))
        actE.append(nn.ReLU(inplace=True))
        if numNodeOut>0:
            self.node_out_layers=nn.Sequential(*actN,nn.Linear(ch,numNodeOut))
        else:
            self.node_out_layers=lambda x:  None
        if numEdgeOut>0:
            self.edge_out_layers=nn.Sequential(*actE,nn.Linear(ch,numEdgeOut))
        else:
            self.edge_out_layers=lambda x:  None


        if layerType=='attention':
            heads = config['num_heads'] if 'num_heads' in config else 4
            soft_prune_edges = config['soft_prune_edges'] if 'soft_prune_edges' in config else False
            if 'prune_with_classifier' in config and config['prune_with_classifier']:
                edge_decider = self.edge_out_layers
            else:
                edge_decider = None
            if soft_prune_edges=='last':
                soft_prune_edges_l = ([False]*(layerCount-1)) + [True]
            elif soft_prune_edges:
                soft_prune_edges_l = [True]*layerCount
            else:
                soft_prune_edges_l = [False]*layerCount


            layers = [MetaGraphAttentionLayer(ch,heads=heads,dropout=dropout,norm=norm,useRes=True,useGlobal=False,hidden_ch=None,agg_thinker='cat',soft_prune_edges=soft_prune_edges_l[i],edge_decider=edge_decider) for i in range(layerCount)]
            self.main_layers = nn.Sequential(*layers)
        elif layerType=='mean':
            layers = [MetaGraphMeanLayer(ch,False) for i in range(layerCount)]
            self.main_layers = nn.Sequential(*layers)
        else:
            print('Unknown layer type: {}'.format(layerType))
            exit()

        self.input_layers=None
        if 'encode_type' in config:
            inputLayerType = config['encode_type']
            if 'fc' in inputLayerType:
                infeats = config['infeats']
                infeatsEdge = config['infeats_edge'] if 'infeats_edge' in config else 0
                self.input_layers = MetaGraphFCEncoderLayer(infeats,infeatsEdge,ch)
            if 'attention' in inputLayerType:
                #layers = [MetaGraphAttentionLayer(ch,heads=heads,dropout=dropout,norm=norm,useRes=True,useGlobal=False,hidden_ch=None,agg_thinker='cat') for i in range(layerCount)]
                #self.input_layers = nn.Sequential(*layers)
                layer = MetaGraphAttentionLayer(ch,heads=heads,dropout=dropout,norm=norm,useRes=True,useGlobal=False,hidden_ch=None,agg_thinker='cat')
                if self.input_layers is None:
                    self.input_layers = layer
                else:
                    self.input_layers = nn.Sequential(self.input_layers,layer)
            self.force_encoding = config['force_encoding'] if 'force_encoding' in config else False
        else:
            assert(hasEdgeInfo==True)




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

        out_nodes = []
        out_edges = [] #for holding each repititions outputs, so we can backprop on all of them

        if self.trackAtt:
            self.attn=[]

        if self.input_layers is not None:
            node_features, edge_indexes, edge_features, u_features = self.input_layers((node_features, edge_indexes, edge_features, u_features))
            if self.trackAtt:
                self.attn.append(self.input_layers.mhAtt.attn)

            if self.force_encoding:
                node_out = self.node_out_layers(node_features)
                edge_out = self.edge_out_layers(edge_features)

                if node_out is not None:
                    out_nodes.append(node_out)
                if edge_out is not None:
                    out_edges.append(edge_out)
        
        for i in range(repetitions):
            node_featuresT, edge_indexesT, edge_featuresT, u_featuresT = self.main_layers((node_features, edge_indexes, edge_features, u_features))
            if self.trackAtt:
                for layer in self.main_layers:
                    self.attn.append(layer.mhAtt.attn)
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

            if node_out is not None:
                out_nodes.append(node_out)
            if edge_out is not None:
                out_edges.append(edge_out)

        if len(out_nodes)>0:
            out_nodes = torch.stack(out_nodes,dim=1) #we introduce a 'time' dimension
        else:
            out_nodes = None
        if len(out_edges)>0:
            out_edges = torch.stack(out_edges,dim=1) #we introduce a 'time' dimension
        else:
            out_edges = None

        return out_nodes, out_edges



    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)
