#from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
import math, copy

import torch

from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from .simpleNN import SimpleNN
from .net_builder import getGroupSize


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
        output = torch.spmm(adj[0], support)
        #normalize based on how many things are summed
        output /= adj[1][:,None]
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

    def __init__(self, num_features,norm='group_norm',dropout=0.1, depth=2):
        super(GraphResConv, self).__init__()
        #self.side1=GraphConvWithAct(num_features,num_features,norm,dropout)
        #self.side2=GraphConvWithAct(num_features,num_features,norm,dropout)
        self.sideLayers=nn.ModuleList()
        for i in range(depth):
            sideLayers.append(GraphConvWithAct(num_features,num_features,norm,dropout))
            

    def forward(self,node_features,adjacencyMatrix,numBBs):
        #side = self.side1(node_features,adjacencyMatrix,numBBs)
        #side = self.side2(side,adjacencyMatrix,numBBs)
        side=node_features
        for layer in self.sideLayers:
            side = layer(side,adjacencyMatrix,numBBs)
        return node_features+side

#These are taken from the Annotated Transformer (http://nlp.seas.harvard.edu/2018/04/03/attention.html#attention)
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    ###
    #scores.fill_(0.1)
    ###
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
def learned_attention(query, key, value, mask=None, dropout=None,network=None):
    "Compute Attention using provided network"

    #naive "everywhere" implmenetation
    assert(len(query.size())==4)
    query_ex = query[:,:,:,None,:].expand(-1,-1,query.size(2),key.size(2),-1)
    key_ex = key[:,:,None,:,:].expand(-1,-1,query.size(2),key.size(2),-1)
    comb = torch.cat((query_ex,key_ex),dim=4)
    comb = comb.view(comb.size(0),comb.size(1),-1,comb.size(-1))
    scores = network(comb[:,0])
    #scores = torch.stack(scores,dim=1)
    scores = scores.view(scores.size(0),1,query.size(2),key.size(2))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, mod=None):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) #W_q W_k W_v W_o
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.mod=mod #learned: use network for attention instead of dot product, half: use only half of query/keys for dot product
        if mod=='learned':
            self.attNet = nn.Sequential(
                    #nn.GroupNorm(getGroupSize(self.d_k*2),self.d_k*2),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.d_k*2,self.d_k//4),
                    #nn.GroupNorm(getGroupSize(self.d_k//4),self.d_k//4),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.d_k//4,1) 
                    ) 
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask[None,None,...]#mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        if self.mod=='half':
            x, self.attn = attention(query[...,:self.d_k//2], key[...,:self.d_k//2], value, mask=mask, 
                                     dropout=self.dropout)
        elif self.mod=='learned':
            x, self.attn = learned_attention(query, key, value, mask=mask, 
                                     dropout=self.dropout,network=self.attNet)
        else:
            x, self.attn = attention(query, key, value, mask=mask, 
                                     dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class GraphSelfAttention(nn.Module):
    """
    Graph convolution using self attention across neighbors
    """

    def __init__(self, in_features, heads=8):
        super(GraphSelfAttention, self).__init__()
        self.mhAtt = MultiHeadedAttention(heads,in_features)

    def forward(self, input, adj):
        #construct mask s.t. 1s where edges exist
        #locs = torch.LongTensor(adj).t()
        #ones = torch.ones(len(adj))
        #mask = torch.sparse.FloatTensor(locs,ones,torch.Size([input.size(0),input.size(0)]))
        mask=adj
        input_ = input[None,...] # add batch dim
        return self.mhAtt(input_,input_,input_,mask)

    def __repr__(self):
        return self.__class__.__name__ +'(heads:{})'.format(self.mhAtt.h)

class GraphTransformerBlock(nn.Module):

    def __init__(self, features, num_heads, num_ffnn_layers=2, ffnn_features=None,split=False):
        super(GraphTransformerBlock, self).__init__()
        if ffnn_features is None:
            ffnn_features=features

        config_ffnn = {
                'feat_size': features,
                'num_layers': num_ffnn_layers-1,
                'hidden_size': ffnn_features,
                'out_size': features,
                #'reverse': True,
                'norm': None,
                'dropout': 0.1
                }
        self.ffnn = SimpleNN(config_ffnn)
        self.att = GraphSelfAttention(features,num_heads)
        #self.norm1 = nn.GroupNorm(getGroupSize(features),features)
        #self.norm2 = nn.GroupNorm(getGroupSize(features),features)
        self.norm1 = nn.GroupNorm(1,features)
        self.norm2 = nn.GroupNorm(1,features)

    def forward(self,input,adj=None,numBBs=None):
        if adj is None:
            input,adjMine,numBBs = input
        else:
            adjMine=adj
        #import pdb;pdb.set_trace()
        side1=self.att(input,adjMine)[0]
        side1+=input
        #TODO allow splitting into rel and box sides
        side1 = self.norm1(side1)
        side2=self.ffnn(side1)
        #return self.norm2(side2),adj,numBBs
        if adj is None:
            return self.norm2(side2+side1),adj,numBBs
        else:
            return self.norm2(side2+side1)
