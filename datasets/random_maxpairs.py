import torch.utils.data
import numpy as np
import math
import torch

def collate(batch):
    assert(len(batch)==1)
    return batch[0]

class RandomMaxPairsDataset(torch.utils.data.Dataset):

    def __init__(self,config):
        self.max_nodes = config['max_nodes'] if 'max_nodes' in config else 7
        self.max_edges = 25

    def __len__(self):
        return 25

    def __getitem__(self,index):
        num_nodes = np.random.randint(3,self.max_nodes+1)

        edges=set()
        visited=set()
        unvisited=set(range(num_nodes))

        #add edges
        ends = []
        curNode=nextNode = 0
        visited.add(0)
        unvisited.remove(0)
        while len(unvisited)>0 and len(edges)<self.max_edges:
            if len(ends)>0 and np.random.rand()<0.5:
                curNode = ends.pop()
            while nextNode==curNode:
                if np.random.rand()<0.4:
                    nextNode = np.random.choice(list(unvisited))
                    visited.add(nextNode)
                    unvisited.remove(nextNode)
                else:
                    nextNode = np.random.choice(list(visited))
                
            edges.add((min(curNode,nextNode),max(curNode,nextNode)))
            #edges.add((nextNode,curNode))

            if np.random.rand()<0.2:
                ends.append(nextNode)
            else:
                curNode=nextNode

        node_values = torch.FloatTensor(num_nodes,1).uniform_()
        
        edges=list(edges)
        edgesT = torch.LongTensor(edges)
        first,second = edgesT.t()
        edge_values = node_values[first] * node_values[second]
    

        #all_configs=?
        #all_edges = edgesT[all_configs]

        #max_value = edge_values(viable_configs).max()

        max_value=0
        for setting in range(1,2**len(edges)):
            #print('{}, {} / {}'.format(num_nodes,setting,2**len(edges)))
            #check if valid setting
            selectedEdge = [bool((2**i) & setting) for i in range(0,len(edges))]
            selectedEdge = torch.ByteTensor(selectedEdge)
            value = edge_values[selectedEdge].sum()
            if value<=max_value:
                continue
            sel_edges = edgesT[selectedEdge]
            first,second = sel_edges.t()

            valid=True
            for node in range(num_nodes):
                l_first = first==node
                c_first = (l_first).sum()
                l_second = second==node
                c_second = (l_second).sum()
                #if c_first!=c_second or c_first>1 or (l_first.any() and sel_edges[l_first][:,1]!=sel_edges[l_second][:,0]):
                #    valid=False
                #    break
                if c_first+c_second>1:
                    valid=False
                    break
            if valid:
                max_value=value
                gt=selectedEdge

        #gt=gt.float()
        gt = gt.repeat(2)
        gt=gt[:,None]

        edges+=[(y,x) for (x,y) in edges]
        edgesT = torch.LongTensor(edges)
        edgeLocs = edgesT.t()
        #ones = torch.ones(len(edges))
        #adjacencyMatrix = torch.sparse.FloatTensor(edgeLocs,ones,torch.Size([num_nodes,num_nodes]))

        return node_values, edgeLocs, gt

    def diffuse(self,features,edges,node,depth,chs,visited=set()):
        rmax = 1 - (4-depth)*0.2
        rmin = max(.4 - (4-depth)*0.2,0)
        for ch in chs:
            if self.with_sum:
                features[node,ch]+=np.random.uniform(rmin,rmax)
            else:
                features[node,ch] = max(features[node,ch],np.random.uniform(rmin,rmax))
        visited.add(node)
        if depth>1:
            for n1,n2 in edges:
                if n1==node and n2 not in visited:
                    self.diffuse(features,edges,n2,depth-1,chs,set(visited))

