import torch.utils.data
import numpy as np
import math
import torch

def collate(batch):
    assert(len(batch)==1)
    return batch[0]

class RandomDiffusionDataset(torch.utils.data.Dataset):

    def __init__(self,config):
        self.max_on = 4
        self.chs=6
        self.max_nodes = 30
        self.with_sum=False

    def __len__(self):
        return 25

    def __getitem__(self,index):
        num_on = np.random.randint(2,self.max_on+1)
        num_nodes = np.random.randint(num_on+1,self.max_nodes+1)

        edges=set()
        visited=set()
        unvisited=set(range(num_nodes))
        ends = []
        curNode=nextNode = 0
        visited.add(0)
        unvisited.remove(0)
        while len(unvisited)>0:
            if len(ends)>0 and np.random.rand()<0.5:
                curNode = ends.pop()
            while nextNode==curNode:
                if np.random.rand()<0.7:
                    nextNode = np.random.choice(list(unvisited))
                    visited.add(nextNode)
                    unvisited.remove(nextNode)
                else:
                    nextNode = np.random.choice(list(visited))
                
            edges.add((curNode,nextNode))
            edges.add((nextNode,curNode))

            if np.random.rand()<0.2:
                ends.append(nextNode)
            else:
                curNode=nextNode

        features = torch.zeros(num_nodes,self.chs).float()
        gt = torch.zeros(num_nodes,1).byte()

        for p in range(num_on):
            num_ch = np.random.randint(1,self.chs-1)
            chs = np.random.choice(list(range(self.chs)),num_ch,False)
            self.diffuse(features,edges,p,4,chs)
            gt[p]=1

        edgeLocs = torch.LongTensor(list(edges)).t()
        ones = torch.ones(len(edges))
        adjacencyMatrix = torch.sparse.FloatTensor(edgeLocs,ones,torch.Size([num_nodes,num_nodes]))

        return features, adjacencyMatrix, gt, num_nodes

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

