import torch.utils.data
import numpy as np
import math
import torch

def collate(batch):
    assert(len(batch)==1)
    return batch[0]

class RandomMessagesDataset(torch.utils.data.Dataset):

    def __init__(self,config):
        self.max_pairs = 3
        self.max_nodes = 6
        self.message_len = 10
        self.feature_len = self.message_len + 2*self.max_pairs

    def __len__(self):
        return 25

    def __getitem__(self,index):
        num_pairs = np.random.randint(1,self.max_pairs+1)
        num_nodes = np.random.randint(num_pairs*2,self.max_nodes+1)

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

        features = torch.zeros(num_nodes,self.feature_len).byte()
        gt = torch.zeros(num_pairs*2,self.message_len).byte()

        for p in range(num_pairs):
            rand = bin(np.random.randint(1,2**self.message_len))
            message = torch.FloatTensor(self.message_len)
            for i in range(self.message_len):
                if rand[-(i+1)]=='b':
                    break
                message[i] = float(rand[-(i+1)])
            features[2*p+0,p]=1
            features[2*p+0,2*self.max_pairs:]=message
            features[2*p+1,self.max_pairs+p]=1
            gt[2*p+1]=message

        edgeLocs = torch.LongTensor(list(edges)).t()
        ones = torch.ones(len(edges))
        adjacencyMatrix = torch.sparse.FloatTensor(edgeLocs,ones,torch.Size([num_nodes,num_nodes]))

        return features, adjacencyMatrix, gt, num_pairs*2
