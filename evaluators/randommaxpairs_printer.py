import os
import numpy as np
import torch
import cv2
import math
from model.loss import *
from datasets.test_random_maxpairs import display



def RandomMaxPairsDataset_printer(config,instance, model, gpu, metrics, outDir=None, startIndex=None, lossFunc=None):
    def __eval_metrics(data,target):
        acc_metrics = np.zeros((output.shape[0],len(metrics)))
        for ind in range(output.shape[0]):
            for i, metric in enumerate(metrics):
                acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
        return acc_metrics

    def __to_tensor(instance,gpu):
        features, adjaceny, gt = instance

        if gpu is not None:
            features = features.float().to(gpu)
            adjaceny = adjaceny.to(gpu)
            gt = gt.to(gpu)
        else:
            features = features.float()
        return features, adjaceny, gt

    if 'repetitions' in config:
        model.repetitions=config['repetitions']

    
    features,edgeIndices, gt = __to_tensor(instance,gpu)
    if True:
        _,output = model((features,edgeIndices,None,None))
        #print(output[:,0])
        gts=gt[:,None,:].expand(gt.size(0),output.size(1),gt.size(1))
    else:
        output,_ = model(features,(adj,None),num)
    if lossFunc is not None:
        loss = lossFunc(output,gts.float())
        loss = loss.item()
    else:
        loss=0

    if 'avg' in config:
        output = (output[:output.size(0)//2] + output[output.size(0)//2:])/2
        gt = gt[:gt.size(0)//2]
    #acc = ((torch.sigmoid(output[:,-1])>0.5)==gt).float().mean().item()
    #import pdb;pdb.set_trace()
    accAll = ((torch.sigmoid(output)>0.5)==gts).float().mean(dim=0)
    acc = accAll[-1].item()
    accDiff=[]
    ret ={   'loss':loss,
                'acc':acc
                }
    for i in range(1,accAll.size(0)):
        accDiff.append(accAll[i]-accAll[i-1])
        ret['gain [{}] to [{}]'.format(i-1,i)] = accDiff[i-1].item()

    #print(loss)
    if 'score' not in config:
        display(instance,torch.sigmoid(output[:,-1].cpu()))

    return (
            ret,
             loss
            )


