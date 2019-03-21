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

    acc = ((torch.sigmoid(output[:,-1])>0.5)==gt).float().mean().item()

    #print(loss)
    if 'score' not in config:
        display(instance,torch.sigmoid(output[:,-1].cpu()))

    return (
            {'loss':loss, 'acc':acc},
             loss
            )


