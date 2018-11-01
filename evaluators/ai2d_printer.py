from skimage import color, io
import os
import numpy as np
import torch
import cv2
from utils import util
from model.alignment_loss import alignment_loss
import math
from model.loss import *
from collections import defaultdict

def AI2D_printer(config, instance, model, gpu, metrics, outDir=None, startIndex=None):
    #for key, value in metrics.items():
    #    print(key+': '+value)
    def __eval_metrics(data,target):
        acc_metrics = np.zeros((output.shape[0],len(metrics)))
        for ind in range(output.shape[0]):
            for i, metric in enumerate(metrics):
                acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
        return acc_metrics

    def __to_tensor(data, gpu):
        if type(data) is np.ndarray:
            data = torch.FloatTensor(data.astype(np.float32))
        elif type(data) is torch.Tensor:
            data = data.type(torch.FloatTensor)
        if gpu is not None:
            data = data.to(gpu)
        return data

    data, target = instance
    dataT = __to_tensor(data,gpu)
    output = model(dataT)

    data = data.cpu().data.numpy()
    output = output.cpu().data.numpy()
    target = target.data.numpy()
    metricsOut = __eval_metrics(output,target)
    if outDir is None:
        return {'map':metricsOut[0]}, 0

    batchSize = data.shape[0]
    for i in range(batchSize):
        image = (1-np.transpose(data[i][0:3,:,:],(1,2,0)))/2.0
        queryMask = data[i][3,:,:]

        grayIm = color.rgb2grey(image)

        invQuery = 1-queryMask
        invTarget = 1-target[i]
        invOutput = output[i]<=0.0 #assume not sigmoided


        highlightIm = np.stack([grayIm*invOutput, grayIm*invTarget, grayIm*invQuery],axis=2)

        saveName = '{:06}'.format(startIndex+i)
        for j in range(metricsOut.shape[1]):
            saveName+='_m:{0:.3f}'.format(metricsOut[i,j])
        saveName+='.png'
        io.imsave(os.path.join(outDir,saveName),highlightIm)
        
    return {'map':metricsOut[0]}, 0
