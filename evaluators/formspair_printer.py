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
from model import *

detector=None

def FormsPair_printer(config, instance, model, gpu, metrics, outDir=None, startIndex=None):
    global detector
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

    if detector is None:
        checkpoint = torch.load(config["trainer"]['detector_checkpoint'])
        detector_config = config["trainer"]['detector_config'] if 'detector_config' in config else checkpoint['config']['model']
        
        if 'state_dict' in checkpoint:
            detector = eval(checkpoint['config']['arch'])(detector_config)
            detector.load_state_dict(checkpoint['state_dict'])
        else:
            detector = checkpoint['model']
        detector.forPairing=True
        detector = detector.to(gpu)
        detector.eval()

    data, imageName, target = instance
    dataT = __to_tensor(data,gpu)
    padH=(detector.scale-(dataT.size(2)%detector.scale))%detector.scale
    padW=(detector.scale-(dataT.size(3)%detector.scale))%detector.scale
    if padH!=0 or padW!=0:
        padder = torch.nn.ZeroPad2d((0,padW,0,padH))
        dataT = padder(dataT)
    detector(dataT[:,:1])
    final_features=detector.final_features
    output = model(dataT,final_features)
    output = output[...,:target.size(-2),:target.size(-1)]

    data = data.cpu().data.numpy()
    output = output.cpu().data.numpy()
    target = target.data.numpy()
    metricsOut = __eval_metrics(output,target)
    if outDir is None:
        return {'map':metricsOut[0]}, 0

    batchSize = data.shape[0]
    for i in range(batchSize):
        image = (1-np.transpose(data[i][0:3,:,:],(1,2,0)))/2.0
        #image = cv2.resize(image,(target.size(-1),target.size(-2)))
        queryMask = data[i][1,:,:]
        #queryMask = cv2.resize(image,(target.size(-1),target.size(-2)))

        grayIm = color.rgb2grey(image)

        invQuery = 1-queryMask
        invTarget = 1-target[i]
        invOutput = output[i]<=0.0 #assume not sigmoided
        invTarget = cv2.resize(invTarget.astype(np.float),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
        invOutput = cv2.resize(invOutput.astype(np.float),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)


        highlightIm = np.stack([grayIm*invOutput, grayIm*invTarget, grayIm*invQuery],axis=2)

        saveName = '{:06}_{}'.format(startIndex+i,imageName[i])
        for j in range(metricsOut.shape[1]):
            saveName+='_m:{0:.3f}'.format(metricsOut[i,j])
        saveName+='.png'
        io.imsave(os.path.join(outDir,saveName),highlightIm)
        
    return {'map':metricsOut[0]}, 0
