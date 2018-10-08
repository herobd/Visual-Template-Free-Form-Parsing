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


def FormsLF_printer(config,instance, model, gpu, metrics, outDir=None, startIndex=None):
    def _to_tensor( *datas):
        ret=(_to_tensor_individual(datas[0]),)
        for i in range(1,len(datas)):
            ret+=(_to_tensor_individual(datas[i]),)
        return ret
    def _to_tensor_individual( data):
        if type(data)==list:
            return [_to_tensor_individual(d) for d in data]
        if (len(data.size())==1 and data.size(0)==1):
            return data[0]

        if type(data) is np.ndarray:
            data = torch.FloatTensor(data.astype(np.float32))
        elif type(data) is torch.Tensor:
            data = data.type(torch.FloatTensor)
        if gpu is not None:
            data = data.to(gpu)
        return data

    b=0 #assume batchsize of 1

    data, positions_xyxy, positions_xyrs, steps, forwards, detected_end_points = _to_tensor(*instance)
    #print(steps)
    output_xyxy, output_xyrs, output_end = model(
            data,
            positions_xyrs[0],
            forwards,
            steps=steps,
            skip_grid=True,
            detected_end_points=detected_end_points)
    loss = lf_line_loss(output_xyxy, positions_xyxy)
    image = (1-((1+np.transpose(instance[0][b][:,:,:].numpy(),(1,2,0)))/2.0)).copy()
    #print(image.shape)
    #print(type(image))
    minX=minY=9999999
    maxX=maxY=-1

    if outDir is not None:
        for j in range(len(instance[5]).shape[0]):
            conf = instance[5][b,j,0]
            x = int(instance[5][b,j,1])
            y = int(instance[5][b,j,2])

            cv2.circle(image,(x,y),2,(conf,conf,0),-1)

        for pointPair in  instance[1]:
            pointPair=pointPair[b].numpy()
            #print (pointPair)
            xU=int(pointPair[0,0])
            yU=int(pointPair[1,0])
            xL=int(pointPair[0,1])
            yL=int(pointPair[1,1])
            cv2.circle(image,(xU,yU),2,(0.25,1,0),-1)
            cv2.circle(image,(xL,yL),2,(0,1,0.25),-1)
            minX=min(minX,xU,xL)
            maxX=max(maxX,xU,xL)
            minY=min(minY,yU,yL)
            maxY=max(maxY,yU,yL)


        j=0
        for pointPair in output_xyxy:
            pointPair = pointPair[b].data.cpu().numpy()
            xU=int(pointPair[0,0])
            yU=int(pointPair[1,0])
            xL=int(pointPair[0,1])
            yL=int(pointPair[1,1])
            cv2.circle(image,(xU,yU),2,(1,0,0),-1)
            cv2.circle(image,(xL,yL),2,(0,0,1),-1)
            minX=min(minX,xU,xL)
            maxX=max(maxX,xU,xL)
            minY=min(minY,yU,yL)
            maxY=max(maxY,yU,yL)
            
            if j>0:
                endScore = output_end[j-1][b]
                x=(xU+xL)//2
                y=(yU+yL)//2
                cv2.circle(image,(x,y),4,(endScore,0,endScore),-1)

            j+=1

        horzPad = int((maxX-minX)/2)
        vertPad = int((maxY-minY)/2)
        image=image[max(0,minY-vertPad):min(image.shape[0],maxY+vertPad) , max(0,minX-horzPad):min(image.shape[1],maxX+horzPad)]

        saveName = '{:06}_lf_l:{:.3f}.png'.format(startIndex+b,loss.item())
        io.imsave(os.path.join(outDir,saveName),image)

    return {
            "loss":{'xy':[loss.item()]}
            }
