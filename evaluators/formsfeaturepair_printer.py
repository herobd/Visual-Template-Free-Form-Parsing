from skimage import color, io
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from utils import util
from model.alignment_loss import alignment_loss
import math
from model.loss import *
from collections import defaultdict
from utils.yolo_tools import non_max_sup_iou, AP_iou

#THRESH=0
THRESH=0.9

def FormsFeaturePair_printer(config,instance, model, gpu, metrics, outDir=None, startIndex=None, lossFunc=None):
    def plotRect(img,color,xyrhw):
        xc=xyrhw[0]
        yc=xyrhw[1]
        rot=xyrhw[2]
        h=xyrhw[3]
        w=xyrhw[4]
        h = min(30000,h)
        w = min(30000,w)
        tr = ( int(w*math.cos(rot)-h*math.sin(rot) + xc),  int(-w*math.sin(rot)-h*math.cos(rot) + yc) )
        tl = ( int(-w*math.cos(rot)-h*math.sin(rot) + xc), int(w*math.sin(rot)-h*math.cos(rot) + yc) )
        br = ( int(w*math.cos(rot)+h*math.sin(rot) + xc),  int(-w*math.sin(rot)+h*math.cos(rot) + yc) )
        bl = ( int(-w*math.cos(rot)+h*math.sin(rot) + xc), int(w*math.sin(rot)+h*math.cos(rot) + yc) )

        cv2.line(img,tl,tr,color,1)
        cv2.line(img,tr,br,color,1)
        cv2.line(img,br,bl,color,1)
        cv2.line(img,bl,tl,color,1)
    def __eval_metrics(data,target):
        acc_metrics = np.zeros((output.shape[0],len(metrics)))
        for ind in range(output.shape[0]):
            for i, metric in enumerate(metrics):
                acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
        return acc_metrics

    #totensor

    #print(type(instance['pixel_gt']))
    #if type(instance['pixel_gt']) == list:
    #    print(instance)
    #    print(startIndex)
    #data, targetBB, targetBBSizes = instance
    imageName = instance['imgName']
    imagePath = instance['imgPath']
    data = instance['data']
    qXY = instance['qXY']
    iXY = instance['iXY']
    label = instance['label']
    dataT = data.to(gpu)#__to_tensor(data,gpu)

    pred = model(dataT)
    pred = F.sigmoid(pred)
    
    #ossThis, position_loss, conf_loss, class_loss, recall, precision = yolo_loss(outputOffsets,targetBBsT,targetBBsSizes)
    image = cv2.imread(imagePath,1)
    assert(image.shape[2]==3)
    batchSize = data.size(0)
    #draw GT
    for b in range(batchSize):
        x,y = qXY[b]
        r = data[b,2].item()
        h = data[b,0].item()
        w = data[b,1].item()
        plotRect(image,(0,0,1),(x,y,r,h,w))

        x2,y2 = iXY[b]
        #r = data[b,2].item()
        #h = data[b,0].item()
        #w = data[b,1].item()
        #plotRect(image,(1,0,0),(x,y,r,h,w))

        #if label[b].item()> 0:
        #    cv2.line(image,(int(x),int(y)),(int(x2),int(y2)),(0,1,0),1)


    totalPreds=0
    totalGTs=0
    truePs=0
    for b in range(batchSize):
        x,y = qXY[b]
        x2,y2 = iXY[b]
        if (pred[b].item()>THRESH):
            totalPreds+=1
            if label[b].item()>0:
                truePs+=1
            cv2.line(image,(int(x),int(y+3)),(int(x2),int(y2-3)),(1,0,0),1)
        if label[b].item()>0:
            totalGTs+=1
    
    if totalGTs>0:
        recall = truePs/float(totalGTs)
    else:
        recall = 1
    if totalPreds>0:
        prec = truePs/float(totalPreds)
    else:
        prec = 1
    if outDir is not None:
        saveName = '{}_r:{:.4f}_p:{:.4f}_.png'.format(imageName,recall,prec)
        cv2.imwrite(os.path.join(outDir,saveName),image)
        cv2.imshow('dfsdf',image)
        cv2.waitkey()

        
    #return metricsOut
    return (
             #{ 'ap_5':np.array(aps_5).sum(axis=0),
             #  'ap_3':np.array(aps_3).sum(axis=0),
             #  'ap_7':np.array(aps_7).sum(axis=0),
             #  'recall':np.array(recalls_5).sum(axis=0),
             #  'prec':np.array(precs_5).sum(axis=0),
             #}, 
             { 
               'recall':[recall],
               'prec':[prec],
             }, 
             (recall, prec)
            )


