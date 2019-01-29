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
from utils.yolo_tools import computeAP
from model.optimize import optimizeRelationships,optimizeRelationshipsSoft,optimizeRelationshipsBlind
import random

#THRESH=0

def FormsFeaturePair_printer(config,instance, model, gpu, metrics, outDir=None, startIndex=None, lossFunc=None):
    THRESH=0.9
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
    data = instance['data'][0]
    if data.size(0)==0:
        return (
             { 
               'recall':[],
               'prec':[],
             }, 
             (1, 1))
    qXY = instance['qXY']
    iXY = instance['iXY']
    label = instance['label']
    dataT = data.to(gpu)#__to_tensor(data,gpu)
    relNodeIds = instance['nodeIds']
    gtNumNeighbors=instance['numNeighbors']+1

    predAll = model(dataT)
    pred = predAll[:,0]

    if predAll.size(1)==3:
        predNN = predAll[:,1:]+1
        newPredNN=defaultdict(list)
    else:
        predNN=newPredNN = None
    pred = torch.sigmoid(pred)

    

    #merge, as we have mirrored relationships here
    newPred=torch.FloatTensor(pred.size(0)//2).to(pred.device)
    newLabel=torch.FloatTensor(pred.size(0)//2).to(pred.device)
    newData=torch.FloatTensor(pred.size(0)//2,data.size(1))
    newGtNumNeighbors=torch.FloatTensor(pred.size(0)//2,2)
    newNodeIds=[]
    newi=0
    for i in range(len(relNodeIds)):
        id1,id2=relNodeIds[i]
        if id1 is not None:
            j = relNodeIds.index((id2,id1),i+1)
            newPred[newi]=(pred[i]+pred[j])/2 #we average the two predictions
            #newNNPred[newi]=(nnPred[i]+nnPred[j])/2
            newLabel[newi]=label[i]
            newNodeIds.append(relNodeIds[i]) #ensure order is the same
            newData[newi]=data[i]
            if predNN is not None:
                newPredNN[id1]+=[predNN[i,0].item(),predNN[j,1].item()] # (predNN[i,0]+predNN[j,1])/2
                newPredNN[id2]+=[predNN[i,1].item(),predNN[j,0].item()]
            newGtNumNeighbors[newi]=gtNumNeighbors[i]
            relNodeIds[j]=(None,None)
            newi+=1
    pred=newPred
    #nnPred=newNNPred
    relNodeIds=newNodeIds
    label=newLabel
    predNN={}
    for id,li in newPredNN.items():
        predNN[id]=np.mean(li)
    gtNumNeighbors=newGtNumNeighbors
 
    if 'optimize' in config and config['optimize']:
        #We first need to prune down as there are far too many possible pairings
        thresh=0.3
        while thresh>0:
            keep = pred>thresh
            newPred=pred[keep]
            if newPred.size(0)<1500:
                break
            thresh-=0.01
        newIds=[]
        newLabel=label[keep]
        if config['optimize']=='blind':
            idNum=0
            numIds=[]
            idNumMap={}
            for index,(id1,id2) in enumerate(relNodeIds):
                if keep[index]:
                    if id1 not in idNumMap:
                        idNumMap[id1]=idNum
                        idNum+=1
                    if id2 not in idNumMap:
                        idNumMap[id2]=idNum
                        idNum+=1
                    numIds.append( [idNumMap[id1],idNumMap[id2]] )
            print('size being optimized: {}'.format(newPred.size(0)))
            pred[keep] *= torch.from_numpy( optimizeRelationshipsBlind(newPred,numIds) ).float()
        elif predNN is not None and config['optimize']!='gt' and config['optimize']!='gt_noisy':
            idMap={}
            newId=0
            numIds=[]
            numNeighbors=[]
            for index,(id1,id2) in enumerate(newNodeIds):
                if keep[index]:
                    if id1 not in idMap:
                        idMap[id1]=newId
                        numNeighbors.append(predNN[id1])
                        newId+=1
                    if id2 not in idMap:
                        idMap[id2]=newId
                        numNeighbors.append(predNN[id2])
                        newId+=1
                    numIds.append( [idMap[id1],idMap[id2]] )
            assert((newPred.size(0))<1500)
            print('size being optimized soft: {}'.format(newPred.size(0)))
            pred[keep] *= torch.from_numpy( optimizeRelationshipsSoft(newPred,numIds,numNeighbors) ).float()
        else:
            for i in range(keep.size(0)):
                if keep[i]:
                    newIds.append(relNodeIds[i])
            
            numNeighborsD=defaultdict(lambda: 0)
            i=0
            for id1,id2 in newIds:
                if newLabel[i]:
                    numNeighborsD[id1]+=1
                    numNeighborsD[id2]+=1
                else:
                    numNeighborsD[id1]+=0
                    numNeighborsD[id2]+=0
                i+=1
            numNeighbors=[0]*len(numNeighborsD)
            idNum=0
            idNumMap={}
            numIds=[]
            for id,count in numNeighborsD.items():
                if config['optimize']=='gt_noisy':
                    numNeighbors[idNum]=random.gauss(count,0.5)
                else:
                    numNeighbors[idNum]=count
                idNumMap[id]=idNum
                idNum+=1
            numIds = [ [idNumMap[id1],idNumMap[id2]] for id1,id2 in newIds ]
            print('size being optimized: {}'.format(newPred.size(0)))
            if config['optimize']=='gt_noisy':
                pred[keep] *= torch.from_numpy( optimizeRelationshipsSoft(newPred,numIds,numNeighbors) ).float()
            else:
                pred[keep] *= torch.from_numpy( optimizeRelationships(newPred,numIds,numNeighbors) ).float()


        pred[1-keep] *= 0
        THRESH=0
   
    #ossThis, position_loss, conf_loss, class_loss, recall, precision = yolo_loss(outputOffsets,targetBBsT,targetBBsSizes)
    image = cv2.imread(imagePath,1)
    #image[40:50,40:50,0]=255
    #image[50:60,50:60,1]=255
    assert(image.shape[2]==3)
    batchSize = pred.size(0) #data.size(0)
    #draw GT
    for b in range(batchSize):
        x,y = qXY[b]
        r = data[b,2].item()*math.pi
        h = data[b,0].item()*50/2
        w = data[b,1].item()*400/2
        plotRect(image,(0,0,255),(x,y,r,h,w))
        x2,y2 = iXY[b]
        r = data[b,6].item()*math.pi
        h = data[b,4].item()*50/2
        w = data[b,5].item()*400/2
        plotRect(image,(0,0,255),(x2,y2,r,h,w))

        #r = data[b,2].item()
        #h = data[b,0].item()
        #w = data[b,1].item()
        #plotRect(image,(1,0,0),(x,y,r,h,w))

        if label[b].item()> 0:
           cv2.line(image,(int(x),int(y)),(int(x2),int(y2)),(0,255,0),1)


    totalPreds=0
    totalGTs=0
    truePs=0
    scores=[]
    wroteIds=set()
    for b in range(batchSize):
        id1,id2 = relNodeIds[b]
        x,y = qXY[b]
        x2,y2 = iXY[b]
        x=int(x)
        y=int(y)
        x2=int(x2)
        y2=int(y2)
        if (pred[b].item()>THRESH):
            totalPreds+=1
            if label[b].item()>0:
                truePs+=1
            color = int(255*(pred[b].item()-THRESH)/(1-THRESH))
            cv2.line(image,(x,y+3),(x2,y2-3),(color,0,0),1)
        if label[b].item()>0:
            scores.append( (pred[b],True) )
            totalGTs+=1
        else:
            scores.append( (pred[b],False) )
        color = int(min(abs(predNN[id1]-gtNumNeighbors[b,0]),2)*127)
        if id1 not in wroteIds:
            cv2.putText(image,'{:.2f}/{}'.format(predNN[id1],gtNumNeighbors[b,0]),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(color,0,0),2,cv2.LINE_AA)
            wroteIds.add(id1)
        color = int(min(abs(predNN[id2]-gtNumNeighbors[b,1]),2)*127)
        if id2 not in wroteIds:
            cv2.putText(image,'{:.2f}/{}'.format(predNN[id2],gtNumNeighbors[b,1]),(x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(color,0,0),2,cv2.LINE_AA)
            wroteIds.add(id2)
    
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
        #cv2.imshow('dfsdf',image)
        #cv2.waitKey()

    ap=computeAP(scores)

        
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
               'AP':[ap],
             }, 
             (recall, prec,ap)
            )


