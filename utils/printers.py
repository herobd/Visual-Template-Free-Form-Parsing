from skimage import color, io
import os
import numpy as np
import torch
import cv2
from utils import util
from model.alignment_loss import alignment_loss
import math

def AI2D_printer(instance, model, gpu, metrics, outDir=None, startIndex=None):
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
        return metricsOut

    batchSize = data.shape[0]
    for i in range(batchSize):
        image = np.transpose(data[i][0:3,:,:],(1,2,0))
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
        
    return metricsOut

def FormsDetect_printer(instance, model, gpu, metrics, outDir=None, startIndex=None):
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
    #print(instance)
    #data, target, targetSizes = instance
    data = instance['img']
    batchSize = data.shape[0]
    target = instance['sol_eol_gt']
    targetSizes = instance['label_sizes']


    dataT = __to_tensor(data,gpu)
    output = model(dataT)
    index=0
    for name, targ in target.items():
        output[index] = util.pt_xyrs_2_xyxy(output[index])
        index+=1

    alignmentPred={}
    alignmentTarg={}
    loss=0
    index=0
    ttt_hit=None
    #if 22>=startIndex and 22<startIndex+batchSize:
    #    ttt_hit=22-startIndex
    #else:
    #    return 0
    for name,targ in target.items():
        if gpu is not None:
            sendTarg=targ.to(gpu)
        else:
            sendTarg=targ
        lossThis, predIndexes, targetIndexes = alignment_loss(output[index],sendTarg,targetSizes[name],return_alignment=True, debug=ttt_hit)
        alignmentPred[name]=predIndexes
        alignmentTarg[name]=targetIndexes
        index+=1

    data = data.cpu().data.numpy()
    #output = output.cpu().data.numpy()
    outputOld = output
    targetOld = target
    output={}
    target={}
    i=0
    for name,targ in targetOld.items():
        target[name] = targ.data.numpy()
        output[name] = outputOld[i].cpu().data.numpy()
        i+=1
    #metricsOut = __eval_metrics(output,target)
    metricsOut = 0
    if outDir is None:
        return metricsOut
    
    for b in range(batchSize):
        #print('image {} has {} {}'.format(startIndex+b,targetSizes[name][b],name))
        #lineImage = np.ones_like(image)
        for name, out in output.items():
            image = (1-((1+np.transpose(data[b][:,:,:],(1,2,0)))/2.0)).copy()
            #if name=='text_start_gt':
            for j in range(targetSizes[name][b]):
                p1 = (target[name][b,j,0], target[name][b,j,1])
                p2 = (target[name][b,j,2], target[name][b,j,3])
                #mid = ( int(round((p1[0]+p2[0])/2.0)), int(round((p1[1]+p2[1])/2.0)) )
                #rad = round(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/2.0)
                #print(mid)
                #print(rad)
                #cv2.circle(image,mid,rad,(1,0.5,0),1)
                cv2.line(image,p1,p2,(1,0.5,0),1)
            lines=[]
            maxConf = out[b,:,0].max()
            threshConf = maxConf*0.1
            for j in range(out.shape[1]):
                conf = out[b,j,0]
                if conf>threshConf:
                    p1 = (out[b,j,1],out[b,j,2])
                    p2 = (out[b,j,3],out[b,j,4])
                    lines.append((conf,p1,p2,j))
            lines.sort(key=lambda a: a[0]) #so most confident lines are draw last (on top)
            for conf, p1, p2, j in lines:
                shade = 0.0+conf/maxConf
                #print(shade)
                #if name=='text_start_gt' or name=='field_end_gt':
                #    cv2.line(lineImage[:,:,1],p1,p2,shade,2)
                #if name=='text_end_gt':
                #    cv2.line(lineImage[:,:,2],p1,p2,shade,2)
                #elif name=='field_end_gt' or name=='field_start_gt':
                #    cv2.line(lineImage[:,:,0],p1,p2,shade,2)
                if name=='text_start_gt':
                    color=(0,shade,0)
                elif name=='text_end_gt':
                    color=(0,0,shade)
                elif name=='field_end_gt':
                    color=(shade,shade,0)
                elif name=='field_start_gt':
                    color=(shade,0,0)
                cv2.line(image,p1,p2,color,1)
                if j in alignmentPred[name][b]:
                    mid = ( int(round((p1[0]+p2[0])/2.0)), int(round((p1[1]+p2[1])/2.0)) )
                    rad = round(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/2.0)
                    #print(mid)
                    #print(rad)
                    cv2.circle(image,mid,rad,(0,1,1),1)
            #for j in alignmentTarg[name][b]:
            #    p1 = (target[name][b,j,0], target[name][b,j,1])
            #    p2 = (target[name][b,j,0], target[name][b,j,1])
            #    mid = ( int(round((p1[0]+p2[0])/2.0)), int(round((p1[1]+p2[1])/2.0)) )
            #    rad = round(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/2.0)
            #    #print(mid)
            #    #print(rad)
            #    cv2.circle(image,mid,rad,(1,0,1),1)

            saveName = '{:06}_{}'.format(startIndex+b,name)
            #for j in range(metricsOut.shape[1]):
            #    saveName+='_m:{0:.3f}'.format(metricsOut[i,j])
            saveName+='.png'
            io.imsave(os.path.join(outDir,saveName),image)
        
    return metricsOut
