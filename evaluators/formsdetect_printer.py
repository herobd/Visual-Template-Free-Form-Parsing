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


def FormsDetect_printer(config,instance, model, gpu, metrics, outDir=None, startIndex=None):
    def __eval_metrics(data,target):
        acc_metrics = np.zeros((output.shape[0],len(metrics)))
        for ind in range(output.shape[0]):
            for i, metric in enumerate(metrics):
                acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
        return acc_metrics

    def __to_tensor_old(data, gpu):
        if type(data) is np.ndarray:
            data = torch.FloatTensor(data.astype(np.float32))
        elif type(data) is torch.Tensor:
            data = data.type(torch.FloatTensor)
        if gpu is not None:
            data = data.to(gpu)
        return data
    def __to_tensor(instance,gpu):
        data = instance['img']
        if 'line_gt' in instance:
            targetLines = instance['line_gt']
            targetLines_sizes = instance['line_label_sizes']
        else:
            targetLines = {}
            targetLines_sizes = {}
        if 'point_gt' in instance:
            targetPoints = instance['point_gt']
            targetPoints_sizes = instance['point_label_sizes']
        else:       
            targetPoints = {}
            targetPoints_sizes = {}
        if 'pixel_gt' in instance:
            targetPixels = instance['pixel_gt']
        else:
            targetPixels = None
        if type(data) is np.ndarray:
            data = torch.FloatTensor(data.astype(np.float32))
        elif type(data) is torch.Tensor:
            data = data.type(torch.FloatTensor)
                    
        def sendToGPU(targets):
            new_targets={}
            for name, target in targets.items():
                if target is not None:
                    new_targets[name] = target.to(gpu)
                else:
                    new_targets[name] = None
            return new_targets
            
        if gpu is not None:
            data = data.to(gpu)
            targetLines=sendToGPU(targetLines)
            targetPoints=sendToGPU(targetPoints)
            if targetPixels is not None:
                targetPixels=targetPixels.to(gpu)
        return data, targetLines, targetLines_sizes, targetPoints, targetPoints_sizes, targetPixels


    #print(type(instance['pixel_gt']))
    #if type(instance['pixel_gt']) == list:
    #    print(instance)
    #    print(startIndex)
    #data, targetLine, targetLineSizes = instance
    data = instance['img']
    batchSize = data.shape[0]
    targetLines = instance['line_gt']
    targetPoints = instance['point_gt']
    targetPixels = instance['pixel_gt']
    imageName = instance['imgName']
    scale = instance['scale']
    dataT, targetLinesT, targetLinesSizes, targetPointsT, targetPointsSizes, targetPixelsT = __to_tensor(instance,gpu)

    resultsDirName='results'
    if outDir is not None and resultsDirName is not None:
        rPath = os.path.join(outDir,resultsDirName)
        if not os.path.exists(rPath):
            os.mkdir(rPath)
        for name in targetLines:
            nPath = os.path.join(rPath,name)
            if not os.path.exists(nPath):
                os.mkdir(nPath)

    #dataT = __to_tensor(data,gpu)
    print('{}: {} x {}'.format(imageName,data.shape[2],data.shape[3]))
    outputLines_xyrs, outputPoints, outputPixels = model(dataT)
    outputPixels = torch.sigmoid(outputPixels)
    index=0
    outputLines = []
    for name, targ in targetLines.items():
        outputLines.append(util.pt_xyrs_2_xyxy(outputLines_xyrs[index]))
        index+=1

    alignmentLinesPred={}
    alignmentLinesTarg={}
    loss=0
    index=0
    ttt_hit=True
    #if 22>=startIndex and 22<startIndex+batchSize:
    #    ttt_hit=22-startIndex
    #else:
    #    return 0
    for name,targ in targetLinesT.items():
        #if gpu is not None:
        #    sendTarg=targ.to(gpu)
        #else:
        #    sendTarg=targ
        lossThis, predIndexes, targetLinesIndexes = alignment_loss(outputLines[index],targ,targetLinesSizes[name],**config['loss_params']['line'],return_alignment=True, debug=ttt_hit)
        alignmentLinesPred[name]=predIndexes
        alignmentLinesTarg[name]=targetLinesIndexes
        index+=1
    alignmentPointsPred={}
    alignmentPointsTarg={}
    index=0
    for name,targ in targetPointsT.items():
        #print(outputPoints[0].shape)
        #print(targetPointsSizes)
        #print('{} {}'.format(index, name))
        lossThis, predIndexes, targetPointsIndexes = alignment_loss(outputPoints[index],targ,targetPointsSizes[name],**config['loss_params']['point'],return_alignment=True, debug=ttt_hit, points=True)
        alignmentPointsPred[name]=predIndexes
        alignmentPointsTarg[name]=targetPointsIndexes
        index+=1

    data = data.cpu().data.numpy()
    #outputLine = outputLine.cpu().data.numpy()
    outputLinesOld = outputLines
    outputLinesOld_xyrs = outputLines_xyrs
    targetLinesOld = targetLines
    outputLines={}
    outputLines_xyrs={}
    targetLines={}
    i=0
    for name,targ in targetLinesOld.items():
        if targ is not None:
            targetLines[name] = targ.data.numpy()
        else:
             targetLines[name]=None
        outputLines[name] = outputLinesOld[i].cpu().data.numpy()
        outputLines_xyrs[name] = outputLinesOld_xyrs[i].cpu().data.numpy()
        i+=1
    outputPointsOld = outputPoints
    targetPointsOld = targetPoints
    outputPoints={}
    targetPoints={}
    i=0
    for name,targ in targetPointsOld.items():
        if targ is not None:
            targetPoints[name] = targ.data.numpy()
        else:
            targetPoints[name]=None
        outputPoints[name] = outputPointsOld[i].cpu().data.numpy()
        i+=1
    outputPixels = outputPixels.cpu().data.numpy()
    #metricsOut = __eval_metrics(outputLines,targetLines)
    #metricsOut = 0
    #if outDir is None:
    #    return metricsOut
    
    dists=defaultdict(list)
    dists_x=defaultdict(list)
    dists_y=defaultdict(list)
    scaleDiffs=defaultdict(list)
    rotDiffs=defaultdict(list)
    for b in range(batchSize):
        #print('image {} has {} {}'.format(startIndex+b,targetLinesSizes[name][b],name))
        #lineImage = np.ones_like(image)
        for name, out in outputLines.items():
            if outDir is not None:
                #Write the results so we can train LF with them
                saveFile = os.path.join(outDir,resultsDirName,name,'{}'.format(imageName[b]))
                #we must rescale the output to be according to the original image
                rescaled_outputLines_xyrs = outputLines_xyrs[name][b]
                rescaled_outputLines_xyrs[:,1] /= scale[b]
                rescaled_outputLines_xyrs[:,2] /= scale[b]
                rescaled_outputLines_xyrs[:,4] /= scale[b]

                np.save(saveFile,rescaled_outputLines_xyrs)

                image = (1-((1+np.transpose(data[b][:,:,:],(1,2,0)))/2.0)).copy()
                #if name=='text_start_gt':
                for j in range(targetLinesSizes[name][b]):
                    p1 = (targetLines[name][b,j,0], targetLines[name][b,j,1])
                    p2 = (targetLines[name][b,j,2], targetLines[name][b,j,3])
                    #mid = ( int(round((p1[0]+p2[0])/2.0)), int(round((p1[1]+p2[1])/2.0)) )
                    #rad = round(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/2.0)
                    #print(mid)
                    #print(rad)
                    #cv2.circle(image,mid,rad,(1,0.5,0),1)
                    #print(p1)
                    #print(p2)
                    cv2.line(image,p1,p2,(1,0.5,0),1)
            lines=[]
            #pred_points=[]
            maxConf = out[b,:,0].max()
            threshConf = maxConf*0.1
            for j in range(out.shape[1]):
                conf = out[b,j,0]
                if conf>threshConf:
                    p1 = (out[b,j,1],out[b,j,2])
                    p2 = (out[b,j,3],out[b,j,4])
                    lines.append((conf,p1,p2,j))
                #pred_points.append(
            lines.sort(key=lambda a: a[0]) #so most confident lines are draw last (on top)
            for conf, p1, p2, j in lines:
                #circle aligned predictions
                if alignmentLinesPred[name] is not None and j in alignmentLinesPred[name][b]:
                    alignmentIndex = np.where(alignmentLinesPred[name][b]==j)[0][0]
                    gtLine = targetLines[name][b,alignmentLinesTarg[name][b][alignmentIndex]]
                    dx = gtLine[0]-gtLine[2]
                    dy = gtLine[1]-gtLine[3]
                    scale_targ = math.sqrt(dx**2 + dy**2)
                    mx_targ = (gtLine[0]+gtLine[2])/2.0
                    my_targ = (gtLine[1]+gtLine[3])/2.0
                    theta_targ = -math.atan2(dx, -dy)
                    dx = p1[0]-p2[0]
                    dy = p1[1]-p2[1]
                    scale_pred = math.sqrt(dx**2 + dy**2)
                    mx_pred = (p1[0]+p2[0])/2.0
                    my_pred = (p1[1]+p2[1])/2.0
                    theta_pred = -math.atan2(dx, -dy)
                    dists[name].append( math.sqrt((mx_targ-mx_pred)**2 + (my_targ-my_pred)**2) )
                    dists_x[name].append(mx_targ-mx_pred)
                    dists_y[name].append(my_targ-my_pred)
                    scaleDiffs[name].append( scale_targ-scale_pred )
                    rotDiffs[name].append( theta_targ-theta_pred )
                    #mid = ( int(round((p1[0]+p2[0])/2.0)), int(round((p1[1]+p2[1])/2.0)) )
                    #rad = round(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/2.0)
                    #print(mid)
                    #print(rad)
                    #cv2.circle(image,mid,rad,(0,1,1),1)
                if outDir is not None:
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
                    else:
                        color=(shade,0,0)
                    cv2.line(image,p1,p2,color,1)

            if outDir is not None:
                #for j in alignmentLinesTarg[name][b]:
                #    p1 = (targetLines[name][b,j,0], targetLines[name][b,j,1])
                #    p2 = (targetLines[name][b,j,0], targetLines[name][b,j,1])
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

        if outDir is not None:
            for name, out in outputPoints.items():
                image = (1-((1+np.transpose(data[b][:,:,:],(1,2,0)))/2.0)).copy()
                #if name=='text_start_gt':
                for j in range(targetPointsSizes[name][b]):
                    p1 = (targetPoints[name][b,j,0], targetPoints[name][b,j,1])
                    cv2.circle(image,p1,2,(1,0.5,0),-1)
                points=[]
                maxConf = max(out[b,:,0].max(),1.0)
                threshConf = maxConf*0.1
                for j in range(out.shape[1]):
                    conf = out[b,j,0]
                    if conf>threshConf:
                        p1 = (out[b,j,1],out[b,j,2])
                        points.append((conf,p1,j))
                points.sort(key=lambda a: a[0]) #so most confident lines are draw last (on top)
                for conf, p1, j in points:
                    shade = 0.0+conf/maxConf
                    if name=='table_points':
                        color=(0,0,shade)
                    else:
                        color=(shade,0,0)
                    cv2.circle(image,p1,2,color,-1)
                    if alignmentPointsPred[name] is not None and j in alignmentPointsPred[name][b]:
                        mid = p1 #( int(round((p1[0]+p2[0])/2.0)), int(round((p1[1]+p2[1])/2.0)) )
                        rad = 4 #round(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/2.0)
                        #print(mid)
                        #print(rad)
                        #cv2.circle(image,mid,rad,(0,1,1),1)
                #for j in alignmentLinesTarg[name][b]:
                #    p1 = (targetLines[name][b,j,0], targetLines[name][b,j,1])
                #    p2 = (targetLines[name][b,j,0], targetLines[name][b,j,1])
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

            image = (1-((1+np.transpose(data[b][:,:,:],(1,2,0)))/2.0)).copy()
            for ch in range(outputPixels.shape[1]):
                image[:,:,ch] = 1-outputPixels[b,ch,:,:]
            saveName = '{:06}_pixels.png'.format(startIndex+b,name)
            io.imsave(os.path.join(outDir,saveName),image)
            #print('finished writing {}'.format(startIndex+b))
        
    #return metricsOut
    return { 'dists':dists,
             'dists_x':dists_x,
             'dists_y':dists_y,
             'scaleDiffs':scaleDiffs,
             'rotDiffs':rotDiffs
             }


