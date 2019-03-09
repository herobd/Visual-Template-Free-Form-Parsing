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
from utils.yolo_tools import non_max_sup_iou, AP_iou


def FormsBoxPair_printer(config,instance, model, gpu, metrics, outDir=None, startIndex=None, lossFunc=None):
    def plotRect(img,color,xyrhw):
        xc=xyrhw[0].item()
        yc=xyrhw[1].item()
        rot=xyrhw[2].item()
        h=xyrhw[3].item()
        w=xyrhw[4].item()
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

    def __to_tensor(instance,gpu):
        data = instance['img']
        if 'responseBBs' in instance:
            targetBoxes = instance['responseBBs']
            targetBoxes_sizes = instance['responseBB_sizes']
        else:
            targetBoxes = None
            targetBoxes_sizes = []
        queryMask = instance['queryMask']
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
            queryMask = queryMask.to(gpu)
            if targetBoxes is not None:
                targetBoxes=targetBoxes.to(gpu)
        return data, queryMask, targetBoxes, targetBoxes_sizes


    THRESH = config['THRESH'] if 'THRESH' in config else 0.92
    #print(type(instance['pixel_gt']))
    #if type(instance['pixel_gt']) == list:
    #    print(instance)
    #    print(startIndex)
    #data, targetBB, targetBBSizes = instance
    if lossFunc is None:
        yolo_loss = YoloLoss(model.numBBTypes,model.rotation,model.scale,model.anchors,**config['loss_params'])
    else:
        yolo_loss = lossFunc
    image = instance['img']
    queryMask = instance['queryMask']
    batchSize = image.shape[0]
    targetBBs = instance['responseBBs']
    imageName = instance['imgName']
    if image.size(2)==image.size(3) and image.size(2)==1024:
        imageNameP=None
    else:
        imageNameP=imageName
    scale = instance['scale']
    imageT, queryMaskT, targetBBsT, targetBBsSizes = __to_tensor(instance,gpu)

    resultsDirName='results'
    #if outDir is not None and resultsDirName is not None:
        #rPath = os.path.join(outDir,resultsDirName)
        #if not os.path.exists(rPath):
        #    os.mkdir(rPath)
        #for name in targetBBs:
        #    nPath = os.path.join(rPath,name)
        #    if not os.path.exists(nPath):
        #        os.mkdir(nPath)

    #dataT = __to_tensor(data,gpu)
    #print('{}: {} x {}'.format(imageName,data.shape[2],data.shape[3]))
    from_gt = config['trainer']['from_gt'] if 'from_gt' in config['trainer'] else False
    if from_gt:
        outputBBs, outputOffsets = model(imageT,queryMaskT,
                imageName,
                scale=instance['scale'],
                cropPoint=instance['cropPoint'])
    else:
        outputBBs, outputOffsets = model(imageT,queryMaskT,imageNameP)
    
    index=0
    loss=0
    index=0
    ttt_hit=True
    #if 22>=startIndex and 22<startIndex+batchSize:
    #    ttt_hit=22-startIndex
    #else:
    #    return 0
    lossThis, position_loss, conf_loss, class_loss, recall, precision = yolo_loss(outputOffsets,targetBBsT,targetBBsSizes)

    bestConf=[]
    secondConf=[]
    for b in range(batchSize):
        c,i = torch.max(outputBBs[b,:,0],dim=0)
        bestConf.append( c )
        temp = outputBBs[b,i,0]
        outputBBs[b,i,0] = -99999
        secondConf.append( torch.max(outputBBs[b,:,0]) )
        outputBBs[b,i,0]=temp
    image = image.cpu().data.numpy()
    maxConf = outputBBs[:,:,0].max().item()
    threshConf = max(maxConf*THRESH,0.5)
    #print("maxConf:{}, threshConf:{}".format(maxConf,threshConf))
    if model.rotation:
        outputBBs = non_max_sup_dist(outputBBs.cpu(),threshConf,0.4)
    else:
        outputBBs = non_max_sup_iou(outputBBs.cpu(),threshConf,0.4)
    #aps_3=[]
    aps_5=[]
    class_aps=[]
    apsDiff=[]
    apsSame=[]
    #aps_7=[]
    recalls_5=[]
    recallsDiff=[]
    recallsSame=[]
    precs_5=[]
    precsDiff=[]
    precsSame=[]
    bestBBIdx=[]
    secondBBIdx=[]
    for b in range(batchSize):
        if targetBBs is not None:
            target_for_b = targetBBs[b,:targetBBsSizes[b],:]
        else:
            target_for_b = torch.empty(0)
        if model.rotation:
            ap_5, prec_5, recall_5, class_ap =AP_dist(target_for_b,outputBBs[b],0.9,ignoreClasses=False, getClassAP=True)
            #apCls_5, precCls_5, recallCls_5 =AP_dist(target_for_b,outputBBs[b],0.9,ignoreClasses=False)
        else:
            ap_5, prec_5, recall_5, class_ap =AP_iou(target_for_b,outputBBs[b],0.5,ignoreClasses=False, getClassAP=True)
            #apCls_5, precCls_5, recallCls_5 =AP_iou(target_for_b,outputBBs[b],0.5,ignoreClasses=False)
        #ap_3, prec_3, recall_3 =AP_iou(target_for_b,outputBBs[b],0.3,model.numBBTypes)
        #ap_7, prec_7, recall_7 =AP_iou(target_for_b,outputBBs[b],0.7,model.numBBTypes)

        aps_5.append(ap_5 )
        class_aps.append(class_ap)
        #aps_3.append(ap_3 )
        #aps_7.append(ap_7 )
        recalls_5.append(recall_5)
        precs_5.append(prec_5)
        for classIdx in range(len(apCls_5)):
            #print('{} == {}'.format(classIdx,instance['queryClass']))
            if classIdx == instance['queryClass'][b]:
                apsSame.append(apCls_5[classIdx])
                precsSame.append(precCls_5[classIdx])
                recallsSame.append(recallCls_5[classIdx])
            else:
                apsDiff.append(apCls_5[classIdx])
                precsDiff.append(precCls_5[classIdx])
                recallsDiff.append(recallCls_5[classIdx])
        #for b in range(len(outputBBs)):
        outputBBs[b] = outputBBs[b].data.numpy()
        if outputBBs[b].shape[0]>0:
            bestBBIdx.append( np.argmax(outputBBs[b][:,0]) )
            if outputBBs[b].shape[0]>1:
                temp = outputBBs[b][bestBBIdx[-1],0]
                outputBBs[b][bestBBIdx[-1],0] = -999999
                secondBBIdx.append( np.argmax(outputBBs[b][:,0]) )
                outputBBs[b][bestBBIdx[-1],0] = temp
            else:
                secondBBIdx.append(-1)
        else:
            bestBBIdx.append(-1)
            secondBBIdx.append(-1)
        #import pdb; pdb.set_trace()
    
    dists=defaultdict(list)
    dists_x=defaultdict(list)
    dists_y=defaultdict(list)
    scaleDiffs=defaultdict(list)
    rotDiffs=defaultdict(list)
    for b in range(batchSize):
        #print('image {} has {} {}'.format(startIndex+b,targetBBsSizes[name][b],name))
        #bbImage = np.ones_like(image)
        if outDir is not None:
            #Write the results so we can train LF with them
            #saveFile = os.path.join(outDir,resultsDirName,name,'{}'.format(imageName[b]))
            #we must rescale the output to be according to the original image
            #rescaled_outputBBs_xyrs = outputBBs_xyrs[name][b]
            #rescaled_outputBBs_xyrs[:,1] /= scale[b]
            #rescaled_outputBBs_xyrs[:,2] /= scale[b]
            #rescaled_outputBBs_xyrs[:,4] /= scale[b]

            #np.save(saveFile,rescaled_outputBBs_xyrs)
            imageB = (1-((1+np.transpose(image[b][:,:,:],(1,2,0)))/2.0)).copy()
            if imageB.shape[2]==1:
                imageB = cv2.cvtColor(imageB,cv2.COLOR_GRAY2RGB)
            imageB[:,:,1] *= 1-queryMask[b,0]
            if queryMask.shape[1]>1:
                imageB[:,:,2] *= (1+queryMask[b,1])/2
            #if name=='text_start_gt':

            for j in range(targetBBsSizes[b]):
                plotRect(imageB,(0,1,0),targetBBs[b,j,0:5])
                #if alignmentBBs[b] is not None:
                #    aj=alignmentBBs[b][j]
                #    xc_gt = targetBBs[b,j,0]
                #    yc_gt = targetBBs[b,j,1]
                #    xc=outputBBs[b,aj,1]
                #    yc=outputBBs[b,aj,2]
                #    cv2.line(imageB,(xc,yc),(xc_gt,yc_gt),(0,1,0),1)
                #    shade = 0.0+(outputBBs[b,aj,0]-threshConf)/(maxConf-threshConf)
                #    shade = max(0,shade)
                #    if outputBBs[b,aj,6] > outputBBs[b,aj,7]:
                #        color=(0,shade,shade) #text
                #    else:
                #        color=(shade,shade,0) #field
                #    plotRect(imageB,color,outputBBs[b,aj,1:6])

            #bbs=[]
            #pred_points=[]
            #maxConf = outputBBs[b,:,0].max()
            #threshConf = 0.5 
            #threshConf = max(maxConf*0.9,0.5)
            #print("threshConf:{}".format(threshConf))
            #for j in range(outputBBs.shape[1]):
            #    conf = outputBBs[b,j,0]
            #    if conf>threshConf:
            #        bbs.append((conf,j))
            #    #pred_points.append(
            #bbs.sort(key=lambda a: a[0]) #so most confident bbs are draw last (on top)
            #import pdb; pdb.set_trace()
            if outDir is not None:
                bbs = outputBBs[b]
                for j in range(bbs.shape[0]):
                    #circle aligned predictions
                    conf = bbs[j,0]
                    shade = 0.0+(conf-threshConf)/(maxConf-threshConf)
                    #print(shade)
                    #if name=='text_start_gt' or name=='field_end_gt':
                    #    cv2.bb(bbImage[:,:,1],p1,p2,shade,2)
                    #if name=='text_end_gt':
                    #    cv2.bb(bbImage[:,:,2],p1,p2,shade,2)
                    #elif name=='field_end_gt' or name=='field_start_gt':
                    #    cv2.bb(bbImage[:,:,0],p1,p2,shade,2)
                    if j==bestBBIdx[b]:
                        if bbs[j,6] > bbs[j,7]:
                            color=(0,shade,shade) #textF
                        else:
                            color=(shade,0,shade) #field
                    elif j==secondBBIdx[b]:
                        if bbs[j,6] > bbs[j,7]:
                            color=(0,0.5*shade,shade) #textF
                        else:
                            color=(shade,0,0.5*shade) #field
                    else:
                        if bbs[j,6] > bbs[j,7]:
                            color=(0,0,shade) #text
                        else:
                            color=(shade,0,0) #field
                    plotRect(imageB,color,bbs[j,1:6])
                #conf = outputBBs[b][bestBBIdx[b],0]
                #shade = 0.0+(conf-threshConf)/(maxConf-threshConf)

            #for j in alignmentBBsTarg[name][b]:
            #    p1 = (targetBBs[name][b,j,0], targetBBs[name][b,j,1])
            #    p2 = (targetBBs[name][b,j,0], targetBBs[name][b,j,1])
            #    mid = ( int(round((p1[0]+p2[0])/2.0)), int(round((p1[1]+p2[1])/2.0)) )
            #    rad = round(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/2.0)
            #    #print(mid)
            #    #print(rad)
            #    cv2.circle(imageB,mid,rad,(1,0,1),1)

            saveName = '{:04}_n:{}_pairing_AP:{:.2f}_prec:{:.2f}_recall:{:.2f}_bestCnf:{:.3f}_2ndCnf:{:.3f}_thresh:{:.3f}'.format(startIndex+b,imageName[b],aps_5[b][0],precs_5[b][0],recalls_5[b][0],bestConf[b],secondConf[b],threshConf)
            #for j in range(metricsOut.shape[1]):
            #    saveName+='_m:{0:.3f}'.format(metricsOut[i,j])
            saveName+='.png'
            io.imsave(os.path.join(outDir,saveName),imageB)

        
    #return metricsOut
    return (
             #{ 'ap_5':np.array(aps_5).sum(axis=0),
             #  'ap_3':np.array(aps_3).sum(axis=0),
             #  'ap_7':np.array(aps_7).sum(axis=0),
             #  'recall':np.array(recalls_5).sum(axis=0),
             #  'prec':np.array(precs_5).sum(axis=0),
             #}, 
             { 'ap_5':aps_5,
               'class_aps':class_aps,
               #'ap_3':aps_3,
               #'ap_7':aps_7,
               'recall':recalls_5,
               'prec':precs_5,

               'apSame':apsSame,
               'recallSame':recallsSame,
               'precSame':precsSame,

               'apDiff':apsDiff,
               'recallDiff':recallsDiff,
               'precDiff':precsDiff,
             }, 
             (lossThis, position_loss, conf_loss, class_loss, recall, precision)
            )


