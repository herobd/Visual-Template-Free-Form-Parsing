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
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist


def plotRect(img,color,xyrhw):
    xc=xyrhw[0].item()
    yc=xyrhw[1].item()
    rot=xyrhw[2].item()
    h=xyrhw[3].item()
    w=xyrhw[4].item()
    h = min(30000,h)
    w = min(30000,w)
    tr = ( int(w*math.cos(rot)-h*math.sin(rot) + xc),  int(w*math.sin(rot)+h*math.cos(rot) + yc) )
    tl = ( int(-w*math.cos(rot)-h*math.sin(rot) + xc), int(-w*math.sin(rot)+h*math.cos(rot) + yc) )
    br = ( int(w*math.cos(rot)+h*math.sin(rot) + xc),  int(w*math.sin(rot)-h*math.cos(rot) + yc) )
    bl = ( int(-w*math.cos(rot)+h*math.sin(rot) + xc), int(-w*math.sin(rot)-h*math.cos(rot) + yc) )

    cv2.line(img,tl,tr,color,1)
    cv2.line(img,tr,br,color,1)
    cv2.line(img,br,bl,color,1)
    cv2.line(img,bl,tl,color,1)

def FormsGraphPair_printer(config,instance, model, gpu, metrics, outDir=None, startIndex=None, lossFunc=None):
    def __eval_metrics(data,target):
        acc_metrics = np.zeros((output.shape[0],len(metrics)))
        for ind in range(output.shape[0]):
            for i, metric in enumerate(metrics):
                acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
        return acc_metrics

    def __to_tensor(instance,gpu):
        image = instance['img']
        bbs = instance['bb_gt']
        adjaceny = instance['adj']

        if gpu is not None:
            image = image.to(gpu)
            if bbs is not None:
                bbs = bbs.to(gpu)
            #adjacenyMatrix = adjacenyMatrix.to(self.gpu)
        return image, bbs, adjaceny    

    EDGE_THRESH = config['THRESH'] if 'THRESH' in config else 0.5
    #print(type(instance['pixel_gt']))
    #if type(instance['pixel_gt']) == list:
    #    print(instance)
    #    print(startIndex)
    #data, targetBB, targetBBSizes = instance
    lossWeights = config['loss_weights'] if 'loss_weights' in config else {"box": 1, "edge":1}
    if lossFunc is None:
        yolo_loss = YoloLoss(model.numBBTypes,model.rotation,model.scale,model.anchors,**config['loss_params']['box'])
    else:
        yolo_loss = lossFunc
    data = instance['img']
    batchSize = data.shape[0]
    assert(batchSize==1)
    targetBBs = instance['bb_gt']
    adjacency = instance['adj']
    imageName = instance['imgName']
    scale = instance['scale']
    dataT, targetBBsT, adjT = __to_tensor(instance,gpu)

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
    outputBBs, outputOffsets, edgePred = model(dataT)

    if targetBBsT is not None:
        targetSize=targetBBsT.size(1)
    else:
        targetSize=0
    lossThis, position_loss, conf_loss, class_loss, recall, precision = yolo_loss(outputOffsets,targetBBsT,[targetSize])

    #TODO edge loss

    edgeCand = edgePred[0]
    edgePred = torch.sigmoid(edgePred[1])

    data = data.numpy()
    #threshed in model
    maxConf = outputBBs[:,0].max().item()
    minConf = outputBBs[:,0].min().item()
    #threshConf = max(maxConf*THRESH,0.5)
    #if model.rotation:
    #    outputBBs = non_max_sup_dist(outputBBs.cpu(),threshConf,3)
    #else:
    #    outputBBs = non_max_sup_iou(outputBBs.cpu(),threshConf,0.4)

    if targetBBs is not None:
        target_for_b = targetBBs[0,:,:]
    else:
        target_for_b = torch.empty(0)
    if model.rotation:
        ap_5, prec_5, recall_5 =AP_dist(target_for_b,outputBBs,0.9,model.numBBTypes)
    else:
        ap_5, prec_5, recall_5 =AP_iou(target_for_b,outputBBs,0.5,model.numBBTypes)


    #for b in range(len(outputBBs)):
    outputBBs = outputBBs.data.numpy()
    
    
    dists=defaultdict(list)
    dists_x=defaultdict(list)
    dists_y=defaultdict(list)
    scaleDiffs=defaultdict(list)
    rotDiffs=defaultdict(list)
    b=0
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
        image = (1-((1+np.transpose(data[b][:,:,:],(1,2,0)))/2.0)).copy()
        if image.shape[2]==1:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        #if name=='text_start_gt':

        for j in range(targetSize):
            plotRect(image,(1,0.5,0),targetBBs[0,j,0:5])
            #if alignmentBBs[b] is not None:
            #    aj=alignmentBBs[b][j]
            #    xc_gt = targetBBs[b,j,0]
            #    yc_gt = targetBBs[b,j,1]
            #    xc=outputBBs[b,aj,1]
            #    yc=outputBBs[b,aj,2]
            #    cv2.line(image,(xc,yc),(xc_gt,yc_gt),(0,1,0),1)
            #    shade = 0.0+(outputBBs[b,aj,0]-threshConf)/(maxConf-threshConf)
            #    shade = max(0,shade)
            #    if outputBBs[b,aj,6] > outputBBs[b,aj,7]:
            #        color=(0,shade,shade) #text
            #    else:
            #        color=(shade,shade,0) #field
            #    plotRect(image,color,outputBBs[b,aj,1:6])

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
        bbs = outputBBs
        for j in range(bbs.shape[0]):
            #circle aligned predictions
            conf = bbs[j,0]
            if outDir is not None:
                shade = 0.0+(conf-minConf)/(maxConf-minConf)
                #print(shade)
                #if name=='text_start_gt' or name=='field_end_gt':
                #    cv2.bb(bbImage[:,:,1],p1,p2,shade,2)
                #if name=='text_end_gt':
                #    cv2.bb(bbImage[:,:,2],p1,p2,shade,2)
                #elif name=='field_end_gt' or name=='field_start_gt':
                #    cv2.bb(bbImage[:,:,0],p1,p2,shade,2)
                if bbs[j,6] > bbs[j,7]:
                    color=(0,0,shade) #text
                else:
                    color=(shade,0,0) #field
                plotRect(image,color,bbs[j,1:6])

        #for j in alignmentBBsTarg[name][b]:
        #    p1 = (targetBBs[name][b,j,0], targetBBs[name][b,j,1])
        #    p2 = (targetBBs[name][b,j,0], targetBBs[name][b,j,1])
        #    mid = ( int(round((p1[0]+p2[0])/2.0)), int(round((p1[1]+p2[1])/2.0)) )
        #    rad = round(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/2.0)
        #    #print(mid)
        #    #print(rad)
        #    cv2.circle(image,mid,rad,(1,0,1),1)
        
        for i in range(len(edgeCand)):
            if edgePred[i]>EDGE_THRESH:
                ind1 = edgeCand[i][0]
                ind2 = edgeCand[i][1]
                x1 = round(bbs[ind1,0])
                y1 = round(bbs[ind1,1])
                x2 = round(bbs[ind2,0])
                y2 = round(bbs[ind2,2])

                cv2.line(image,(x1,y1),(x2,y2),(0,1,0),1)



        saveName = '{}_boxes_prec:{:.2f},{:.2f}_recall:{:.2f},{:.2f}'.format(imageName,prec_5[0],prec_5[1],recall_5[0],recall_5[1])
        #for j in range(metricsOut.shape[1]):
        #    saveName+='_m:{0:.3f}'.format(metricsOut[i,j])
        saveName+='.png'
        io.imsave(os.path.join(outDir,saveName),image)
        #print('saved: '+os.path.join(outDir,saveName))

        
    #return metricsOut
    return (
             #{ 'ap_5':np.array(aps_5).sum(axis=0),
             #  'ap_3':np.array(aps_3).sum(axis=0),
             #  'ap_7':np.array(aps_7).sum(axis=0),
             #  'recall':np.array(recalls_5).sum(axis=0),
             #  'prec':np.array(precs_5).sum(axis=0),
             #}, 
             { 'ap_5':ap_5,
               'recall':recall_5,
               'prec':prec_5,
             }, 
             (lossThis, position_loss, conf_loss, class_loss, recall, precision)
            )


