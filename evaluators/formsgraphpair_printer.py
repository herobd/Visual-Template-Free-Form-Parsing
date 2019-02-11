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
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist, getTargIndexForPreds_iou, getTargIndexForPreds_dist, computeAP
from model.optimize import optimizeRelationships, optimizeRelationshipsSoft


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
        num_neighbors = instance['num_neighbors']

        if gpu is not None:
            image = image.to(gpu)
            if bbs is not None:
                bbs = bbs.to(gpu)
            if num_neighbors is not None:
                num_neighbors = num_neighbors.to(gpu)
            #adjacenyMatrix = adjacenyMatrix.to(self.gpu)
        return image, bbs, adjaceny, num_neighbors

    EDGE_THRESH = config['THRESH'] if 'THRESH' in config else 0.0
    #print(type(instance['pixel_gt']))
    #if type(instance['pixel_gt']) == list:
    #    print(instance)
    #    print(startIndex)
    #data, targetBB, targetBBSizes = instance
    lossWeights = config['loss_weights'] if 'loss_weights' in config else {"box": 1, "rel":1}
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
    gtNumNeighbors = instance['num_neighbors']
    if not model.detector.predNumNeighbors:
        instance['num_neighbors']=None
    dataT, targetBBsT, adjT, gtNumNeighborsT = __to_tensor(instance,gpu)


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
    outputBBs, outputOffsets, relPred, relIndexes = model(dataT)

    if model.detector.predNumNeighbors:
        predNN = outputBBs[:,6]
    else:
        predNN = None

    if targetBBsT is not None:
        targetSize=targetBBsT.size(1)
    else:
        targetSize=0
    lossThis, position_loss, conf_loss, class_loss, nn_loss, recall, precision = yolo_loss(outputOffsets,targetBBsT,[targetSize], gtNumNeighborsT)


    relCand = relIndexes
    relPred = 2*torch.sigmoid(relPred)[:,0] -1

    numClasses=2
    if model.rotation:
        targIndex, predWithNoIntersection = getTargIndexForPreds_dist(targetBBs[0],outputBBs,1.1,numClasses)
    else:
        targIndex, predWithNoIntersection = getTargIndexForPreds_iou(targetBBs[0],outputBBs,0.4,numClasses)
    if targetBBs is not None:
        target_for_b = targetBBs[0,:,:]
    else:
        target_for_b = torch.empty(0)
    if 'optimize' in config and config['optimize']:
        if 'penalty' in config:
            penalty = config['penalty']
        else:
            penalty = 0.5
        thresh=0.3
        while thresh<0.9:
            keep = relPred>thresh
            newRelPred = relPred[keep]
            if newRelPred.size(0)<700:
                break
        if newRelPred.size(0)>0:
            #newRelCand = [ cand for i,cand in enumerate(relCand) if keep[i] ]
            usePredNN= predNN is not None and config['optimize']!='gt'
            idMap={}
            newId=0
            newRelCand=[]
            numNeighbors=[]
            for index,(id1,id2) in enumerate(relCand):
                if keep[index]:
                    if id1 not in idMap:
                        idMap[id1]=newId
                        if not usePredNN:
                            numNeighbors.append(gtNumNeighbors[0,targIndex[index]])
                        else:
                            numNeighbors.append(predNN[id1])
                        newId+=1
                    if id2 not in idMap:
                        idMap[id2]=newId
                        if not usePredNN:
                            numNeighbors.append(gtNumNeighbors[0,targIndex[index]])
                        else:
                            numNeighbors.append(predNN[id2])
                        newId+=1
                    newRelCand.append( [idMap[id1],idMap[id2]] )            


            if not usePredNN:
                decision = optimizeRelationships(newRelPred,newRelCand,numNeighbors,penalty)
            else:
                decision= optimizeRelationshipsSoft(newRelPred,newRelCand,numNeighbors,penalty)
            decision= torch.from_numpy( np.round_(decision).astype(int) )
            relPred[keep] = torch.where(0==decision,relPred[keep]-2,relPred[keep])
            relPred[1-keep] -=2
            EDGE_THRESH=-1

    data = data.numpy()
    #threshed in model
    maxConf = outputBBs[:,0].max().item()
    minConf = outputBBs[:,0].min().item()
    #threshConf = max(maxConf*THRESH,0.5)
    #if model.rotation:
    #    outputBBs = non_max_sup_dist(outputBBs.cpu(),threshConf,3)
    #else:
    #    outputBBs = non_max_sup_iou(outputBBs.cpu(),threshConf,0.4)

    if model.detector.predNumNeighbors:
        useOutputBBs=torch.cat((outputBBs[:,0:6],outputBBs[:,7:]),dim=1) #throw away NN pred
    else:
        useOutputBBs=outputBBs
    if model.rotation:
        ap_5, prec_5, recall_5 =AP_dist(target_for_b,useOutputBBs,0.9,model.numBBTypes)
    else:
        ap_5, prec_5, recall_5 =AP_iou(target_for_b,useOutputBBs,0.5,model.numBBTypes)
    useOutputBBs=None

    truePred=falsePred=badPred=0
    scores=[]
    matches=0
    i=0
    for n0,n1 in relCand:
        t0 = targIndex[n0].item()
        t1 = targIndex[n1].item()
        if t0>=0 and t1>=0:
            if (min(t0,t1),max(t0,t1)) in adjacency:
                matches+=1
                scores.append( (relPred[i],True) )
                if relPred[i]>EDGE_THRESH:
                    truePred+=1
            else:
                scores.append( (relPred[i],False) )
                if relPred[i]>EDGE_THRESH:
                    falsePred+=1
        else:
            scores.append( (relPred[i],False) )
            if relPred[i]>EDGE_THRESH:
                badPred+=1
        i+=1
    for i in range(len(adjacency)-matches):
        scores.append( (float('nan'),True) )
    rel_ap=computeAP(scores)
    if len(adjacency)>0:
        relRecall = truePred/len(adjacency)
    else:
        relRecall = 1
    if falsePred>0:
        relPrec = truePred/(truePred+falsePred)
    else:
        relPrec = 1
    if falsePred+badPred>0:
        fullPrec = truePred/(truePred+falsePred+badPred)
    else:
        fullPrec = 1

    #import pdb;pdb.set_trace()

    #for b in range(len(outputBBs)):
    outputBBs = outputBBs.data.numpy()
    
    
    dists=defaultdict(list)
    dists_x=defaultdict(list)
    dists_y=defaultdict(list)
    scaleDiffs=defaultdict(list)
    rotDiffs=defaultdict(list)
    b=0
    #print('image {} has {} {}'.format(startIndex+b,targetBBsSizes[name][b],name))
    #bbImage = np.ones_like(image):w

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
            #x=int(targetBBs[b,j,0])
            #y=int(targetBBs[b,j,1]+targetBBs[b,j,3])
            #cv2.putText(image,'{:.2f}'.format(gtNumNeighbors[b,j]),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0.6,0.3,0),2,cv2.LINE_AA)
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

                if model.detector.predNumNeighbors:
                    x=int(bbs[j,1])
                    y=int(bbs[j,2]-bbs[j,4])
                    targ_j = targIndex[j].item()
                    if targ_j>=0:
                        gtNN = gtNumNeighbors[targ_j]
                    else:
                        gtNN = 0
                    color = int(min(abs(predNN[j]-gtNN),2)*127)
                    cv2.putText(image,'{}/{}'.format(predNN[j],gtNN),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 3,(color,0,0),2,cv2.LINE_AA)

        #for j in alignmentBBsTarg[name][b]:
        #    p1 = (targetBBs[name][b,j,0], targetBBs[name][b,j,1])
        #    p2 = (targetBBs[name][b,j,0], targetBBs[name][b,j,1])
        #    mid = ( int(round((p1[0]+p2[0])/2.0)), int(round((p1[1]+p2[1])/2.0)) )
        #    rad = round(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/2.0)
        #    #print(mid)
        #    #print(rad)
        #    cv2.circle(image,mid,rad,(1,0,1),1)
        for i,j in adjacency:
            x1 = round(targetBBs[0,i,0].item())
            y1 = round(targetBBs[0,i,1].item())
            x2 = round(targetBBs[0,j,0].item())
            y2 = round(targetBBs[0,j,1].item())
            cv2.line(image,(x1,y1),(x2,y2),(0.25,0,0.25),3)

        numrelpred=0
        for i in range(len(relCand)):
            #print('{},{} : {}'.format(relCand[i][0],relCand[i][1],relPred[i]))
            if relPred[i]>EDGE_THRESH:
                ind1 = relCand[i][0]
                ind2 = relCand[i][1]
                x1 = round(bbs[ind1,1])
                y1 = round(bbs[ind1,2])
                x2 = round(bbs[ind2,1])
                y2 = round(bbs[ind2,2])

                shade = (relPred[i].item()-EDGE_THRESH)/(1-EDGE_THRESH)

                #print('draw {} {} {} {} '.format(x1,y1,x2,y2))
                cv2.line(image,(x1,y1),(x2,y2),(0,shade,0),1)
                numrelpred+=1
        #print('number of pred rels: {}'.format(numrelpred))

        for predI in range(bbs.shape[0]):
            targI=targIndex[predI].item()
            if targI>0:
                x1 = round(bbs[predI,1])
                y1 = round(bbs[predI,2])

                x2 = round(targetBBs[0,targI,0].item())
                y2 = round(targetBBs[0,targI,1].item())
                cv2.line(image,(x1,y1),(x2,y2),(1,0,1),1)



        saveName = '{}_boxes_prec:{:.2f},{:.2f}_recall:{:.2f},{:.2f}_rels_AP:{:.3f}'.format(imageName,prec_5[0],prec_5[1],recall_5[0],recall_5[1],rel_ap)
        #for j in range(metricsOut.shape[1]):
        #    saveName+='_m:{0:.3f}'.format(metricsOut[i,j])
        saveName+='.png'
        io.imsave(os.path.join(outDir,saveName),image)
        #print('saved: '+os.path.join(outDir,saveName))

        
    retData= { 'bb_ap':[ap_5],
               'bb_recall':[recall_5],
               'bb_prec':[prec_5],
               'bb_Fm': (recall_5[0]+recall_5[1]+prec_5[0]+prec_5[1])/4,
               'nn_loss': nn_loss,
               'rel_recall':relRecall,
               'rel_prec':relPrec,
               'rel_Fm':(relRecall+relPrec)/2,
               'rel_fullPrec':fullPrec,
               'rel_fullFm':(relRecall+fullPrec)/2,

             }
    if rel_ap is not None: #none ap if no relationships
        retData['rel_AP']=rel_ap
    return (
             retData,
             (lossThis, position_loss, conf_loss, class_loss, recall, precision)
            )


