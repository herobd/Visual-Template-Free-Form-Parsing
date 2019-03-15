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
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist, getTargIndexForPreds_iou, getTargIndexForPreds_dist, computeAP
from model.optimize import optimizeRelationships, optimizeRelationshipsSoft
import json
from utils.forms_annotations import fixAnnotations, getBBInfo


def plotRect(img,color,xyrhw,lineWidth=1):
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

    cv2.line(img,tl,tr,color,lineWidth)
    cv2.line(img,tr,br,color,lineWidth)
    cv2.line(img,br,bl,color,lineWidth)
    cv2.line(img,bl,tl,color,lineWidth)

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

    rel_thresholds = [config['THRESH']] if 'THRESH' in config else [0.5]
    if ('sweep_threshold' in config and config['sweep_threshold']) or ('sweep_thresholds' in config and config['sweep_thresholds']):
        rel_thresholds = np.arange(0.1,1.0,0.05)
    if ('sweep_threshold_big' in config and config['sweep_threshold_big']) or ('sweep_thresholds_big' in config and config['sweep_thresholds_big']):
        rel_thresholds = np.arange(0,20.0,1)
    if ('sweep_threshold_small' in config and config['sweep_threshold_small']) or ('sweep_thresholds_small' in config and config['sweep_thresholds_small']):
        rel_thresholds = np.arange(0,0.1,0.01)
    draw_rel_thresh = config['draw_thresh'] if 'draw_thresh' in config else rel_thresholds[0]
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
    targetBoxes = instance['bb_gt']
    adjacency = instance['adj']
    adjacency = list(adjacency)
    imageName = instance['imgName']
    scale = instance['scale']
    target_num_neighbors = instance['num_neighbors']
    if not model.detector.predNumNeighbors:
        instance['num_neighbors']=None
    dataT, targetBoxesT, adjT, target_num_neighborsT = __to_tensor(instance,gpu)


    pretty = config['pretty'] if 'pretty' in config else False
    useDetections = config['useDetections'] if 'useDetections' in config else False
    if 'useDetect' in config:
        useDetections = config['useDetect']
    confThresh = config['conf_thresh'] if 'conf_thresh' in config else None


    numClasses=2 #TODO no hard code

    resultsDirName='results'
    #if outDir is not None and resultsDirName is not None:
        #rPath = os.path.join(outDir,resultsDirName)
        #if not os.path.exists(rPath):
        #    os.mkdir(rPath)
        #for name in targetBoxes:
        #    nPath = os.path.join(rPath,name)
        #    if not os.path.exists(nPath):
        #        os.mkdir(nPath)

    #dataT = __to_tensor(data,gpu)
    #print('{}: {} x {}'.format(imageName,data.shape[2],data.shape[3]))
    if useDetections=='gt':
        outputBoxes, outputOffsets, relPred, relIndexes, bbPred = model(dataT,targetBoxesT,target_num_neighborsT,True,
                otherThresh=confThresh,
                otherThreshIntur=1 if confThresh is not None else None,
                hard_detect_limit=600)
        outputBoxes=torch.cat((torch.ones(targetBoxes.size(1),1),targetBoxes[0,:,0:5],targetBoxes[0,:,-numClasses:]),dim=1) #add score
    elif type(useDetections) is str:
        dataset=config['DATASET']
        jsonPath = os.path.join(useDetections,imageName+'.json')
        with open(os.path.join(jsonPath)) as f:
            annotations = json.loads(f.read())
        fixAnnotations(dataset,annotations)
        savedBoxes = torch.FloatTensor(len(annotations['byId']),6+model.detector.predNumNeighbors+numClasses)
        for i,(id,bb) in enumerate(annotations['byId'].items()):
            qX, qY, qH, qW, qR, qIsText, qIsField, qIsBlank, qNN = getBBInfo(bb,dataset.rotate,useBlankClass=not dataset.no_blanks)
            savedBoxes[i,0]=1 #conf
            savedBoxes[i,1]=qX*scale #x-center, already scaled
            savedBoxes[i,2]=qY*scale #y-center
            savedBoxes[i,3]=qR #rotation
            savedBoxes[i,4]=qH*scale/2
            savedBoxes[i,5]=qW*scale/2
            if model.detector.predNumNeighbors:
                extra=1
                savedBoxes[i,6]=qNN
            else:
                extra=0
            savedBoxes[i,6+extra]=qIsText
            savedBoxes[i,7+extra]=qIsField
            
        if gpu is not None:
            savedBoxes=savedBoxes.to(gpu)
        outputBoxes, outputOffsets, relPred, relIndexes, bbPred = model(dataT,savedBoxes,None,"saved",
                otherThresh=confThresh,
                otherThreshIntur=1 if confThresh is not None else None,
                hard_detect_limit=600)
        outputBoxes=savedBoxes.cpu()
    elif useDetections:
        print('Unknown detection flag: '+useDetections)
        exit()
    else:
        outputBoxes, outputOffsets, relPred, relIndexes, bbPred = model(dataT,
                otherThresh=confThresh,
                otherThreshIntur=1 if confThresh is not None else None,
                hard_detect_limit=600)

    if model.predNN and bbPred is not None:
        predNN = bbPred[:,0]
    else:
        predNN=None
    if  model.detector.predNumNeighbors and not useDetections:
        #useOutputBBs=torch.cat((outputBoxes[:,0:6],outputBoxes[:,7:]),dim=1) #throw away NN pred
        extraPreds=1
        if not model.predNN:
            predNN = outputBoxes[:,6]
    else:
        extraPreds=0
        if not model.predNN:
            predNN = None
        #useOutputBBs=outputBoxes

    if targetBoxesT is not None:
        targetSize=targetBoxesT.size(1)
    else:
        targetSize=0
    lossThis, position_loss, conf_loss, class_loss, nn_loss, recall, precision = yolo_loss(outputOffsets,targetBoxesT,[targetSize], target_num_neighborsT)

    if 'rule' in config:
        if config['rule']=='closest':
            dists = torch.FloatTensor(relPred.size())
            differentClass = torch.FloatTensor(relPred.size())
            predClasses = torch.argmax(outputBoxes[:,extraPreds+6:extraPreds+6+numClasses],dim=1)
            for i,(bb1,bb2) in enumerate(relIndexes):
                dists[i] = math.sqrt((outputBoxes[bb1,1]-outputBoxes[bb2,1])**2 + (outputBoxes[bb1,2]-outputBoxes[bb2,2])**2)
                differentClass[i] = predClasses[bb1]!=predClasses[bb2]
            maxDist = torch.max(dists)
            minDist = torch.min(dists)
            relPred = 1-(dists-minDist)/(maxDist-minDist)
            relPred *= differentClass
        elif config['rule']=='icdar':
            height = torch.FloatTensor(relPred.size())
            dists = torch.FloatTensor(relPred.size())
            right = torch.FloatTensor(relPred.size())
            sameClass = torch.FloatTensor(relPred.size())
            predClasses = torch.argmax(outputBoxes[:,extraPreds+6:extraPreds+6+numClasses],dim=1)
            for i,(bb1,bb2) in enumerate(relIndexes):
                sameClass[i] = predClasses[bb1]==predClasses[bb2]
                
                #g4 of the paper
                height[i] = max(outputBoxes[bb1,4],outputBoxes[bb2,4])/min(outputBoxes[bb1,4],outputBoxes[bb2,4])

                #g5 of the paper
                if predClasses[bb1]==0:
                    widthLabel = outputBoxes[bb1,5]*2 #we predict half width
                    widthValue = outputBoxes[bb2,5]*2
                    dists[i] = math.sqrt(((outputBoxes[bb1,1]+widthLabel)-(outputBoxes[bb2,1]-widthValue))**2 + (outputBoxes[bb1,2]-outputBoxes[bb2,2])**2)
                else:
                    widthLabel = outputBoxes[bb2,5]*2 #we predict half width
                    widthValue = outputBoxes[bb1,5]*2
                    dists[i] = math.sqrt(((outputBoxes[bb1,1]-widthValue)-(outputBoxes[bb2,1]+widthLabel))**2 + (outputBoxes[bb1,2]-outputBoxes[bb2,2])**2)
                if dists[i]>2*widthLabel:
                    dists[i]/=widthLabel
                else: #undefined
                    dists[i] = min(1,dists[i]/widthLabel)
            
                #g6 of the paper
                if predClasses[bb1]==0:
                    widthValue = outputBoxes[bb2,5]*2
                    hDist = outputBoxes[bb1,1]-outputBoxes[bb2,1]
                else:
                    widthValue = outputBoxes[bb1,5]*2
                    hDist = outputBoxes[bb2,1]-outputBoxes[bb1,1]
                right[i] = hDist/widthValue

            relPred = 1-(height+dists+right + 10000*sameClass)
        else:
            print('ERROR, unknown rule {}'.format(config['rule']))
            exit()
    elif relPred is not None:
        relPred = torch.sigmoid(relPred)[:,0]




    relCand = relIndexes
    if relCand is None:
        relCand=[]

    if model.rotation:
        bbAlignment, bbFullHit = getTargIndexForPreds_dist(targetBoxes[0],outputBoxes,0.9,numClasses,extraPreds,hard_thresh=False)
    else:
        bbAlignment, bbFullHit = getTargIndexForPreds_iou(targetBoxes[0],outputBoxes,0.5,numClasses,extraPreds,hard_thresh=False)
    if targetBoxes is not None:
        target_for_b = targetBoxes[0,:,:]
    else:
        target_for_b = torch.empty(0)

    if outputBoxes.size(0)>0:
        maxConf = outputBoxes[:,0].max().item()
        minConf = outputBoxes[:,0].min().item()
        if useDetections:
            minConf=0
    #threshConf = max(maxConf*THRESH,0.5)
    #if model.rotation:
    #    outputBoxes = non_max_sup_dist(outputBoxes.cpu(),threshConf,3)
    #else:
    #    outputBoxes = non_max_sup_iou(outputBoxes.cpu(),threshConf,0.4)
    if model.rotation:
        ap_5, prec_5, recall_5 =AP_dist(target_for_b,outputBoxes,0.9,model.numBBTypes,beforeCls=extraPreds)
    else:
        ap_5, prec_5, recall_5 =AP_iou(target_for_b,outputBoxes,0.5,model.numBBTypes,beforeCls=extraPreds)

    #precisionHistory={}
    #precision=-1
    #minStepSize=0.025
    #targetPrecisions=[None]
    #for targetPrecision in targetPrecisions:
    #    if len(precisionHistory)>0:
    #        closestPrec=9999
    #        for prec in precisionHistory:
    #            if abs(targetPrecision-prec)<abs(closestPrec-targetPrecision):
    #                closestPrec=prec
    #        precision=prec
    #        stepSize=precisionHistory[prec][0]
    #    else:
    #        stepSize=0.1
    #
    #    while True: #abs(precision-targetPrecision)>0.001:
    toRet={}
    for rel_threshold in rel_thresholds:

            if 'optimize' in config and config['optimize']:
                if 'penalty' in config:
                    penalty = config['penalty']
                else:
                    penalty = 0.25
                print('optimizing with penalty {}'.format(penalty))
                thresh=0.15
                while thresh<0.45:
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
                                    numNeighbors.append(target_num_neighbors[0,bbAlignment[id1]])
                                else:
                                    numNeighbors.append(predNN[id1])
                                newId+=1
                            if id2 not in idMap:
                                idMap[id2]=newId
                                if not usePredNN:
                                    numNeighbors.append(target_num_neighbors[0,bbAlignment[id2]])
                                else:
                                    numNeighbors.append(predNN[id2])
                                newId+=1
                            newRelCand.append( [idMap[id1],idMap[id2]] )            


                    #if not usePredNN:
                        #    decision = optimizeRelationships(newRelPred,newRelCand,numNeighbors,penalty)
                    #else:
                    decision= optimizeRelationshipsSoft(newRelPred,newRelCand,numNeighbors,penalty, rel_threshold)
                    decision= torch.from_numpy( np.round_(decision).astype(int) )
                    decision=decision.to(relPred.device)
                    relPred[keep] = torch.where(0==decision,relPred[keep]-1,relPred[keep])
                    relPred[1-keep] -=1
                    rel_threshold_use=0#-0.5
                else:
                    rel_threshold_use=rel_threshold
            else:
                rel_threshold_use=rel_threshold

            #threshed in model
            #if len(precisionHistory)==0:
            if len(toRet)==0:
                #align bb predictions (final) with GT
                if bbPred is not None and bbPred.size(0)>0:
                    #create aligned GT
                    #this was wrong...
                        #first, remove unmatched predicitons that didn't overlap (weren't close) to any targets
                        #toKeep = 1-((bbNoIntersections==1) * (bbAlignment==-1))
                    #remove predictions that overlapped with GT, but not enough
                    if model.predNN:
                        start=1
                        toKeep = 1-((bbFullHit==0) * (bbAlignment!=-1)) #toKeep = not (incomplete_overlap and did_overlap)
                        if toKeep.any():
                            bbPredNN_use = bbPred[toKeep][:,0]
                            bbAlignment_use = bbAlignment[toKeep]
                            #becuase we used -1 to indicate no match (in bbAlignment), we add 0 as the last position in the GT, as unmatched 
                            if target_num_neighborsT is not None:
                                target_num_neighbors_use = torch.cat((target_num_neighborsT[0].float(),torch.zeros(1).to(target_num_neighborsT.device)),dim=0)
                            else:
                                target_num_neighbors_use = torch.zeros(1).to(bbPred.device)
                            alignedNN_use = target_num_neighbors_use[bbAlignment_use]
                        else:
                            bbPredNN_use=None
                            alignedNN_use=None
                    else:
                        start=0
                    if model.predClass:
                        #We really don't care about the class of non-overlapping instances
                        if targetBoxes is not None:
                            toKeep = bbFullHit==1
                            if toKeep.any():
                                bbPredClass_use = bbPred[toKeep][:,start:start+model.numBBTypes]
                                bbAlignment_use = bbAlignment[toKeep]
                                alignedClass_use =  targetBoxesT[0][bbAlignment_use][:,13:13+model.numBBTypes] #There should be no -1 indexes in hereS
                            else:
                                bbPredClass_use=None
                                alignedClass_use=None
                        else:
                            alignedClass_use = None
                else:
                    bbPredNN_use = None
                    bbPredClass_use = None
                if model.predNN and bbPredNN_use is not None and bbPredNN_use.size(0)>0:
                    nn_loss_final = F.mse_loss(bbPredNN_use,alignedNN_use)
                    #nn_loss_final *= self.lossWeights['nn']

                    #loss += nn_loss_final
                    nn_loss_final = nn_loss_final.item()
                else:
                    nn_loss_final=0
                if model.predNN and predNN is not None:
                    predNN_p=bbPred[:,0]
                    diffs=torch.abs(predNN_p-target_num_neighborsT[0][bbAlignment].float())
                    nn_acc = (diffs<0.5).sum().item()
                    nn_acc /= predNN.size(0)
                elif model.predNN:
                    nn_acc = 0 
                if model.detector.predNumNeighbors and not useDetections:
                    predNN_d = outputBoxes[:,6]
                    diffs=torch.abs(predNN_d-target_num_neighbors[0][bbAlignment].float())
                    nn_acc_d = (diffs<0.5).sum().item()
                    nn_acc_d /= predNN.size(0)

                if model.predClass and bbPredClass_use is not None and bbPredClass_use.size(0)>0:
                    class_loss_final = F.binary_cross_entropy_with_logits(bbPredClass_use,alignedClass_use)
                    #class_loss_final *= self.lossWeights['class']
                    #loss += class_loss_final
                    class_loss_final = class_loss_final.item()
                else:
                    class_loss_final = 0
            #class_acc=0
            useOutputBBs=None

            truePred=falsePred=badPred=0
            scores=[]
            matches=0
            i=0
            numMissedByHeur=0
            targGotHit=set()
            for i,(n0,n1) in enumerate(relCand):
                t0 = bbAlignment[n0].item()
                t1 = bbAlignment[n1].item()
                if t0>=0 and bbFullHit[n0]:
                    targGotHit.add(t0)
                if t1>=0 and bbFullHit[n1]:
                    targGotHit.add(t1)
                if t0>=0 and t1>=0 and bbFullHit[n0] and bbFullHit[n1]:
                    if (min(t0,t1),max(t0,t1)) in adjacency:
                        matches+=1
                        scores.append( (relPred[i],True) )
                        if relPred[i]>rel_threshold_use:
                            truePred+=1
                    else:
                        scores.append( (relPred[i],False) )
                        if relPred[i]>rel_threshold_use:
                            falsePred+=1
                else:
                    scores.append( (relPred[i],False) )
                    if relPred[i]>rel_threshold_use:
                        badPred+=1
            for i in range(len(adjacency)-matches):
                numMissedByHeur+=1
                scores.append( (float('nan'),True) )
            rel_ap=computeAP(scores)

            numMissedByDetect=0
            for t0,t1 in adjacency:
                if t0 not in targGotHit or t1 not in targGotHit:
                    numMissedByHeur-=1
                    numMissedByDetect+=1
            heurRecall = (len(adjacency)-numMissedByHeur)/len(adjacency)
            detectRecall = (len(adjacency)-numMissedByDetect)/len(adjacency)
            if len(adjacency)>0:
                relRecall = truePred/len(adjacency)
            else:
                relRecall = 1
            #if falsePred>0:
            #    relPrec = truePred/(truePred+falsePred)
            #else:
            #    relPrec = 1
            if falsePred+badPred>0:
                precision = truePred/(truePred+falsePred+badPred)
            else:
                precision = 1
    

            toRet['prec@{}'.format(rel_threshold)]=precision
            toRet['recall@{}'.format(rel_threshold)]=relRecall
            if relRecall+precision>0:
                toRet['F-M@{}'.format(rel_threshold)]=2*relRecall*precision/(relRecall+precision)
            else:
                toRet['F-M@{}'.format(rel_threshold)]=0
            toRet['rel_AP@{}'.format(rel_threshold)]=rel_ap
            #precisionHistory[precision]=(draw_rel_thresh,stepSize)
            #if targetPrecision is not None:
            #    if abs(precision-targetPrecision)<0.001:
            #        break
            #    elif stepSize<minStepSize:
            #        if precision<targetPrecision:
            #            draw_rel_thresh += stepSize*2
            #            continue
            #        else:
            #            break
            #    elif precision<targetPrecision:
            #        draw_rel_thresh += stepSize
            #        if not wasTooSmall:
            #            reverse=True
            #            wasTooSmall=True
            #        else:
            #            reverse=False
            #    else:
            #        draw_rel_thresh -= stepSize
            #        if wasTooSmall:
            #            reverse=True
            #            wasTooSmall=False
            #        else:
            #            reverse=False
            #    if reverse:
            #        stepSize *= 0.5
            #else:
            #    break


            #import pdb;pdb.set_trace()

            #for b in range(len(outputBoxes)):
            
            
            dists=defaultdict(list)
            dists_x=defaultdict(list)
            dists_y=defaultdict(list)
            scaleDiffs=defaultdict(list)
            rotDiffs=defaultdict(list)
            b=0
            #print('image {} has {} {}'.format(startIndex+b,targetBoxesSizes[name][b],name))
            #bbImage = np.ones_like(image):w

    if outDir is not None:
        outputBoxes = outputBoxes.data.numpy()
        data = data.numpy()

        image = (1-((1+np.transpose(data[b][:,:,:],(1,2,0)))/2.0)).copy()
        if image.shape[2]==1:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        #if name=='text_start_gt':

        #Draw GT bbs
        if not pretty:
            for j in range(targetSize):
                plotRect(image,(1,0.5,0),targetBoxes[0,j,0:5])
            #x=int(targetBoxes[b,j,0])
            #y=int(targetBoxes[b,j,1]+targetBoxes[b,j,3])
            #cv2.putText(image,'{:.2f}'.format(target_num_neighbors[b,j]),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0.6,0.3,0),2,cv2.LINE_AA)
            #if alignmentBBs[b] is not None:
            #    aj=alignmentBBs[b][j]
            #    xc_gt = targetBoxes[b,j,0]
            #    yc_gt = targetBoxes[b,j,1]
            #    xc=outputBoxes[b,aj,1]
            #    yc=outputBoxes[b,aj,2]
            #    cv2.line(image,(xc,yc),(xc_gt,yc_gt),(0,1,0),1)
            #    shade = 0.0+(outputBoxes[b,aj,0]-threshConf)/(maxConf-threshConf)
            #    shade = max(0,shade)
            #    if outputBoxes[b,aj,6] > outputBoxes[b,aj,7]:
            #        color=(0,shade,shade) #text
            #    else:
            #        color=(shade,shade,0) #field
            #    plotRect(image,color,outputBoxes[b,aj,1:6])

        #bbs=[]
        #pred_points=[]
        #maxConf = outputBoxes[b,:,0].max()
        #threshConf = 0.5 
        #threshConf = max(maxConf*0.9,0.5)
        #print("threshConf:{}".format(threshConf))
        #for j in range(outputBoxes.shape[1]):
        #    conf = outputBoxes[b,j,0]
        #    if conf>threshConf:
        #        bbs.append((conf,j))
        #    #pred_points.append(
        #bbs.sort(key=lambda a: a[0]) #so most confident bbs are draw last (on top)
        #import pdb; pdb.set_trace()

        #Draw pred bbs
        bbs = outputBoxes
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
                if bbs[j,6+extraPreds] > bbs[j,7+extraPreds]:
                    color=(0,0,shade) #text
                else:
                    color=(0,shade,shade) #field
                if pretty=='light':
                    lineWidth=2
                else:
                    lineWidth=1
                plotRect(image,color,bbs[j,1:6],lineWidth)

                if predNN is not None and not pretty: #model.detector.predNumNeighbors:
                    x=int(bbs[j,1])
                    y=int(bbs[j,2])#-bbs[j,4])
                    targ_j = bbAlignment[j].item()
                    if targ_j>=0:
                        gtNN = target_num_neighbors[0,targ_j].item()
                    else:
                        gtNN = 0
                    pred_nn = predNN[j].item()
                    color = min(abs(pred_nn-gtNN),1)#*0.5
                    cv2.putText(image,'{:.2}/{}'.format(pred_nn,gtNN),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(color,0,0),2,cv2.LINE_AA)

        #for j in alignmentBBsTarg[name][b]:
        #    p1 = (targetBoxes[name][b,j,0], targetBoxes[name][b,j,1])
        #    p2 = (targetBoxes[name][b,j,0], targetBoxes[name][b,j,1])
        #    mid = ( int(round((p1[0]+p2[0])/2.0)), int(round((p1[1]+p2[1])/2.0)) )
        #    rad = round(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)/2.0)
        #    #print(mid)
        #    #print(rad)
        #    cv2.circle(image,mid,rad,(1,0,1),1)

        draw_rel_thresh = relPred.max() * draw_rel_thresh


        #Draw pred pairings
        numrelpred=0
        hits = [False]*len(adjacency)
        for i in range(len(relCand)):
            #print('{},{} : {}'.format(relCand[i][0],relCand[i][1],relPred[i]))
            if pretty:
                if relPred[i]>0 or pretty=='light':
                    score = relPred[i]
                    pruned=False
                    lineWidth=2
                else:
                    score = relPred[i]+1
                    pruned=True
                    lineWidth=1
                #else:
                #    score = (relPred[i]+1)/2
                #    pruned=False
                #    lineWidth=2
                #if pretty=='light':
                #    lineWidth=3
            else:
                lineWidth=1
            if relPred[i]>draw_rel_thresh or (pretty and score>draw_rel_thresh):
                ind1 = relCand[i][0]
                ind2 = relCand[i][1]
                x1 = round(bbs[ind1,1])
                y1 = round(bbs[ind1,2])
                x2 = round(bbs[ind2,1])
                y2 = round(bbs[ind2,2])

                if pretty:
                    targ1 = bbAlignment[ind1].item()
                    targ2 = bbAlignment[ind2].item()
                    aId=None
                    if bbFullHit[ind1] and bbFullHit[ind2]:
                        if (targ1,targ2) in adjacency:
                            aId = adjacency.index((targ1,targ2))
                        elif (targ2,targ1) in adjacency:
                            aId = adjacency.index((targ2,targ1))
                    if aId is None:
                        if pretty=='clean' and pruned:
                            color=np.array([1,1,0])
                        else:
                            color=np.array([1,0,0])
                    else:
                        if pretty=='clean' and pruned:
                            color=np.array([1,0,1])
                        else:
                            color=np.array([0,1,0])
                        hits[aId]=True
                    #if pruned:
                    #    color = color*0.7
                    cv2.line(image,(x1,y1),(x2,y2),color.tolist(),lineWidth)
                    #color=color/3
                    #x = int((x1+x2)/2)
                    #y = int((y1+y2)/2)
                    #if pruned:
                    #    cv2.putText(image,'[{:.2}]'.format(score),(x,y), cv2.FONT_HERSHEY_PLAIN, 0.6,color.tolist(),1)
                    #else:
                    #    cv2.putText(image,'{:.2}'.format(score),(x,y), cv2.FONT_HERSHEY_PLAIN,1.1,color.tolist(),1)
                else:
                    shade = (relPred[i].item()-draw_rel_thresh)/(1-draw_rel_thresh)

                    #print('draw {} {} {} {} '.format(x1,y1,x2,y2))
                    cv2.line(image,(x1,y1),(x2,y2),(0,shade,0),lineWidth)
                numrelpred+=1
        if pretty and pretty!="light" and pretty!="clean":
            for i in range(len(relCand)):
                #print('{},{} : {}'.format(relCand[i][0],relCand[i][1],relPred[i]))
                if relPred[i]>-1:
                    score = (relPred[i]+1)/2
                    pruned=False
                else:
                    score = (relPred[i]+2+1)/2
                    pruned=True
                if relPred[i]>draw_rel_thresh or (pretty and score>draw_rel_thresh):
                    ind1 = relCand[i][0]
                    ind2 = relCand[i][1]
                    x1 = round(bbs[ind1,1])
                    y1 = round(bbs[ind1,2])
                    x2 = round(bbs[ind2,1])
                    y2 = round(bbs[ind2,2])

                    targ1 = bbAlignment[ind1].item()
                    targ2 = bbAlignment[ind2].item()
                    aId=None
                    if bbFullHit[ind1] and bbFullHit[ind2]:
                        if (targ1,targ2) in adjacency:
                            aId = adjacency.index((targ1,targ2))
                        elif (targ2,targ1) in adjacency:
                            aId = adjacency.index((targ2,targ1))
                    if aId is None:
                        color=np.array([1,0,0])
                    else:
                        color=np.array([0,1,0])
                    color=color/2
                    x = int((x1+x2)/2)
                    y = int((y1+y2)/2)
                    if pruned:
                        cv2.putText(image,'[{:.2}]'.format(score),(x,y), cv2.FONT_HERSHEY_PLAIN, 0.6,color.tolist(),1)
                    else:
                        cv2.putText(image,'{:.2}'.format(score),(x,y), cv2.FONT_HERSHEY_PLAIN,1.1,color.tolist(),1)
        #print('number of pred rels: {}'.format(numrelpred))
        #Draw GT pairings
        if not pretty:
            gtcolor=(0.25,0,0.25)
            wth=3
        else:
            #gtcolor=(1,0,0.6)
            gtcolor=(1,0.6,0)
            wth=2
        for aId,(i,j) in enumerate(adjacency):
            if not pretty or not hits[aId]:
                x1 = round(targetBoxes[0,i,0].item())
                y1 = round(targetBoxes[0,i,1].item())
                x2 = round(targetBoxes[0,j,0].item())
                y2 = round(targetBoxes[0,j,1].item())
                cv2.line(image,(x1,y1),(x2,y2),gtcolor,wth)

        #Draw alginment between gt and pred bbs
        if not pretty:
            for predI in range(bbs.shape[0]):
                targI=bbAlignment[predI].item()
                x1 = int(round(bbs[predI,1]))
                y1 = int(round(bbs[predI,2]))
                if targI>0:

                    x2 = round(targetBoxes[0,targI,0].item())
                    y2 = round(targetBoxes[0,targI,1].item())
                    cv2.line(image,(x1,y1),(x2,y2),(1,0,1),1)
                else:
                    #draw 'x', indicating not match
                    cv2.line(image,(x1-5,y1-5),(x1+5,y1+5),(.1,0,.1),1)
                    cv2.line(image,(x1+5,y1-5),(x1-5,y1+5),(.1,0,.1),1)



        saveName = '{}_boxes_prec:{:.2f},{:.2f}_recall:{:.2f},{:.2f}_rels_AP:{:.3f}'.format(imageName,prec_5[0],prec_5[1],recall_5[0],recall_5[1],rel_ap)
        #for j in range(metricsOut.shape[1]):
        #    saveName+='_m:{0:.3f}'.format(metricsOut[i,j])
        saveName+='.png'
        io.imsave(os.path.join(outDir,saveName),image)
        #print('saved: '+os.path.join(outDir,saveName))

    print('\n{} ap:{}\tnumMissedByDetect:{}\tmissedByHuer:{}'.format(imageName,rel_ap,numMissedByDetect,numMissedByHeur))
    retData= { 'bb_ap':[ap_5],
               'bb_recall':[recall_5],
               'bb_prec':[prec_5],
               'bb_Fm': -1,#(recall_5[0]+recall_5[1]+prec_5[0]+prec_5[1])/4,
               'nn_loss': nn_loss,
               'rel_recall':relRecall,
               'rel_precision':precision,
               'rel_Fm':2*relRecall*precision/(relRecall+precision) if relRecall+precision>0 else 0,
               'relMissedByHeur':numMissedByHeur,
               'relMissedByDetect':numMissedByDetect,
               'heurRecall': heurRecall,
               'detectRecall': detectRecall,
               **toRet

             }
    if rel_ap is not None: #none ap if no relationships
        retData['rel_AP']=rel_ap
        retData['no_targs']=0
    else:
        retData['no_targs']=1
    if model.predNN:
        retData['nn_loss_final']=nn_loss_final
        retData['nn_loss_diff']=nn_loss_final-nn_loss
        retData['nn_acc_final'] = nn_acc
    if model.detector.predNumNeighbors and not useDetections:
        retData['nn_acc_detector'] = nn_acc_d
    if model.predClass:
        retData['class_loss_final']=class_loss_final
        retData['class_loss_diff']=class_loss_final-class_loss
    return (
             retData,
             (lossThis, position_loss, conf_loss, class_loss, recall, precision)
            )


