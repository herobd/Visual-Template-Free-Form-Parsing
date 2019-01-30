from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import *
from model.graph_net import GraphNet
from model.binary_pair_net import BinaryPairNet
from model.binary_pair_real import BinaryPairReal
from model.roi_align.roi_align import RoIAlign
from skimage import draw
from model.net_builder import make_layers, getGroupSize
from utils.yolo_tools import non_max_sup_iou, non_max_sup_dist
import math
import random
import json

import timeit
import cv2

MAX_CANDIDATES=325 #450
MAX_GRAPH_SIZE=370
#max seen 428, so why'd it crash on 375?

class PairingGraph(BaseModel):
    def __init__(self, config):
        super(PairingGraph, self).__init__(config)

        if 'detector_checkpoint' in config:
            checkpoint = torch.load(config['detector_checkpoint'])
            detector_config = json.load(open(config['detector_config']))['model'] if 'detector_config' in config else checkpoint['config']['model']
            if 'state_dict' in checkpoint:
                self.detector = eval(checkpoint['config']['arch'])(detector_config)
                self.detector.load_state_dict(checkpoint['state_dict'])
            else:
                self.detector = checkpoint['model']
        else:
            detector_config = config['detector_config']
            self.detector = eval(detector_config['arch'])(detector_config)
        useBeginningOfLast = config['use_beg_det_feats'] if 'use_beg_det_feats' in config else False
        useFeatsLayer = config['use_detect_layer_feats'] if 'use_detect_layer_feats' in config else -1
        self.detector.setForGraphPairing(useBeginningOfLast,useFeatsLayer)
        if (config['start_frozen'] if 'start_frozen' in config else False):
            for param in self.detector.parameters(): 
                param.will_use_grad=param.requires_grad 
                param.requires_grad=False 
            self.detector_frozen=True
        else:
            self.detector_frozen=False


        self.numBBTypes = self.detector.numBBTypes
        self.rotation = self.detector.rotation
        self.scale = self.detector.scale
        self.anchors = self.detector.anchors
        self.confThresh = config['conf_thresh'] if 'conf_thresh' in config else 0.5

        graph_in_channels = config['graph_config']['in_channels'] if 'in_channels' in config['graph_config'] else 1
        self.useBBVisualFeats=True
        if config['graph_config']['arch'][:10]=='BinaryPair':
            self.useBBVisualFeats=False
        self.includeRelRelEdges= config['use_rel_rel_edges'] if 'use_rel_rel_edges' in config else True
        #rel_channels = config['graph_config']['rel_channels']
        self.pool_h = config['featurizer_start_h']
        self.pool_w = config['featurizer_start_w']


        if 'use_rel_shape_feats' in config:
             config['use_shape_feats'] =  config['use_rel_shape_feats']
        self.useShapeFeats= config['use_shape_feats'] if 'use_shape_feats' in config else False
        #HACK, fixed values
        self.normalizeHorz=400
        self.normalizeVert=50

        assert(self.detector.scale[0]==self.detector.scale[1])
        detect_scale = self.detector.scale[0]
        if self.useShapeFeats:
           added_feats=8+2*self.numBBTypes #we'll append some extra feats
           added_featsBB=3+self.numBBTypes
           if self.useShapeFeats!='old':
               added_feats+=4
        else:
           added_feats=0
           added_featsBB=0
        config['graph_config']['num_shape_feats']=added_feats
        featurizer_fc = config['featurizer_fc'] if 'featurizer_fc' in config else []
        if self.useShapeFeats!='only':
            self.roi_align = RoIAlign(self.pool_h,self.pool_w,1.0/detect_scale)
            self.avg_box = RoIAlign(2,3,1.0/detect_scale)

            self.expandedRelContext = config['expand_rel_context'] if 'expand_rel_context' in config else None
            if self.expandedRelContext is not None:
                bbMasks=3
            else:
                bbMasks=2

            feat_norm = detector_config['norm_type'] if 'norm_type' in detector_config else None
            featurizer_conv = config['featurizer_conv'] if 'featurizer_conv' in config else [512,'M',512]
            featurizer_conv = [self.detector.last_channels+bbMasks] + featurizer_conv #bbMasks are appended
            scaleX=1
            scaleY=1
            for a in featurizer_conv:
                if a=='M' or (type(a) is str and a[0]=='D'):
                    scaleX*=2
                    scaleY*=2
                elif type(a) is str and a[0]=='U':
                    scaleX/=2
                    scaleY/=2
                elif type(a) is str and a[0:4]=='long': #long pool
                    scaleX*=3
                    scaleY*=2
            #self.scale=(scaleX,scaleY) this holds scale for detector
            fsizeX = self.pool_w//scaleX
            fsizeY = self.pool_h//scaleY
            layers, last_ch_relC = make_layers(featurizer_conv,norm=feat_norm,dropout=True) 
            if featurizer_fc is None: #we don't have a FC layer, so channels need to be the same as graph model expects
                if last_ch_relC+added_feats!=graph_in_channels:
                    new_layer = [last_ch_relC,'k1-{}'.format(graph_in_channels-added_feats)]
                    print('WARNING: featurizer_conv did not line up with graph_in_channels, adding layer k1-{}'.format(graph_in_channels-added_feats))
                    new_layer, last_ch_relC = make_layers(new_layer,norm=feat_norm,dropout=True) 
                    layers+=new_layer
            layers.append( nn.AvgPool2d((fsizeY,fsizeX)) )
            self.relFeaturizerConv = nn.Sequential(*layers)
        else:
            last_ch_relC=0

        if config['graph_config']['arch'][:10]=='BinaryPair':
            feat_norm=None
        if featurizer_fc is not None:
            featurizer_fc = [last_ch_relC+added_feats] + featurizer_fc + ['FCnR{}'.format(graph_in_channels)]
            layers, last_ch_rel = make_layers(featurizer_fc,norm=feat_norm,dropout=True) 
            self.relFeaturizerFC = nn.Sequential(*layers)
        else:
            self.relFeaturizerFC = None

        if self.useBBVisualFeats:
            #TODO un-hardcode
            #self.bbFeaturizerConv = nn.MaxPool2d((2,3))
            #self.bbFeaturizerConv = nn.Sequential( nn.Conv2d(self.detector.last_channels,self.detector.last_channels,kernel_size=(2,3)), nn.ReLU(inplace=True) )
            featurizer = config['node_featurizer'] if 'node_featurizer' in config else []
            if len(featurizer)>0:
                convOut=featurizer[0]
            else:
                convOut=graph_in_channels-added_featsBB
            self.bbFeaturizerConv = nn.Sequential( 
                    nn.Conv2d(self.detector.last_channels,convOut,kernel_size=(2,3)),
                    #nn.GroupNorm(getGroupSize(convOut),convOut)
                    )
            assert(len(featurizer)==0)
           #if len(featurizer)>0:
            #    featurizer = featurizer + ['FCnR{}'.format(graph_in_channels)]
            #    layers, last_ch_node = make_layers(featurizer,norm=feat_norm)
            #    self.bbFeaturizerFC = nn.Sequential(*layers)
            #else:
            #    self.bbFeaturizerFC = None


        #self.pairer = GraphNet(config['graph_config'])
        self.pairer = eval(config['graph_config']['arch'])(config['graph_config'])


        if 'DEBUG' in config:
            self.detector.setDEBUG()
            self.setDEBUG()
            self.debug=True
        else:
            self.debug=False

 
    def unfreeze(self): 
        if self.detector_frozen:
            for param in self.detector.parameters(): 
                param.requires_grad=param.will_use_grad 
            self.detector_frozen=False
            print('Unfroze detector')
        

    def forward(self, image, gtBBs=None, useGTBBs=False, otherThresh=None, otherThreshIntur=None, hard_detect_limit=300):
        ##tic=timeit.default_timer()
        bbPredictions, offsetPredictions, _,_,_,_ = self.detector(image)
        _=None
        final_features=self.detector.final_features
        self.detector.final_features=None
        ##print('detector: {}'.format(timeit.default_timer()-tic))

        if final_features is None:
            print('ERROR:no final features!')
            import pdb;pdb.set_trace()

        
        ##tic=timeit.default_timer()
        maxConf = bbPredictions[:,:,0].max().item()
        if otherThreshIntur is None:
            confThreshMul = self.confThresh
        else:
            confThreshMul = self.confThresh*(1-otherThreshIntur) + otherThresh*otherThreshIntur
        threshConf = max(maxConf*confThreshMul,0.5)
        if self.rotation:
            bbPredictions = non_max_sup_dist(bbPredictions.cpu(),threshConf,2.5,hard_detect_limit)
        else:
            bbPredictions = non_max_sup_iou(bbPredictions.cpu(),threshConf,0.4,hard_detect_limit)
        #I'm assuming batch size of one
        assert(len(bbPredictions)==1)
        bbPredictions=bbPredictions[0]
        ##print('process boxes: {}'.format(timeit.default_timer()-tic))
        #bbPredictions should be switched for GT for training? Then we can easily use BCE loss. 
        #Otherwise we have to to alignment first
        if not useGTBBs:
            if bbPredictions.size(0)==0:
                return bbPredictions, offsetPredictions, None, None
            useBBs = bbPredictions[:,1:] #remove confidence score
        else:
            if gtBBs is None:
                return bbPredictions, offsetPredictions, None, None
            useBBs = gtBBs[0,:,0:5]
            if self.useShapeFeats:
                classes = gtBBs[0,:,13:]
                #pos = random.uniform(0.51,0.99)
                #neg = random.uniform(0.01,0.49)
                #classes = torch.where(classes==0,torch.tensor(neg).to(classes.device),torch.tensor(pos).to(classes.device))
                pos = torch.rand_like(classes)/2 +0.5
                neg = torch.rand_like(classes)/2
                classes = torch.where(classes==0,neg,pos)
                useBBs = torch.cat((useBBs,classes),dim=1)
        if useBBs.size(0)>1:
            #bb_features, adjacencyMatrix, rel_features = self.createGraph(useBBs,final_features)
            bbAndRel_features, adjacencyMatrix, numBBs, numRel, relIndexes = self.createGraph(useBBs,final_features,image.size(-2),image.size(-1), debug_image=None)
            if bbAndRel_features is None:
                return bbPredictions, offsetPredictions, None, None

            ##tic=timeit.default_timer()
            #nodeOuts, relOuts = self.pairer(bb_features, adjacencyMatrix, rel_features)
            bbOuts, relOuts = self.pairer(bbAndRel_features, adjacencyMatrix, numBBs)
            #bbOuts = graphOut[:numBBs]
            #relOuts = graphOut[numBBs:]
            ##print('pairer: {}'.format(timeit.default_timer()-tic))

            #adjacencyMatrix = torch.zeros((bbPredictions.size(1),bbPredictions.size(1)))
            #for rel in relOuts:
            #    i,j,a=graphToDetectionsMap(

            return bbPredictions, offsetPredictions, relOuts, relIndexes
        else:
            return bbPredictions, offsetPredictions, None, None

    def createGraph(self,bbs,features,imageHeight,imageWidth,debug_image=None):
        ##tic=timeit.default_timer()
        candidates = self.selectCandidateEdges(bbs,imageHeight,imageWidth)
        ##print('  candidate: {}'.format(timeit.default_timer()-tic))
        if len(candidates)==0:
            return None,None,None,None,None
        ##tic=timeit.default_timer()

        #stackedEdgeFeatWindows = torch.FloatTensor((len(candidates),features.size(1)+2,self.relWindowSize,self.relWindowSize)).to(features.device())

        #get corners from bb predictions
        x = bbs[:,0]
        y = bbs[:,1]
        r = bbs[:,2]
        h = bbs[:,3]
        w = bbs[:,4]
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        tlX = -w*cos_r + -h*sin_r +x
        tlY =  w*sin_r + -h*cos_r +y
        trX =  w*cos_r + -h*sin_r +x
        trY = -w*sin_r + -h*cos_r +y
        brX =  w*cos_r + h*sin_r +x
        brY = -w*sin_r + h*cos_r +y
        blX = -w*cos_r + h*sin_r +x
        blY =  w*sin_r + h*cos_r +y

        tlX = tlX.cpu()
        tlY = tlY.cpu()
        trX = trX.cpu()
        trY = trY.cpu()
        blX = blX.cpu()
        blY = blY.cpu()
        brX = brX.cpu()
        brY = brY.cpu()

        if self.useShapeFeats!='only':
            #get axis aligned rectangle from corners
            rois = torch.zeros((len(candidates),5)) #(batchIndex,x1,y1,x2,y2) as expected by ROI Align
            for i,(index1, index2) in enumerate(candidates):
                maxX = max(tlX[index1],tlX[index2],trX[index1],trX[index2],blX[index1],blX[index2],brX[index1],brX[index2])
                minX = min(tlX[index1],tlX[index2],trX[index1],trX[index2],blX[index1],blX[index2],brX[index1],brX[index2])
                maxY = max(tlY[index1],tlY[index2],trY[index1],trY[index2],blY[index1],blY[index2],brY[index1],brY[index2])
                minY = min(tlY[index1],tlY[index2],trY[index1],trY[index2],blY[index1],blY[index2],brY[index1],brY[index2])
                if self.expandedRelContext is not None:
                    maxX = min(maxX.item()+self.expandedRelContext,imageWidth-1)
                    minX = max(minX.item()-self.expandedRelContext,0)
                    maxY = min(maxY.item()+self.expandedRelContext,imageHeight-1)
                    minY = max(minY.item()-self.expandedRelContext,0)
                rois[i,1]=minX
                rois[i,2]=minY
                rois[i,3]=maxX
                rois[i,4]=maxY




                ###DEBUG
                if debug_image is not None and i<10:
                    assert(self.rotation==False)
                    #print('crop {}: ({},{}), ({},{})'.format(i,minX.item(),maxX.item(),minY.item(),maxY.item()))
                    #print(bbs[index1])
                    #print(bbs[index2])
                    crop = debug_image[0,:,int(minY):int(maxY),int(minX):int(maxX)+1].cpu()
                    crop = (2-crop)/2
                    if crop.size(0)==1:
                        crop = crop.expand(3,crop.size(1),crop.size(2))
                    crop[0,int(tlY[index1].item()-minY):int(brY[index1].item()-minY)+1,int(tlX[index1].item()-minX):int(brX[index1].item()-minX)+1]*=0.5
                    crop[1,int(tlY[index2].item()-minY):int(brY[index2].item()-minY)+1,int(tlX[index2].item()-minX):int(brX[index2].item()-minX)+1]*=0.5
                    crop = crop.numpy().transpose([1,2,0])
                    cv2.imshow('crop {}'.format(i),crop)
                    #import pdb;pdb.set_trace()
                ###
            if debug_image is not None:
                cv2.waitKey()

            #crop from feats, ROI pool
            stackedEdgeFeatWindows = self.roi_align(features,rois.to(features.device))

            #create and add masks
            if self.expandedRelContext is not None:
                #We're going to add a third mask for all bbs, which we'll precompute here
                numMasks=3
                allMasks = torch.zeros(imageHeight,imageWidth)
                for bbIdx in range(bbs.size(0)):
                    rr, cc = draw.polygon([tlY[bbIdx],trY[bbIdx],brY[bbIdx],blY[bbIdx]],[tlX[bbIdx],trX[bbIdx],brX[bbIdx],blX[bbIdx]], [self.pool_h,self.pool_w])
                    allMasks[rr,cc]=1
            else:
                numMasks=2
            masks = torch.zeros(stackedEdgeFeatWindows.size(0),numMasks,stackedEdgeFeatWindows.size(2),stackedEdgeFeatWindows.size(3))
        if self.useShapeFeats:
            if self.useShapeFeats!='old':
                numFeats=8+4
            else:
                numFeats=8
            shapeFeats = torch.FloatTensor(len(candidates),numFeats+2*self.numBBTypes)

        for i,(index1, index2) in enumerate(candidates):
            if self.useShapeFeats!='only':
                #... or make it so index1 is always to top-left one
                if random.random()<0.5 and not self.debug:
                    temp=index1
                    index1=index2
                    index2=temp
                
                #warp to roi space
                feature_w = rois[i,3]-rois[i,1] +1
                feature_h = rois[i,4]-rois[i,2] +1
                w_m = self.pool_w/feature_w
                h_m = self.pool_h/feature_h

                tlX1 = round(((tlX[index1]-rois[i,1])*w_m).item())
                trX1 = round(((trX[index1]-rois[i,1])*w_m).item())
                brX1 = round(((brX[index1]-rois[i,1])*w_m).item())
                blX1 = round(((blX[index1]-rois[i,1])*w_m).item())
                tlY1 = round(((tlY[index1]-rois[i,2])*h_m).item())
                trY1 = round(((trY[index1]-rois[i,2])*h_m).item())
                brY1 = round(((brY[index1]-rois[i,2])*h_m).item())
                blY1 = round(((blY[index1]-rois[i,2])*h_m).item())
                tlX2 = round(((tlX[index2]-rois[i,1])*w_m).item())
                trX2 = round(((trX[index2]-rois[i,1])*w_m).item())
                brX2 = round(((brX[index2]-rois[i,1])*w_m).item())
                blX2 = round(((blX[index2]-rois[i,1])*w_m).item())
                tlY2 = round(((tlY[index2]-rois[i,2])*h_m).item())
                trY2 = round(((trY[index2]-rois[i,2])*h_m).item())
                brY2 = round(((brY[index2]-rois[i,2])*h_m).item())
                blY2 = round(((blY[index2]-rois[i,2])*h_m).item())

                rr, cc = draw.polygon([tlY1,trY1,brY1,blY1],[tlX1,trX1,brX1,blX1], [self.pool_h,self.pool_w])
                masks[i,0,rr,cc]=1
                rr, cc = draw.polygon([tlY2,trY2,brY2,blY2],[tlX2,trX2,brX2,blX2], [self.pool_h,self.pool_w])
                masks[i,1,rr,cc]=1
                if self.expandedRelContext is not None:
                    cropArea = allMasks[round(rois[i,2].item()):round(rois[i,4].item())+1,round(rois[i,1].item()):round(rois[i,3].item())+1]
                    masks[i,2] = F.upsample(cropArea[None,None,...], size=(self.pool_h,self.pool_w), mode='bilinear')[0,0]
                    #masks[i,2] = cv2.resize(cropArea,(stackedEdgeFeatWindows.size(2),stackedEdgeFeatWindows.size(3)))

            if self.useShapeFeats:
                shapeFeats[i,0] = (bbs[index1,0]-bbs[index2,0])/self.normalizeHorz
                shapeFeats[i,1] = (bbs[index1,1]-bbs[index2,1])/self.normalizeVert
                shapeFeats[i,2] = bbs[index1,2]/math.pi
                shapeFeats[i,3] = bbs[index2,2]/math.pi
                shapeFeats[i,4] = bbs[index1,3]/self.normalizeVert
                shapeFeats[i,5] = bbs[index2,3]/self.normalizeVert
                shapeFeats[i,6] = bbs[index1,4]/self.normalizeHorz
                shapeFeats[i,7] = bbs[index2,4]/self.normalizeHorz
                shapeFeats[i,8:8+self.numBBTypes] = torch.sigmoid(bbs[index1,5:])
                shapeFeats[i,8+self.numBBTypes:8+self.numBBTypes+self.numBBTypes] = torch.sigmoid(bbs[index2,5:])
                if self.useShapeFeats!='old':
                    startCorners = 8+self.numBBTypes+self.numBBTypes
                    shapeFeats[i,startCorners +0] = math.sqrt( (tlX[index1]-tlX[index2])**2 + (tlY[index1]-tlY[index2])**2 )
                    shapeFeats[i,startCorners +1] = math.sqrt( (trX[index1]-trX[index2])**2 + (trY[index1]-trY[index2])**2 )
                    shapeFeats[i,startCorners +2] = math.sqrt( (blX[index1]-blX[index2])**2 + (blY[index1]-blY[index2])**2 )
                    shapeFeats[i,startCorners +3] = math.sqrt( (brX[index1]-brX[index2])**2 + (brY[index1]-brY[index2])**2 )


        if self.useShapeFeats!='only':
            stackedEdgeFeatWindows = torch.cat((stackedEdgeFeatWindows,masks.to(stackedEdgeFeatWindows.device)),dim=1)
            #import pdb; pdb.set_trace()
            relFeats = self.relFeaturizerConv(stackedEdgeFeatWindows) #preparing for graph feature size
            relFeats = relFeats.view(relFeats.size(0),relFeats.size(1))
        if self.useShapeFeats:
            if self.useShapeFeats=='only':
                relFeats = shapeFeats.to(features.device)
            else:
                relFeats = torch.cat((relFeats,shapeFeats.to(relFeats.device)),dim=1)
        if self.relFeaturizerFC is not None:
            relFeats = self.relFeaturizerFC(relFeats)
        #if self.useShapeFeats=='sp
    
        #compute features for the bounding boxes by themselves
        #This will be replaced with some type of word embedding
        if self.useBBVisualFeats:
            assert(features.size(0)==1)
            if self.useShapeFeats:
                bb_shapeFeats=torch.FloatTensor(bbs.size(0),3+self.numBBTypes)
            rois = torch.zeros((bbs.size(0),5))
            for i in range(bbs.size(0)):
                minY = round(min(tlY[i].item(),trY[i].item(),blY[i].item(),brY[i].item()))
                maxY = round(max(tlY[i].item(),trY[i].item(),blY[i].item(),brY[i].item()))
                minX = round(min(tlX[i].item(),trX[i].item(),blX[i].item(),brX[i].item()))
                maxX = round(max(tlX[i].item(),trX[i].item(),blX[i].item(),brX[i].item()))
                rois[i,1]=minX
                rois[i,2]=minY
                rois[i,3]=maxX
                rois[i,4]=maxY
                if self.useShapeFeats:
                    bb_shapeFeats[i,0]= (bbs[i,2]+math.pi)/(2*math.pi)
                    bb_shapeFeats[i,1]=bbs[i,3]/self.normalizeVert
                    bb_shapeFeats[i,2]=bbs[i,4]/self.normalizeHorz
                    bb_shapeFeats[i,3:]=torch.sigmoid(bbs[i,5:])
                #bb_features[i]= F.avg_pool2d(features[0,:,minY:maxY+1,minX:maxX+1], (1+maxY-minY,1+maxX-minX)).view(-1)
            bb_features = self.avg_box(features,rois.to(features.device))
            bb_features = self.bbFeaturizerConv(bb_features)
            bb_features = bb_features.view(bb_features.size(0),bb_features.size(1))
            if self.useShapeFeats:
                bb_features = torch.cat( (bb_features,bb_shapeFeats.to(bb_features.device)), dim=1 )
            #bb_features = self.bbFeaturizerFC(bb_features) if uncommented, change rot on bb_shapeFeats
        else:
            bb_features = None
        
        #We're not adding diagonal (self-rels) here!
        #Expecting special handeling during graph conv
        #candidateLocs = torch.LongTensor(candidates).t().to(relFeats.device)
        #ones = torch.ones(len(candidates)).to(relFeats.device)
        #adjacencyMatrix = torch.sparse.FloatTensor(candidateLocs,ones,torch.Size([bbs.size(0),bbs.size(0)]))

        #assert(relFeats.requries_grad)
        #rel_features = torch.sparse.FloatTensor(candidateLocs,relFeats,torch.Size([bbs.size(0),bbs.size(0),relFeats.size(1)]))
        #assert(rel_features.requries_grad)
        relIndexes=candidates
        numBB = bbs.size(0)
        numRel = len(candidates)
        if bb_features is None:
            numBB=0
            bbAndRel_features=relFeats
            adjacencyMatrix = None
            numOfNeighbors = None
        else:
            bbAndRel_features = torch.cat((bb_features,relFeats),dim=0)
            numOfNeighbors = torch.ones(bbs.size(0)+len(candidates)) #starts at one for yourself
            edges=[]
            i=0
            for bb1,bb2 in candidates:
                edges.append( (bb1,numBB+i) )
                edges.append( (bb2,numBB+i) )
                numOfNeighbors[bb1]+=1
                numOfNeighbors[bb2]+=1
                numOfNeighbors[numBB+i]+=2
                i+=1
            if self.includeRelRelEdges:
                relEdges=set()
                i=0
                for bb1,bb2 in candidates:
                    j=0
                    for bbA,bbB in candidates[i:]:
                        if i!=j and bb1==bbA or bb1==bbB or bb2==bbA or bb2==bbB:
                            relEdges.add( (numBB+i,numBB+j) ) #i<j always
                        j+=1   
                    i+=1
                relEdges = list(relEdges)
                for r1, r2 in relEdges:
                    numOfNeighbors[r1]+=1
                    numOfNeighbors[r2]+=1
                edges += relEdges
            #add reverse edges
            edges+=[(y,x) for x,y in edges]
            #add diagonal (self edges)
            for i in range(bbAndRel_features.size(0)):
                edges.append((i,i))

            edgeLocs = torch.LongTensor(edges).t().to(relFeats.device)
            ones = torch.ones(len(edges)).to(relFeats.device)
            adjacencyMatrix = torch.sparse.FloatTensor(edgeLocs,ones,torch.Size([bbAndRel_features.size(0),bbAndRel_features.size(0)]))
            #numOfNeighbors is for convienence in tracking the normalization term
            numOfNeighbors=numOfNeighbors.to(relFeats.device)

        #rel_features = (candidates,relFeats)
        #adjacencyMatrix = None
        ##print('create graph: {}'.format(timeit.default_timer()-tic))
        #return bb_features, adjacencyMatrix, rel_features
        return bbAndRel_features, (adjacencyMatrix,numOfNeighbors), numBB, numRel, relIndexes



    def selectCandidateEdges(self,bbs,imageHeight,imageWidth):
        if bbs.size(0)<2:
            return []
        #return list of index pairs


        sin_r = torch.sin(bbs[:,2])
        cos_r = torch.cos(bbs[:,2])
        #lx = bbs[:,0] - cos_r*bbs[:,4] 
        #ly = bbs[:,1] + sin_r*bbs[:,3]
        #rx = bbs[:,0] + cos_r*bbs[:,4] 
        #ry = bbs[:,1] - sin_r*bbs[:,3]
        #tx = bbs[:,0] - cos_r*bbs[:,4] 
        #ty = bbs[:,1] - sin_r*bbs[:,3]
        #bx = bbs[:,0] + cos_r*bbs[:,4] 
        #by = bbs[:,1] + sin_r*bbs[:,3]
        brX = bbs[:,4]*cos_r-bbs[:,3]*sin_r + bbs[:,0] 
        brY = bbs[:,4]*sin_r+bbs[:,3]*cos_r + bbs[:,1] 
        blX = -bbs[:,4]*cos_r-bbs[:,3]*sin_r + bbs[:,0]
        blY= -bbs[:,4]*sin_r+bbs[:,3]*cos_r + bbs[:,1] 
        trX = bbs[:,4]*cos_r+bbs[:,3]*sin_r + bbs[:,0] 
        trY = bbs[:,4]*sin_r-bbs[:,3]*cos_r + bbs[:,1] 
        tlX = -bbs[:,4]*cos_r+bbs[:,3]*sin_r + bbs[:,0]
        tlY = -bbs[:,4]*sin_r-bbs[:,3]*cos_r + bbs[:,1] 

        minX = min( torch.min(trX), torch.min(tlX), torch.min(blX), torch.min(brX) )
        minY = min( torch.min(trY), torch.min(tlY), torch.min(blY), torch.min(brY) )
        maxX = max( torch.max(trX), torch.max(tlX), torch.max(blX), torch.max(brX) )
        maxY = max( torch.max(trY), torch.max(tlY), torch.max(blY), torch.max(brY) )
        #if (math.isinf(minX) or math.isinf(minY) or math.isinf(maxX) or math.isinf(maxY) ):
        #    import pdb;pdb.set_trace()
        minX = min(max(minX.item(),0),imageWidth)
        minY = min(max(minY.item(),0),imageHeight)
        maxX = min(max(maxX.item(),0),imageWidth)
        maxY = min(max(maxY.item(),0),imageHeight)
        if minX>=maxX or minY>=maxY:
            return []

        #lx-=minX 
        #ly-=minY 
        #rx-=minX 
        #ry-=minY 
        #tx-=minX 
        #ty-=minY 
        #bx-=minX 
        #by-=minY 
        zeros = torch.zeros_like(trX)
        tImageWidth = torch.ones_like(trX)*imageWidth
        tImageHeight = torch.ones_like(trX)*imageHeight
        trX = torch.min(torch.max(trX,zeros),tImageWidth)
        trY = torch.min(torch.max(trY,zeros),tImageHeight)
        tlX = torch.min(torch.max(tlX,zeros),tImageWidth)
        tlY = torch.min(torch.max(tlY,zeros),tImageHeight)
        brX = torch.min(torch.max(brX,zeros),tImageWidth)
        brY = torch.min(torch.max(brY,zeros),tImageHeight)
        blX = torch.min(torch.max(blX,zeros),tImageWidth)
        blY = torch.min(torch.max(blY,zeros),tImageHeight)
        trX-=minX
        trY-=minY
        tlX-=minX
        tlY-=minY
        brX-=minX
        brY-=minY
        blX-=minX
        blY-=minY

        scaleCand = 0.5
        minX*=scaleCand
        minY*=scaleCand
        maxX*=scaleCand
        maxY*=scaleCand
        #lx  *=scaleCand
        #ly  *=scaleCand
        #rx  *=scaleCand
        #ry  *=scaleCand
        #tx  *=scaleCand
        #ty  *=scaleCand
        #bx  *=scaleCand
        #by  *=scaleCand
        trX *=scaleCand
        trY *=scaleCand
        tlX *=scaleCand
        tlY *=scaleCand
        brX *=scaleCand
        brY *=scaleCand
        blX *=scaleCand
        blY *=scaleCand
        h = bbs[:,3]*scaleCand
        w = bbs[:,4]*scaleCand
        r = bbs[:,2]

        distMul=1.0
        while distMul>0.03:

            boxesDrawn = np.zeros( (math.ceil(maxY-minY),math.ceil(maxX-minX)) ,dtype=int)#torch.IntTensor( (maxY-minY,maxX-minX) ).zero_()
            if boxesDrawn.shape[0]==0 or boxesDrawn.shape[1]==0:
                return []
            #import pdb;pdb.set_trace()
            numBoxes = bbs.size(0)
            for i in range(numBoxes):
                
                #cv2.line( boxesDrawn, (int(tlX[i]),int(tlY[i])),(int(trX[i]),int(trY[i])),i,1)
                #cv2.line( boxesDrawn, (int(trX[i]),int(trY[i])),(int(brX[i]),int(brY[i])),i,1)
                #cv2.line( boxesDrawn, (int(blX[i]),int(blY[i])),(int(brX[i]),int(brY[i])),i,1)
                #cv2.line( boxesDrawn, (int(blX[i]),int(blY[i])),(int(tlX[i]),int(tlY[i])),i,1)

                rr,cc = draw.polygon_perimeter([int(tlY[i]),int(trY[i]),int(brY[i]),int(blY[i])],[int(tlX[i]),int(trX[i]),int(brX[i]),int(blX[i])],boxesDrawn.shape,True)
                boxesDrawn[rr,cc]=i+1

            #how to walk?
            #walk until number found.
            # if in list, end
            # else add to list, continue
            #list is candidates
            maxDist = 600*scaleCand*distMul
            maxDistY = 200*scaleCand*distMul
            minWidth=30
            minHeight=20
            numFan=5
            
            def pathWalk(myId,startX,startY,angle,distStart=0,splitDist=100):
                hit=set()
                lineId = myId+numBoxes
                if angle<-180:
                    angle+=360
                if angle>180:
                    angle-=360
                if (angle>45 and angle<135) or (angle>-135 and angle<-45):
                    #compute slope based on y stepa
                    yStep=-1
                    #if angle==90 or angle==-90:

                    xStep=1/math.tan(math.pi*angle/180.0)
                else:
                    #compute slope based on x step
                    xStep=1
                    yStep=-math.tan(math.pi*angle/180.0)
                if angle>=135 or angle<-45:
                    xStep*=-1
                    yStep*=-1
                distSoFar=distStart
                prev=0
                numSteps=0
                y=startY
                while distSoFar<maxDist and abs(y-startY)<maxDistY:
                    x=int(round(startX + numSteps*xStep))
                    y=int(round(startY + numSteps*yStep))
                    numSteps+=1
                    if x<0 or y<0 or x>=boxesDrawn.shape[1] or y>=boxesDrawn.shape[0]:
                        break
                    here = boxesDrawn[y,x]
                    #print('{} {} {} : {}'.format(x,y,here,len(hit)))
                    if here>0 and here<=numBoxes and here!=myId:
                        if here in hit and prev!=here:
                            break
                        else:
                            hit.add(here)
                            #print('hit {} at {}, {}  ({})'.format(here,x,y,len(hit)))
                            #elif here == lineId or here == myId:
                            #break
                    else:
                        boxesDrawn[y,x]=lineId
                    prev=here
                    distSoFar= distStart+math.sqrt((x-startX)**2 + (y-startY)**2)

                    #if hitting and maxDist-distSoFar>splitMin and (distSoFar-distStart)>splitDist and len(toSplit)==0:
                    #    #split
                    #    toSplit.append((myId,x,y,angle+45,distSoFar,hit.copy(),splitDist*1.5))
                    #    toSplit.append((myId,x,y,angle-45,distSoFar,hit.copy(),splitDist*1.5))

                return hit

            def fan(boxId,x,y,angle,num,hit):
                deg = 90/(num+1)
                curDeg = angle-45+deg
                for i in range(num):
                    hit.update( pathWalk(boxId,x,y,curDeg) )
                    curDeg+=deg

            def drawIt():
                x = bbs[:,0]*scaleCand - minX
                y = bbs[:,1]*scaleCand - minY
                drawn = np.zeros( (math.ceil(maxY-minY),math.ceil(maxX-minX),3))#torch.IntTensor( (maxY-minY,maxX-minX) ).zero_()
                numBoxes = bbs.size(0)
                for a,b in candidates:
                    cv2.line( drawn, (int(x[a]),int(y[a])),(int(x[b]),int(y[b])),(random.random()*0.5,random.random()*0.5,random.random()*0.5),1)
                for i in range(numBoxes):
                    
                    #cv2.line( boxesDrawn, (int(tlX[i]),int(tlY[i])),(int(trX[i]),int(trY[i])),i,1)
                    #cv2.line( boxesDrawn, (int(trX[i]),int(trY[i])),(int(brX[i]),int(brY[i])),i,1)
                    #cv2.line( boxesDrawn, (int(blX[i]),int(blY[i])),(int(brX[i]),int(brY[i])),i,1)
                    #cv2.line( boxesDrawn, (int(blX[i]),int(blY[i])),(int(tlX[i]),int(tlY[i])),i,1)

                    rr,cc = draw.polygon_perimeter([int(tlY[i]),int(trY[i]),int(brY[i]),int(blY[i])],[int(tlX[i]),int(trX[i]),int(brX[i]),int(blX[i])])
                    drawn[rr,cc]=(random.random()*0.8+.2,random.random()*0.8+.2,random.random()*0.8+.2)
                cv2.imshow('res',drawn)
                #cv2.waitKey()

                rows,cols=boxesDrawn.shape
                colorMap = [(0,0,0)]
                for i in range(numBoxes):
                    colorMap.append((random.random()*0.8+.2,random.random()*0.8+.2,random.random()*0.8+.2))
                for i in range(numBoxes):
                    colorMap.append( (colorMap[i+1][0]/3,colorMap[i+1][1]/3,colorMap[i+1][2]/3) )
                draw2 = np.zeros((rows,cols,3))
                for r in range(rows):
                    for c in range(cols):
                        draw2[r,c] = colorMap[int(round(boxesDrawn[r,c]))]
                        #draw[r,c] = (255,255,255) if boxesDrawn[r,c]>0 else (0,0,0)

                cv2.imshow('d',draw2)
                cv2.waitKey()


            candidates=set()
            for i in range(numBoxes):
                boxId=i+1
                toSplit=[]
                hit = set()

                horzDiv = 1+math.ceil(w[i]/minWidth)
                vertDiv = 1+math.ceil(h[i]/minHeight)

                if horzDiv==1:
                    leftW=0.5
                    rightW=0.5
                    hit.update( pathWalk(boxId, tlX[i].item()*leftW+trX[i].item()*rightW, tlY[i].item()*leftW+trY[i].item()*rightW,r[i].item()+90) )
                    hit.update( pathWalk(boxId, tlX[i].item()*leftW+trX[i].item()*rightW, tlY[i].item()*leftW+trY[i].item()*rightW,r[i].item()-90) )
                else:
                    for j in range(horzDiv):
                        leftW = 1-j/(horzDiv-1)
                        rightW = j/(horzDiv-1)
                        hit.update( pathWalk(boxId, tlX[i].item()*leftW+trX[i].item()*rightW, tlY[i].item()*leftW+trY[i].item()*rightW,r[i].item()+90) )
                        hit.update( pathWalk(boxId, tlX[i].item()*leftW+trX[i].item()*rightW, tlY[i].item()*leftW+trY[i].item()*rightW,r[i].item()-90) )

                if vertDiv==1:
                    topW=0.5
                    botW=0.5
                    hit.update( pathWalk(boxId, tlX[i].item()*topW+blX[i].item()*botW, tlY[i].item()*topW+blY[i].item()*botW,r[i].item()+180) )
                    hit.update( pathWalk(boxId, trX[i].item()*topW+brX[i].item()*botW, trY[i].item()*topW+brY[i].item()*botW,r[i].item()) )
                else:
                    for j in range(vertDiv):
                        topW = 1-j/(vertDiv-1)
                        botW = j/(vertDiv-1)
                        hit.update( pathWalk(boxId, tlX[i].item()*topW+blX[i].item()*botW, tlY[i].item()*topW+blY[i].item()*botW,r[i].item()+180) )
                        hit.update( pathWalk(boxId, trX[i].item()*topW+brX[i].item()*botW, trY[i].item()*topW+brY[i].item()*botW,r[i].item()) )
                fan(boxId,tlX[i].item(),tlY[i].item(),r[i].item()+135,numFan,hit)
                fan(boxId,trX[i].item(),trY[i].item(),r[i].item()+45,numFan,hit)
                fan(boxId,blX[i].item(),blY[i].item(),r[i].item()+225,numFan,hit)
                fan(boxId,brX[i].item(),brY[i].item(),r[i].item()+315,numFan,hit)

                for jId in hit:
                    candidates.add( (min(i,jId-1),max(i,jId-1)) )
            
            #print('candidates:{} ({})'.format(len(candidates),distMul))
            #if len(candidates)>1:
            #    drawIt()
            if len(candidates)+numBoxes<MAX_GRAPH_SIZE and len(candidates)<MAX_CANDIDATES:
                return list(candidates)
            else:
                distMul*=0.75
        #This is a problem, we couldn't prune down enough
        print("ERROR: could not prune number of candidates down: {} (should be {})".format(len(candidates),MAX_GRAPH_SIZE-numBoxes))
        return list(candidates)[:MAX_GRAPH_SIZE-numBoxes]

    def setDEBUG(self):
        self.debug=True
        def save_layerConv0(module,input,output):
            self.debug_conv0=output.cpu()
        self.relFeaturizerConv[0].register_forward_hook(save_layerConv0)
        def save_layerConv1(module,input,output):
            self.debug_conv1=output.cpu()
        self.relFeaturizerConv[1].register_forward_hook(save_layerConv1)
        #def save_layerFC(module,input,output):
            #    self.debug_fc=output.cpu()
        #self.relFeaturizerConv[0].register_forward_hook(save_layerFC)
