import torch
import torch.utils.data
import numpy as np
import json
from skimage import io
from skimage import draw
import skimage.transform as sktransform
import os
import math
from utils.util import get_image_size
from .box_detect import BoxDetectDataset, collate
import random


class AI2DBoxDetect(BoxDetectDataset):
    """
    Class for reading AI2D dataset and creating bb gt for detection
    """

    def __getResponsePoly(self, neighborId,annotations):
        if neighborId[0]=='T':
            rect=annotations['text'][neighborId]['rectangle']
            poly = [ rect[0], [rect[1][0],rect[0][1]], rect[1], [rect[0][0],rect[1][1]] ]#, rect[0] ]
            return np.array(poly)
        elif neighborId[0]=='A':
            return np.array(annotations['arrows'][neighborId]['polygon'])
        elif neighborId[0]=='B':
            return np.array(annotations['blobs'][neighborId]['polygon'])
        else:
            return None

    def __init__(self, dirPath=None, split=None, config=None, images=None, test=False):
        super(AI2DBoxDetect, self).__init__(dirPath,split,config,images)
        self.only_types = {"boxes":True}
        if images is not None:
            self.images=images
        else:
            with open(os.path.join(dirPath,'categories.json')) as f:
                imageToCategories = json.loads(f.read())
            with open(os.path.join(dirPath,'traintestplit_categories.json')) as f:
                if split=='valid' or split=='validation':
                    trainTest='train'
                else:
                    trainTest=split
                categoriesToUse = json.loads(f.read())[trainTest]
            self.images=[]
            for image, category in imageToCategories.items():
                if category in categoriesToUse:
                    imagePath_orig = os.path.join(dirPath,'images',image)
                    if self.cache_resized:
                        imagePath=os.path.join(self.cache_path,image)
                    else:
                        imagePath=imagePath_orig
                    jsonPath = os.path.join(dirPath,'annotationsMod',image+'.json')
                    rescale=1.0
                    if self.cache_resized:
                        rescale = self.rescale_range[1]
                        if not os.path.exists(path):
                            org_img = cv2.imread(org_path)
                            if org_img is None:
                                print('WARNING, could not read {}'.format(org_img))
                                continue
                            resized = cv2.resize(org_img,(0,0),
                                    fx=self.rescale_range[1],
                                    fy=self.rescale_range[1],
                                    interpolation = cv2.INTER_CUBIC)
                            cv2.imwrite(path,resized)
                    self.images.append({'id':image, 'imagePath':imagePath, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':image[:image.rfind('.')]})

            random.seed(a=123)
            random.shuffle(self.images)
            splitPoint = int(len(self.images)*0.1)
            if split=='valid' or split=='validation':
                self.images = self.images[:splitPoint]
            else:
                self.images = self.images[splitPoint:]


    def parseAnn(self,image,annotations,scale,imageName):
        bbs=[]
        numNeighbors=defaultdict(lambda:0)
        for blobId, blob in annotations['blobs'].items():
            bbs.append( self.transformBB(blob,scale,0) )
            responseIds = getResponseBBIdList_(self,blobId,annotations)
            numNeighbors[blobId]+=len(responseIds)
        for arrowId, arrow in annotations['arrows'].items():
            bbs.append( self.transformBB(arrow,scale,1) )
            responseIds = getResponseBBIdList_(self,arrowId,annotations)
            numNeighbors[arrowId]+=len(responseIds)
        for headId, arrow in annotations['arrowHeads'].items():
            bbs.append( self.transformBB(arrow,scale,2) )
            responseIds = getResponseBBIdList_(self,headId,annotations)
            numNeighbors[headId]+=len(responseIds)
        for textId, text in annotations['text'].items():
            bbs.append( self.transformBB(text,scale,3) )
            responseIds = getResponseBBIdList_(self,textId,annotations)
            numNeighbors[textId]+=len(responseIds)
        
        bbs = np.array(bbs)[None,:,:] #add batch dim on front



        return bbs,{},{},None,4,numNeighbors, None


    def transformBB(self,item,scale,classNum):
        #This has no rotation
        if classNum==2 or classNum==3:
            rect = item['rectangle']
            poly = [ rect[0], [rect[1][0],rect[0][1]], rect[1], [rect[0][0],rect[1][1]] ]
        else:
            poly  = item['polygon']
        tlX = blX = min([p[0] for p in poly])
        trX = brX = max([p[0] for p in poly])
        tlY = trY = min([p[1] for p in poly])
        blY = brY = max([p[1] for p in poly])


        bb=[0]*(8+8+4) #2x4 corners, 2x4 cross-points, 3 classes

        bb[0]=tlX*scale
        bb[1]=tlY*scale
        bb[2]=trX*scale
        bb[3]=trY*scale
        bb[4]=brX*scale
        bb[5]=brY*scale
        bb[6]=blX*scale
        bb[7]=blY*scale
        #we add thescalee for conveince to crop BBs within window
        bb[8]=scale*(tlX+blX)/2.0
        bb[9]=scale*(tlY+blY)/2.0
        bb[10]=scale*(trX+brX)/2.0
        bb[11]=scale*(trY+brY)/2.0
        bb[12]=scale*(tlX+trX)/2.0
        bb[13]=scale*(tlY+trY)/2.0
        bb[14]=scale*(brX+blX)/2.0
        bb[15]=scale*(brY+blY)/2.0

        #classes
        bb[16]=1 if classNum==0 else 0
        bb[17]=1 if classNum==1 else 0
        bb[18]=1 if classNum==2 else 0
        bb[19]=1 if classNum==3 else 0

        return bb
