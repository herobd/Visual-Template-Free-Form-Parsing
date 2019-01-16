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
from .graph_pair import GraphPairDataset

def collate(batch):
    assert(len(batch)==1)
    return batch[0]


class AI2DGraphPair(GraphPairDataset):
    """
    Class for reading AI2D dataset and creating query/result masks from bounding polygons
    """

    def getResponseBBIdList(self,queryId,annotations):
        responsePolyList=[]
        for relId in annotations['relationships']:
            if queryId in relId:
                #print('query: '+queryId)
                #print('rel:   '+relId)
                pos = relId.find(queryId)
                if pos+len(queryId)<len(relId) and relId[pos+len(queryId)]!='+': #ensure 'B1' doesnt match 'B10'
                    continue
                #only the objects listed immediatley before or after this one are important
                if pos>0 and relId[pos-1]=='+':
                    nextPlus = relId.rfind('+',0,pos-1)
                    #print('nextP: '+str(nextPlus))
                    neighborId = relId[nextPlus+1:pos-1]
                    #print('neBe:  '+neighborId)
                    poly = self.__getResponsePoly(neighborId,annotations)
                    if poly is not None:
                        #responsePolyList.append(poly)
                        responsePolyList.append(neighborId)
                if pos+len(queryId)+1<len(relId) and relId[pos+len(queryId)]=='+':
                    nextPlus = relId.find('+',pos+len(queryId)+1)
                    if nextPlus==-1:
                        neighborId=relId[pos+len(queryId)+1:]
                        #print('neAf1: '+neighborId)
                    else:
                        neighborId=relId[pos+len(queryId)+1:nextPlus]
                        #print('neAf2: '+neighborId)
                    poly = self.__getResponsePoly(neighborId,annotations)
                    if poly is not None:
                        #responsePolyList.append(poly)
                        responsePolyList.append(neighborId)
        return responsePolyList

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
        super(AI2DGraphPair, self).__init__(dirPath,split,config,images)
        if images is not None:
            self.images=images
        else:
            with open(os.path.join(dirPath,'categories.json')) as f:
                imageToCategories = json.loads(f.read())
            with open(os.path.join(dirPath,'traintestplit_categories.json')) as f:
                #if split=='valid' or split=='validation':
                #    trainTest='train'
                #else:
                #    trainTest=split
                categoriesToUse = json.loads(f.read())[split]
            self.images=[]
            if test:
                aH=0
                aW=0
                aA=0
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


    def parseAnn(self,annotations,scale):
        ids=[]
        bbs=[]
        for blobId, blob in annotations['blobs'].items():
            ids.append(blobId)
            bbs.append( self.transformBB(blob,scale,0) )
        for arrowId, arrow in annotations['arrows'].items():
            ids.append(arrowId)
            bbs.append( self.transformBB(arrow,scale,1) )
        for textId, text in annotations['text'].items():
            ids.append(textId)
            bbs.append( self.transformBB(text,scale,2) )
        
        bbs = np.array(bbs)[None,:,:] #add batch dim on front
        return bbs,ids,3


    def transformBB(self,item,scale,classNum):
        #This has no rotation
        if classNum==2:
            rect = item['rectangle']
            poly = [ rect[0], [rect[1][0],rect[0][1]], rect[1], [rect[0][0],rect[1][1]] ]
        else:
            poly  = item['polygon']
        tlX = blX = min([p[0] for p in poly])
        trX = brX = max([p[0] for p in poly])
        tlY = trY = min([p[1] for p in poly])
        blY = brY = max([p[1] for p in poly])


        bb=[0]*(8+8+3) #2x4 corners, 2x4 cross-points, 3 classes

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

        return bb
