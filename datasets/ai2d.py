import torch
import torch.utils.data
import numpy as np
import json
from skimage import io
from skimage import draw
import os
import math

class AI2D(torch.utils.data.Dataset):
    """
    Class for reading AI2D dataset and creating query/result masks from bounding polygons
    """

    def __getResponsePolyList(self,queryId,annotations):
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
                        responsePolyList.append(poly)
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
                        responsePolyList.append(poly)
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

    def __init__(self, dirPath=None, split=None, config=None, ids=None, imagePaths=None, queryPolys=None, responsePolyLists=None, test=False):
        if 'augmentation_params' in config['data_loader']:
            self.augmentation_params=config['data_loader']['augmentation_params']
        else:
            self.augmentation_params=None
        if imagePaths is not None:
            self.ids=ids
            self.imagePaths=imagePaths
            self.queryPolys=queryPolys
            self.responsePolyLists=responsePolyLists
        else:
            with open(os.path.join(dirPath,'categories.json')) as f:
                imageToCategories = json.loads(f.read())
            with open(os.path.join(dirPath,'traintestplit_categories.json')) as f:
                #if split=='valid' or split=='validation':
                #    trainTest='train'
                #else:
                #    trainTest=split
                categoriesToUse = json.loads(f.read())[split]
            self.imagePaths=[]
            self.queryPolys=[]
            self.responsePolyLists=[]
            self.ids=[]
            self.helperStats=[]
            if test:
                aH=0
                aW=0
                aA=0
            for image, category in imageToCategories.items():
                if test:
                    im = io.imread(os.path.join(dirPath,'images',image))
                    aH+=im.shape[0]
                    aW+=im.shape[1]
                    aA+=im.shape[0]*im.shape[1]
                if category in categoriesToUse:
                    with open(os.path.join(dirPath,'annotations',image+'.json')) as f:
                        annotations = json.loads(f.read())
                    for blobId, blob in annotations['blobs'].items():
                        self.ids.append(blobId)
                        self.imagePaths.append(os.path.join(dirPath,'images',image))
                        self.queryPolys.append(np.array(blob['polygon']))
                        self.responsePolyLists.append(self.__getResponsePolyList(blobId,annotations))

                        self.helperStats.append(self.__getHelperStats(self.queryPolys[-1], self.responsePolyLists[-1]))

                    for arrowId, arrow in annotations['arrows'].items():
                        self.ids.append(arrowId)
                        self.imagePaths.append(os.path.join(dirPath,'images',image))
                        self.queryPolys.append(np.array(arrow['polygon']))
                        self.responsePolyLists.append(self.__getResponsePolyList(arrowId,annotations))

                        self.helperStats.append(self.__getHelperStats(self.queryPolys[-1], self.responsePolyLists[-1]))

                    for textId, text in annotations['text'].items():
                        self.ids.append(textId)
                        self.imagePaths.append(os.path.join(dirPath,'images',image))
                        rect=text['rectangle']
                        self.queryPolys.append(np.array([ rect[0], [rect[1][0],rect[0][1]], rect[1], [rect[0][0],rect[1][1]] ]))#, rect[0] ]))
                        self.responsePolyLists.append(self.__getResponsePolyList(textId,annotations))

                        self.helperStats.append(self.__getHelperStats(self.queryPolys[-1], self.responsePolyLists[-1]))
        
        if test:
            print('average height: '+str(aH/len(imageToCategories)))
            print('average width:  '+str(aW/len(imageToCategories)))
            print('average area:   '+str(aA/len(imageToCategories)))



    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self,index):
        #print(index)
        #print(self.imagePaths[index])
        #print(self.ids[index])
        image = io.imread(self.imagePaths[index])/255.0
        #TODO color jitter, rotation?, skew?
        queryMask = np.zeros([image.shape[0],image.shape[1]])
        rr, cc = draw.polygon(self.queryPolys[index][:, 1], self.queryPolys[index][:, 0], queryMask.shape)
        queryMask[rr,cc]=1
        responseMask = np.zeros([image.shape[0],image.shape[1]])
        for poly in self.responsePolyLists[index]:
            rr, cc = draw.polygon(poly[:, 1], poly[:, 0], responseMask.shape)
            responseMask[rr,cc]=1

        imageWithQuery = np.append(image,queryMask.reshape(queryMask.shape+(1,)),axis=2)
        imageWithQuery = np.moveaxis(imageWithQuery,2,0)

        sample = (imageWithQuery, responseMask,) + self.helperStats[index] + (self.imagePaths[index]+' '+self.ids[index],)
        if self.augmentation_params is not None:
            sample = self.augment(sample)
        return sample

    def splitValidation(self, config):
        validation_split = config['validation']['validation_split']
        split = int(len(self) * validation_split)
        perm = np.random.permutation(len(self))
        ids = [self.ids[x] for x in perm]
        images = [self.imagePaths[x] for x in perm]
        queryPolys = [self.queryPolys[x] for x in perm]
        responsePolyLists = [self.responsePolyLists[x] for x in perm]

        self.ids=ids[split:]
        self.imagePaths=images[split:]
        self.queryPolys=queryPolys[split:]
        self.responsePolyLists=responsePolyLists[split:]

        return AI2D(config=config, ids=ids[:split], imagePaths=images[:split], queryPolys=queryPolys[:split], responsePolyLists=responsePolyLists[:split])

    def __getHelperStats(self, queryPoly, polyList):
        """
        This returns stats used when putting a batch together, croping and resizeing windows.
        It returns
            the centerpoint of the query mask,
            the furthest response mask point from the center (minimum set by query mask size in case no response)
            the bounding rectangle containing all masks
        """
        x0 = minXQuery = np.amin(queryPoly[:,1])
        x1 = maxXQuery = np.amax(queryPoly[:,1])
        y0 = minYQuery = np.amin(queryPoly[:,0])
        y1 = maxYQuery = np.amax(queryPoly[:,0])
        queryCenterX = (maxXQuery+minXQuery)/2
        queryCenterY = (maxYQuery+minYQuery)/2

        def dist(x,y):
            return math.sqrt((queryCenterX-x)**2 + (queryCenterY-y)**2)

        maxDistFromCenter = maxXQuery-minXQuery+maxYQuery-minYQuery
        for poly in polyList:
            minX = np.amin(poly[:,1])
            maxX = np.amax(poly[:,1])
            minY = np.amin(poly[:,0])
            maxY = np.amax(poly[:,0])
            maxDistFromCenter = max(maxDistFromCenter, dist(minX,minY), dist(minX,maxY), dist(maxX,minY), dist(maxX,maxY))
            x0 = min(x0,minX)
            x1 = max(x1,maxX)
            y0 = min(y0,minY)
            y1 = max(y1,maxY)
        
        return ( queryCenterX, queryCenterY, maxDistFromCenter, int(x0),int(y0),int(x1),int(y1))
