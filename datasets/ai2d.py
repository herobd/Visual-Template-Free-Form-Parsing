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

    def __init__(self, dirPath=None, split=None, config=None, instances=None, test=False):
        if 'augmentation_params' in config['data_loader']:
            self.augmentation_params=config['data_loader']['augmentation_params']
        else:
            self.augmentation_params=None
        patchSize=config['data_loader']['patch_size']
        if instances is not None:
            self.instances=instances
            self.cropResize = self.__cropResizeF(patchSize,0,0)
        else:
            centerJitterFactor=config['data_loader']['center_jitter']
            sizeJitterFactor=config['data_loader']['size_jitter']
            self.cropResize = self.__cropResizeF(patchSize,centerJitterFactor,sizeJitterFactor)
            with open(os.path.join(dirPath,'categories.json')) as f:
                imageToCategories = json.loads(f.read())
            with open(os.path.join(dirPath,'traintestplit_categories.json')) as f:
                #if split=='valid' or split=='validation':
                #    trainTest='train'
                #else:
                #    trainTest=split
                categoriesToUse = json.loads(f.read())[split]
            self.instances=[]
            if test:
                aH=0
                aW=0
                aA=0
            for image, category in imageToCategories.items():
                if category in categoriesToUse:
                    with open(os.path.join(dirPath,'annotationsMod',image+'.json')) as f:
                        annotations = json.loads(f.read())
                        imH = annotations['imageConsts']['height']
                        imW = annotations['imageConsts']['width']
                        #startCount=len(self.instances)
                        if test:
                            aH+=imH
                            aW+=imW
                            aA+=imH*imW
                        for blobId, blob in annotations['blobs'].items():
                            qPoly = np.array(blob['polygon'])
                            responsePolyList = self.__getResponsePolyList(blobId,annotations)
                            self.instances.append({
                                                'id': blobId,
                                                'imagePath': os.path.join(dirPath,'images',image),
                                                'queryPoly': qPoly,
                                                'responsePolyList': responsePolyList,
                                                'helperStats': self.__getHelperStats(qPoly, responsePolyList, imH, imW)
                                            })

                        for arrowId, arrow in annotations['arrows'].items():
                            qPoly = np.array(arrow['polygon'])
                            responsePolyList = self.__getResponsePolyList(arrowId,annotations)
                            self.instances.append({
                                                'id': arrowId,
                                                'imagePath': os.path.join(dirPath,'images',image),
                                                'queryPoly': qPoly,
                                                'responsePolyList': responsePolyList,
                                                'helperStats': self.__getHelperStats(qPoly, responsePolyList, imH, imW)
                                            })

                        for textId, text in annotations['text'].items():
                            rect=text['rectangle']
                            qPoly = np.array([ rect[0], [rect[1][0],rect[0][1]], rect[1], [rect[0][0],rect[1][1]] ])
                            responsePolyList = self.__getResponsePolyList(textId,annotations)
                            self.instances.append({
                                                'id': textId,
                                                'imagePath': os.path.join(dirPath,'images',image),
                                                'queryPoly': qPoly,
                                                'responsePolyList': responsePolyList,
                                                'helperStats': self.__getHelperStats(qPoly, responsePolyList, imH, imW)
                                            })

                        #for i in range(startCount,len(self.instances)):
                            #    try:
                                #        #self.helperStats.append((0,0,0,0,0,0,0))
                        #    except ValueError as err:
                                #        print('error on image '+image+', id '+self.ids[i])
                        #        print(os.path.join(dirPath,'annotationsMod',image+'.json'))
                        #        print(err)
                        #        exit(2)
                elif test:
                    with open(os.path.join(dirPath,'annotationsMod',image+'.json')) as f:
                        annotations = json.loads(f.read())
                        imH = annotations['imageConsts']['height']
                        imW = annotations['imageConsts']['width']
                        aH+=imH
                        aW+=imW
                        aA+=imH*imW
        
        if test:
            print('average height: '+str(aH/len(imageToCategories)))
            print('average width:  '+str(aW/len(imageToCategories)))
            print('average area:   '+str(aA/len(imageToCategories)))



    def __len__(self):
        return len(self.instances)

    def __getitem__(self,index):
        id = self.instances[index]['id']
        imagePath = self.instances[index]['imagePath']
        queryPoly = self.instances[index]['queryPoly']
        responsePolyList = self.instances[index]['responsePolyList']
        xQueryC,yQueryC,reach,x0,y0,x1,y1 = self.instances[index]['helperStats']
        #print(index)
        #print(self.imagePaths[index])
        #print(self.ids[index])
        image = io.imread(imagePath)/255.0
        #TODO color jitter, rotation?, skew?
        queryMask = np.zeros([image.shape[0],image.shape[1]])
        rr, cc = draw.polygon(queryPoly[:, 1], queryPoly[:, 0], queryMask.shape)
        queryMask[rr,cc]=1
        responseMask = np.zeros([image.shape[0],image.shape[1]])
        for poly in responsePolyList:
            rr, cc = draw.polygon(poly[:, 1], poly[:, 0], responseMask.shape)
            responseMask[rr,cc]=1

        imageWithQuery = np.append(image,queryMask.reshape(queryMask.shape+(1,)),axis=2)
        imageWithQuery = np.moveaxis(imageWithQuery,2,0)
        sample = self.cropResize(imageWithQuery, responseMask, xQueryC,yQueryC,reach,x0,y0,x1,y1)
        #sample = (imageWithQuery, responseMask,) + helperStats + (imagePath+' '+id,)
        if self.augmentation_params is not None:
            sample = self.augment(sample)
        return sample #+ (imagePath+' '+id,)

    def splitValidation(self, config):
        validation_split = config['validation']['validation_split']
        split = int(len(self) * validation_split)
        perm = np.random.permutation(len(self))
        instances = [self.instances[x] for x in perm]

        self.instances=instances[split:]

        return AI2D(config=config, instances=instances[:split])

    def __getHelperStats(self, queryPoly, polyList, imH, imW):
        """
        This returns stats used when putting a batch together, croping and resizeing windows.
        It returns
            the centerpoint of the query mask,
            the furthest response mask point from the center (minimum set by query mask size in case no response)
            the bounding rectangle containing all masks
        """
        x0 = minXQuery = np.amin(queryPoly[:,0])
        x1 = maxXQuery = np.amax(queryPoly[:,0])
        y0 = minYQuery = np.amin(queryPoly[:,1])
        y1 = maxYQuery = np.amax(queryPoly[:,1])
        queryCenterX = (maxXQuery+minXQuery)/2
        queryCenterY = (maxYQuery+minYQuery)/2

        if x1>=imW or y1>=imH:
            raise ValueError('query point outside image ('+str(imH)+', '+str(imW)+'): y='+str(y1)+' x='+str(x1)+'   '+str(queryPoly))

        def dist(x,y):
            return math.sqrt((queryCenterX-x)**2 + (queryCenterY-y)**2)

        maxDistFromCenter = maxXQuery-minXQuery+maxYQuery-minYQuery
        for poly in polyList:
            minX = np.amin(poly[:,0])
            maxX = np.amax(poly[:,0])
            minY = np.amin(poly[:,1])
            maxY = np.amax(poly[:,1])
            if maxX>=imW or maxY>=imH:
                raise ValueError('resp point outside image ('+str(imH)+', '+str(imW)+'): y='+str(maxY)+' x='+str(maxX))
            maxDistFromCenter = max(maxDistFromCenter, dist(minX,minY), dist(minX,maxY), dist(maxX,minY), dist(maxX,maxY))
            x0 = min(x0,minX)
            x1 = max(x1,maxX)
            y0 = min(y0,minY)
            y1 = max(y1,maxY)
        ###
        #if (imH==183 and imW==183):
        #    print(( queryCenterX, queryCenterY, maxDistFromCenter, int(x0),int(y0),int(x1),int(y1)))
        return ( queryCenterX, queryCenterY, maxDistFromCenter, int(x0),int(y0),int(x1),int(y1))

    def __cropResizeF(self,patchSize, centerJitterFactor, sizeJitterFactor):
        """
        Returns function which crops and pads data to include all masks (mostly) and be uniform size
        """

        #resizeImage=transforms.Resize((patchSize, patchSize))
        def squareBB(x0,x1,dimLen,toFill):
            run = x0 + dimLen-x1
            if run<=toFill:
                new_x0=0
                new_x1=dimLen
            else:
                play = run-toFill
                new_x0 = max(np.random.randint(x0-toFill,x0),0)
                toFill -= x0-new_x0
                new_x1 = min(x1+toFill,dimLen)
                toFill -= new_x1-x1
                new_x0 = new_x0-toFill
                assert(new_x0>=0)
            return new_x0, new_x1

        def cropResize(image,label,xQueryC,yQueryC,reach,x0,y0,x1,y1):
            xc = int(min(max( xQueryC + np.random.normal(0,reach*centerJitterFactor) ,0),image.shape[2]-1))
            yc = int(min(max( yQueryC + np.random.normal(0,reach*centerJitterFactor) ,0),image.shape[1]-1))
            radius = int(reach + np.random.normal(reach*centerJitterFactor,reach*sizeJitterFactor))

            if radius<=0:
                radius=int(reach//2)
            #make radius smaller if we go off image, randomly
            #then make radius big enough to include all masks, randomly
            if centerJitterFactor==0:
                #not random if valid
                if xc+radius>image.shape[2]-1:
                    radius = image.shape[2]-1-xc
                if xc-radius<0:
                    radius = xc
                if yc+radius+1>image.shape[1]:
                    radius = image.shape[1]-yc-1
                if yc-radius<0:
                    radius = yc

                if xc+radius<x1:
                    radius= x1-xc
                if xc-radius>x0:
                    radius = xc-x0
                if yc+radius<y1:
                    radius = y1-yc
                if yc-radius>y0:
                    radius = yc-y0
            else:
                if xc+radius>image.shape[2]-1:
                    radius = np.random.randint(image.shape[2]-1-xc, radius+1)
                if xc-radius<0:
                    radius = np.random.randint(xc, radius+1)
                if yc+radius+1>image.shape[1]:
                    radius = np.random.randint(image.shape[1]-yc-1, radius+1)
                if yc-radius<0:
                    radius = np.random.randint(yc, radius+1)


                if xc+radius<x1:
                    radius= np.random.randint(radius,x1-xc +1)
                if xc-radius>x0:
                    radius = np.random.randint(radius, xc-x0 +1)
                if yc+radius<y1:
                    radius = np.random.randint(radius, y1-yc +1)
                if yc-radius>y0:
                    radius = np.random.randint(radius, yc-y0 +1)

            

            cropOutX0 = int(max(xc-radius,0))
            cropOutY0 = int(max(yc-radius,0))
            cropOutX1 = int(min(xc+radius+1,image.shape[2]))
            cropOutY1 = int(min(yc+radius+1,image.shape[1]))
            size = (cropOutY1-cropOutY0,cropOutX1-cropOutX0)
            if size[0]!=size[1]:
                #force square, if possible
                if size[0] < size[1]:
                    cropOutY0, cropOutY1 = squareBB(cropOutY0, cropOutY1, image.shape[1], size[1]-size[0])
                else:
                    cropOutX0, cropOutX1 = squareBB(cropOutX0, cropOutX1, image.shape[2], size[0]-size[1])
                size = (cropOutY1-cropOutY0,cropOutX1-cropOutX0)
            bbSize = radius*2+1
            if size[0]!=bbSize and size[1]!=bbSize:
                bbSize = max(size[0],size[1])

            #print(id)
            #print('image shape: '+str(image.shape))
            #print((xQueryC,yQueryC,reach,x0,y0,x1,y1))
            #print((xc,yc,(bbSize-1)/2))
            #print((cropOutX0, cropOutY0, cropOutX1, cropOutY1))
            #print(size)
            
            assert(size[0]<=bbSize and size[1]<=bbSize)
            if size[0]!=size[1]:

                diffH = (bbSize)-size[0]
                if diffH==0 or centerJitterFactor==0:
                    padTop=0
                else:
                    padTop = np.random.randint(0,diffH)

                diffW = (bbSize)-size[1]
                if diffW==0 or centerJitterFactor==0:
                    padLeft=0
                else:
                    padLeft = np.random.randint(0,diffW)

                imagePatch = np.zeros((image.shape[0],bbSize,bbSize), dtype=np.float32)
                imagePatch[:,padTop:size[0]+padTop,padLeft:size[1]+padLeft] = image[:,cropOutY0:cropOutY1,cropOutX0:cropOutX1]
                labelPatch = np.zeros((bbSize,bbSize), dtype=np.float32)
                labelPatch[padTop:size[0]+padTop,padLeft:size[1]+padLeft] = label[cropOutY0:cropOutY1,cropOutX0:cropOutX1]
            else:
                imagePatch = image[:,cropOutY0:cropOutY1,cropOutX0:cropOutX1]
                labelPatch = label[cropOutY0:cropOutY1,cropOutX0:cropOutX1]

            retImage = sktransform.resize(imagePatch.transpose((1, 2, 0)),(patchSize,patchSize)).transpose((2, 0, 1))
            retLabel = sktransform.resize(labelPatch, (patchSize,patchSize))
            return (retImage, retLabel)

        return cropResize
