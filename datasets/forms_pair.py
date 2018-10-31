import torch
import torch.utils.data
import numpy as np
import json
from skimage import io
from skimage import draw
#import skimage.transform as sktransform
import os
import math
import cv2
from utils import augmentation
from utils.crop_transform import CropBoxTransform
from datasets.forms_box_pair import fixAnnotations, getDistMask
import random
from random import shuffle

SKIP=['121','174']



class FormsPair(torch.utils.data.Dataset):
    """
    Class for reading AI2D dataset and creating query/result masks from bounding polygons
    """

    def isSkipField(self,bb):
        return (    (self.no_blanks and (bb['isBlank']=='blank' or bb['isBlank']==3)) or
                    (self.no_print_fields and (bb['isBlank']=='print' or bb['isBlank']==2)) or
                    #TODO no graphics
                    bb['type'] == 'fieldRow' or
                    bb['type'] == 'fieldCol' or
                    bb['type'] == 'fieldRegion'
                )
    def __getResponseBBList(self,queryId,annotations):
        responseBBList=[]
        for pair in annotations['pairs']: #+annotations['samePairs']: added by fixAnnotations
            if queryId in pair:
                if pair[0]==queryId:
                    otherId=pair[1]
                else:
                    otherId=pair[0]
                if otherId in annotations['byId']: #catch for gt error
                    poly = np.array(annotations['byId'][otherId]['poly_points']) #self.__getResponseBB(otherId,annotations)  
                    responseBBList.append(poly)
        return responseBBList


    def __init__(self, dirPath=None, split=None, config=None, instances=None, test=False):
        if split=='valid':
            valid=True
            amountPer=0.25
        else:
            valid=False
        self.cache_resized=False
        if 'augmentation_params' in config:
            self.augmentation_params=config['augmentation_params']
        else:
            self.augmentation_params=None
        if 'no_blanks' in config:
            self.no_blanks = config['no_blanks']
        else:
            self.no_blanks = False
        if 'no_print_fields' in config:
            self.no_print_fields = config['no_print_fields']
        else:
            self.no_print_fields = False
        self.no_graphics =  config['no_graphics'] if 'no_graphics' in config else False
        self.swapCircle = config['swap_circle'] if 'swap_circle' in config else True
        self.color = config['color'] if 'color' in config else True
        self.bbCrop = config['use_bb_crop'] if 'use_bb_crop' in config else True
        #self.rotate = config['rotation'] if 'rotation' in config else True
        self.useDistMask = config['use_dist_mask'] if 'use_dist_mask' in config else False
        self.useVDistMask = config['use_vdist_mask'] if 'use_vdist_mask' in config else False
        self.useHDistMask = config['use_hdist_mask'] if 'use_hdist_mask' in config else False
        patchSize=config['patch_size'] if 'patch_size' in config else None
        self.rescale_range = config['rescale_range']
        self.halfResponse = config['half_response'] if 'half_response' in config else False
        self.cache_resized = config['cache_resized_images'] if 'cache_resized_images' in config else False
        if self.cache_resized:
            self.cache_path = os.path.join(dirPath,'cache_'+str(self.rescale_range[1]))
            if not os.path.exists(self.cache_path):
                os.mkdir(self.cache_path)
        if patchSize is not None:
            self.transform = CropBoxTransform({"crop_size":patchSize})
        else:
            self.transform = None
        if instances is not None:
            self.instances=instances
            self.cropResize = self.__cropResizeF(patchSize,0,0)
            self.transform = None
        else:
            if not self.bbCrop:
                centerJitterFactor=config['center_jitter']
                sizeJitterFactor=config['size_jitter']
                self.cropResize = self.__cropResizeF(patchSize,centerJitterFactor,sizeJitterFactor)
            with open(os.path.join(dirPath,'train_valid_test_split.json')) as f:
                groupsToUse = json.loads(f.read())[split]
            self.instances=[]
            if test:
                aH=0
                aW=0
                aA=0
            for groupName, imageNames in groupsToUse.items():
                if groupName in SKIP:
                    print('Skipped group {}'.format(groupName))
                    continue
                for imageName in imageNames:
                    org_path = os.path.join(dirPath,'groups',groupName,imageName)
                    #print(org_path)
                    if self.cache_resized:
                        path = os.path.join(self.cache_path,imageName)
                    else:
                        path = org_path
                    jsonPath = org_path[:org_path.rfind('.')]+'.json'
                    annotations=None
                    if os.path.exists(jsonPath):
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
                        if annotations is None:
                            with open(os.path.join(jsonPath)) as f:
                                annotations = json.loads(f.read())
                            #print(os.path.join(jsonPath))
                        annotations['byId']={}
                        imH = annotations['height']
                        imW = annotations['width']
                        for bb in annotations['textBBs']:
                            annotations['byId'][bb['id']]=bb
                        for bb in annotations['fieldBBs']:
                            annotations['byId'][bb['id']]=bb

                        #fix assumptions made in GTing
                        fixAnnotations(self,annotations)

                        #print(path)
                        instancesForImage=[]
                        for id,bb in annotations['byId'].items():
                            bbPoints = np.array(bb['poly_points'])
                            responseBBList = self.__getResponseBBList(id,annotations)
                            #print(id)
                            #print(responseBBList)
                            instancesForImage.append({
                                                'id': id,
                                                'imagePath': path,
                                                'imageName': imageName,
                                                'rescaled': rescale,
                                                'queryPoly': bbPoints,
                                                'responsePolyList': responseBBList,
                                                'helperStats': self.__getHelperStats(bbPoints, responseBBList, imH, imW)
                                            })
                        if valid:
                            random.seed(123)
                            shuffle(instancesForImage)
                            self.instances += instancesForImage[:int(amountPer*len(instancesForImage))]
                        else:
                            self.instances += instancesForImage

                        #for i in range(startCount,len(self.instances)):
                            #    try:
                                #        #self.helperStats.append((0,0,0,0,0,0,0))
                        #    except ValueError as err:
                                #        print('error on image '+image+', id '+self.ids[i])
                        #        print(os.path.join(dirPath,'annotationsMod',image+'.json'))
                        #        print(err)
                        #        exit(2)
                if test:
                    with open(os.path.join(dirPath,'annotationsMod',image+'.json')) as f:
                        annotations = json.loads(f.read())
                        imH = annotations['height']
                        imW = annotations['width']
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
        imageName = self.instances[index]['imageName']
        queryPoly = self.instances[index]['queryPoly']
        responsePolyList = self.instances[index]['responsePolyList']
        xQueryC,yQueryC,reach,x0,y0,x1,y1 = self.instances[index]['helperStats']
        #print(index)
        #print(self.imagePath)
        #print(self.ids[index])
        image = cv2.imread(imagePath, 1 if self.color else 0)
        
        if self.bbCrop:
            #resize
            rescaled=self.instances[index]['rescaled']
            scale = np.random.uniform(self.rescale_range[0], self.rescale_range[1])
            partial_rescale = scale/rescaled
            image = cv2.resize(image,(0,0),
                    fx=partial_rescale,
                    fy=partial_rescale,
                    interpolation = cv2.INTER_CUBIC)
            if not self.color:
                image=image[:,:,None]
            response_bbs = self.getBBGT(responsePolyList,scale)
            query_bb = self.getBBGT([queryPoly],scale)[0]
            queryMask = self.makeQueryMask(image,query_bb[[1,3,5,7]], query_bb[[0,2,4,6]])
            if self.transform is not None:
                out = self.transform({
                    "img": image,
                    "bb_gt": response_bbs[None,...],#expects batch dim
                    "query_bb":query_bb,
                    "point_gt": None,
                    "pixel_gt": queryMask,
                })
                image = out['img']
                response_bbs = out['bb_gt'][0]
                if image.shape[2]==3:
                    image = augmentation.apply_random_color_rotation(image)
                image = augmentation.apply_tensmeyer_brightness(image)
                queryMask = out['pixel_gt']
                #TODO rotate?
            image = 1-image/128.0
            imageWithQuery = np.append(image,queryMask,axis=2)
            imageWithQuery = np.moveaxis(imageWithQuery,2,0)
            responseMask = np.zeros([image.shape[0],image.shape[1]])
            for i in range(response_bbs.shape[0]):
                rr, cc = draw.polygon(response_bbs[i,[1,3,5,7]], response_bbs[i,[0,2,4,6]], responseMask.shape)
                responseMask[rr,cc]=1
            if self.halfResponse:
                responseMask = cv2.resize(  responseMask,
                                            (responseMask.shape[1]//2,responseMask.shape[0]//2),
                                            interpolation = cv2.INTER_CUBIC)
            return (imageWithQuery, imageName,responseMask)
        else:
            image = 1-image/128.0
            if not self.color:
                image=image[:,:,None]
            queryMask = self.makeQueryMask(image,queryPoly[:, 1], queryPoly[:, 0])
            responseMask = np.zeros([image.shape[0],image.shape[1]])
            for poly in responsePolyList:
                rr, cc = draw.polygon(poly[:, 1], poly[:, 0], responseMask.shape)
                responseMask[rr,cc]=1
            imageWithQuery = np.append(image,queryMask,axis=2)
            imageWithQuery = np.moveaxis(imageWithQuery,2,0)
            sample = self.cropResize(imageWithQuery, responseMask, xQueryC,yQueryC,reach,x0,y0,x1,y1)
            #sample = (imageWithQuery, responseMask,) + helperStats + (imagePath+' '+id,)
            if self.augmentation_params is not None:
                sample = self.augment(sample)
        return sample #+ (imagePath+' '+id,)

    def makeQueryMask(self,image,ys,xs):
        queryMask = np.zeros([image.shape[0],image.shape[1]])
        rr, cc = draw.polygon(ys, xs, queryMask.shape)
        queryMask[rr,cc]=1
        masks = [queryMask]
        distMask=None
        if self.useDistMask:
            distMask = getDistMask(queryMask)
            masks.append(distMask)
        if self.useHDistMask:
            if distMask is None:
                distMask = getDistMask(queryMask)
            minY=math.ceil(ys.min())
            maxY=math.floor(ys.max())
            hdistMask = distMask.copy()
            hdistMask[:minY,:]=-1
            hdistMask[maxY:,:]=-1
            masks.append(hdistMask)
        if self.useVDistMask:
            if distMask is None:
                distMask = getDistMask(queryMask)
            minX=math.ceil(xs.min())
            maxX=math.floor(xs.max())
            vdistMask = distMask.copy()
            vdistMask[:,:minX]=-1
            vdistMask[:,maxX:]=-1
            masks.append(vdistMask)

        return np.stack(masks,axis=2)

    def getBBGT(self,usePolys,s):


        bbs = np.empty((len(usePolys), 8+8), dtype=np.float32) #2x4 corners, 2x4 cross-points
        j=0
        for poly in usePolys:
            tlX = poly[0][0]
            tlY = poly[0][1]
            trX = poly[1][0]
            trY = poly[1][1]
            brX = poly[2][0]
            brY = poly[2][1]
            blX = poly[3][0]
            blY = poly[3][1]


            bbs[j,0]=tlX*s
            bbs[j,1]=tlY*s
            bbs[j,2]=trX*s
            bbs[j,3]=trY*s
            bbs[j,4]=brX*s
            bbs[j,5]=brY*s
            bbs[j,6]=blX*s
            bbs[j,7]=blY*s
            #we add these for conveince to crop BBs within window
            bbs[j,8]=s*(tlX+blX)/2
            bbs[j,9]=s*(tlY+blY)/2
            bbs[j,10]=s*(trX+brX)/2
            bbs[j,11]=s*(trY+brY)/2
            bbs[j,12]=s*(tlX+trX)/2
            bbs[j,13]=s*(tlY+trY)/2
            bbs[j,14]=s*(brX+blX)/2
            bbs[j,15]=s*(brY+blY)/2
            j+=1
        return bbs

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

        #if x1>=imW or y1>=imH:
        #    raise ValueError('query point outside image ('+str(imH)+', '+str(imW)+'): y='+str(y1)+' x='+str(x1)+'   '+str(queryPoly))

        def dist(x,y):
            return math.sqrt((queryCenterX-x)**2 + (queryCenterY-y)**2)

        maxDistFromCenter = maxXQuery-minXQuery+maxYQuery-minYQuery
        for poly in polyList:
            minX = np.amin(poly[:,0])
            maxX = np.amax(poly[:,0])
            minY = np.amin(poly[:,1])
            maxY = np.amax(poly[:,1])
            #if maxX>=imW or maxY>=imH:
            #    raise ValueError('resp point outside image ('+str(imH)+', '+str(imW)+'): y='+str(maxY)+' x='+str(maxX))
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

            
            #are to going to expand the image? Don't
            if radius < patchSize/2.0:
                radius = patchSize/2.0 + abs(np.random.normal(0,reach*sizeJitterFactor/2))

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

            #retImage = sktransform.resize(imagePatch.transpose((1, 2, 0)),(patchSize,patchSize)).transpose((2, 0, 1))
            #retLabel = sktransform.resize(labelPatch, (patchSize,patchSize))
            retImage = cv2.resize(imagePatch.transpose((1, 2, 0)),(patchSize,patchSize)).transpose((2, 0, 1))
            retLabel = cv2.resize(labelPatch, (patchSize,patchSize))
            retImage = torch.from_numpy(retImage.astype(np.float32))
            retLabel = torch.from_numpy(retLabel.astype(np.float32))
            return (retImage, retLabel)

        return cropResize
