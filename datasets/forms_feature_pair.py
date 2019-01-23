import torch
import torch.utils.data
import numpy as np
import json


#import skimage.transform as sktransform
import os
import math
import cv2
from collections import defaultdict
import random
from random import shuffle


from utils.forms_annotations import fixAnnotations, getBBInfo
SKIP=['121','174']



def collate(batch):

    ##tic=timeit.default_timer()
    batch_size = len(batch)
    if batch_size==1 and 'imgPath' in batch[0]:
        return batch[0]#special evaluation mode that puts a whole image in a batch
    imageNames=[]
    data=[]
    labels=torch.ByteTensor(batch_size)
    bi=0
    for b in batch:
        imageNames.append(b['imgName'])
        data.append(b['data'])
        labels[bi] = int(b['label'])
        bi+=1

    return {
        "imgName": imageNames,
        'data': torch.cat(data,dim=0),
        'label': labels
    }

class FormsFeaturePair(torch.utils.data.Dataset):
    """
    Class for reading Forms dataset and creating instances of pair features.
    """

    def __getResponseBBList(self,queryId,annotations):
        responseBBList=[]
        for pair in annotations['pairs']: #done already +annotations['samePairs']:
            if queryId in pair:
                if pair[0]==queryId:
                    otherId=pair[1]
                else:
                    otherId=pair[0]
                if otherId in annotations['byId']: #catch for gt error
                    responseBBList.append(annotations['byId'][otherId])
                #if not self.isSkipField(annotations['byId'][otherId]):
                #    poly = np.array(annotations['byId'][otherId]['poly_points']) #self.__getResponseBB(otherId,annotations)  
                #    responseBBList.append(poly)
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
        self.use_corners = config['corners'] if 'corners' in config else False
        self.no_graphics =  config['no_graphics'] if 'no_graphics' in config else False
        self.swapCircle = config['swap_circle'] if 'swap_circle' in config else True
        self.onlyFormStuff = config['only_form_stuff'] if 'only_form_stuff' in config else False
        self.only_opposite_pairs = False
        self.color = config['color'] if 'color' in config else True
        self.rotate = config['rotation'] if 'rotation' in config else True

        self.simple_dataset = config['simple_dataset'] if 'simple_dataset' in config else False
        self.balance = config['balance'] if 'balance' in config else False

        self.eval = config['eval'] if 'eval' in config else False
        

        #width_mean=400.006887263
        #height_mean=47.9102279201
        xScale=400
        yScale=50
        xyScale=(xScale+yScale)/2

        if instances is not None:
            self.instances=instances
        else:
            if self.simple_dataset:
                splitFile = 'simple_train_valid_test_split.json'
            else:
                splitFile = 'train_valid_test_split.json'
            with open(os.path.join(dirPath,splitFile)) as f:
                groupsToUse = json.loads(f.read())[split]
            groupNames = list(groupsToUse.keys())
            groupNames.sort()
            pair_instances=[]
            notpair_instances=[]
            for groupName in groupNames:
                imageNames=groupsToUse[groupName]
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
                        if annotations is None:
                            with open(os.path.join(jsonPath)) as f:
                                annotations = json.loads(f.read())
                            #print(os.path.join(jsonPath))

                            #fix assumptions made in GTing
                            fixAnnotations(self,annotations)

                        #print(path)
                        for id,bb in annotations['byId'].items():
                            if not self.onlyFormStuff or ('paired' in bb and bb['paired']):
                                qX, qY, qH, qW, qR, qIsText = getBBInfo(bb,self.rotate,useBlankClass=not self.no_blanks)
                                tlX = bb['poly_points'][0][0]
                                tlY = bb['poly_points'][0][1]
                                trX = bb['poly_points'][1][0]
                                trY = bb['poly_points'][1][1]
                                brX = bb['poly_points'][2][0]
                                brY = bb['poly_points'][2][1]
                                blX = bb['poly_points'][3][0]
                                blY = bb['poly_points'][3][1]
                                qH /= yScale #math.log( (qH+0.375*height_mean)/height_mean ) #rescaling so 0 height is -1, big height is 1+
                                qW /= xScale #math.log( (qW+0.375*width_mean)/width_mean ) #rescaling so 0 width is -1, big width is 1+
                                qR = qR/math.pi
                                responseBBList = self.__getResponseBBList(id,annotations)
                                responseIds = [bb['id'] for bb in responseBBList]
                                for id2,bb2 in annotations['byId'].items():
                                    if id!=id2:
                                        iX, iY, iH, iW, iR, iIsText = getBBInfo(bb2,self.rotate,useBlankClass=not self.no_blanks)
                                        tlX2 = bb2['poly_points'][0][0]
                                        tlY2 = bb2['poly_points'][0][1]
                                        trX2 = bb2['poly_points'][1][0]
                                        trY2 = bb2['poly_points'][1][1]
                                        brX2 = bb2['poly_points'][2][0]
                                        brY2 = bb2['poly_points'][2][1]
                                        blX2 = bb2['poly_points'][3][0]
                                        blY2 = bb2['poly_points'][3][1]
                                        iH /=yScale #math.log( (iH+0.375*height_mean)/height_mean ) 
                                        iW /=xScale #math.log( (iW+0.375*width_mean)/width_mean ) 
                                        iR = iR/math.pi
                                        xDiff=iX-qX
                                        yDiff=iY-qY
                                        yDiff /= yScale #math.log( (yDiff+0.375*yDiffScale)/yDiffScale ) 
                                        xDiff /= xScale #math.log( (xDiff+0.375*xDiffScale)/xDiffScale ) 
                                        tlDiff = math.sqrt( (tlX-tlX2)**2 + (tlY-tlY2)**2 )/xyScale
                                        trDiff = math.sqrt( (trX-trX2)**2 + (trY-trY2)**2 )/xyScale
                                        brDiff = math.sqrt( (brX-brX2)**2 + (brY-brY2)**2 )/xyScale
                                        blDiff = math.sqrt( (blX-blX2)**2 + (blY-blY2)**2 )/xyScale
                                        tlXDiff = (tlX2-tlX)/xScale
                                        trXDiff = (trX2-trX)/xScale
                                        brXDiff = (brX2-brX)/xScale
                                        blXDiff = (blX2-blX)/xScale
                                        tlYDiff = (tlY2-tlY)/yScale
                                        trYDiff = (trY2-trY)/yScale
                                        brYDiff = (brY2-brY)/yScale
                                        blYDiff = (blY2-blY)/yScale
                                        pair = id2 in responseIds
                                        if pair or self.eval:
                                            instances = pair_instances
                                        else:
                                            instances = notpair_instances
                                        data=[qH,qW,qR,qIsText, iH,iW,iR,iIsText, xDiff, yDiff]
                                        if self.use_corners=='xy':
                                            data+=[tlXDiff,trXDiff,brXDiff,blXDiff,tlYDiff,trYDiff,brYDiff,blYDiff]
                                        elif self.use_corners:
                                            data+=[tlDiff, trDiff, brDiff, blDiff]
                                        instances.append( {
                                            'data': torch.tensor([ data ]),
                                            'label': pair,
                                            'imgName': imageName,
                                            'qXY' : (qX,qY),
                                            'iXY' : (iX,iY),
                                            'ids' : (id,id2)
                                            } )
                        if self.eval:
                            datas=[]
                            labels=[]
                            qXYs=[]
                            iXYs=[]
                            nodeIds=[]
                            for inst in pair_instances:
                                datas.append(inst['data'])
                                labels.append(inst['label'])
                                qXYs.append(inst['qXY'])
                                iXYs.append(inst['iXY'])
                                nodeIds.append(inst['ids'])
                            if len(datas)>0:
                                data = torch.cat(datas,dim=0),
                            else:
                                data = torch.FloatTensor((0,10))
                            notpair_instances.append( {
                                'data': data,
                                'label': torch.ByteTensor(labels),
                                'imgName': imageName,
                                'imgPath' : path,
                                'qXY' : qXYs,
                                'iXY' : iXYs,
                                'nodeIds' : nodeIds
                                } )
                            pair_instances=[]
            self.instances = notpair_instances
            if self.balance and not self.eval:
                dif = len(notpair_instances)/float(len(pair_instances))
                print('not: {}, pair: {}. Adding {}x'.format(len(notpair_instances),len(pair_instances),math.floor(dif)))
                for i in range(math.floor(dif)):
                    self.instances += pair_instances
            else:
                self.instances += pair_instances



    def __len__(self):
        return len(self.instances)

    def __getitem__(self,index):
        return self.instances[index]
