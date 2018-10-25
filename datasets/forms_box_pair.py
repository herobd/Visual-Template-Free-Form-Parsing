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
from collections import defaultdict
from datasets.forms_box_detect import convertBBs
from utils import augmentation
from utils.crop_transform import CropBoxTransform
SKIP=['174']

def avg_y(bb):
    points = bb['poly_points']
    return (points[0][1]+points[1][1]+points[2][1]+points[3][1])/4.0
def avg_x(bb):
    points = bb['poly_points']
    return (points[0][0]+points[1][0]+points[2][0]+points[3][0])/4.0
def left_x(bb):
    points = bb['poly_points']
    return (points[0][0]+points[3][0])/2.0
def right_x(bb):
    points = bb['poly_points']
    return (points[1][0]+points[2][0])/2.0

def collate(batch):

    ##tic=timeit.default_timer()
    batch_size = len(batch)
    imageNames=[]
    scales=[]
    imgs = []
    queryMask=[]
    max_h=0
    max_w=0
    bb_sizes=[]
    bb_dim=None
    for b in batch:
        if b is None:
            continue
        imageNames.append(b['imgName'])
        scales.append(b['scale'])
        imgs.append(b["img"])
        queryMask.append(b['queryMask'])
        max_h = max(max_h,b["img"].size(2))
        max_w = max(max_w,b["img"].size(3))
        gt = b['responseBBs']
        if gt is None:
            bb_sizes.append(0)
        else:
            bb_sizes.append(gt.size(1)) 
            bb_dim=gt.size(2)
    if len(imgs) == 0:
        return None

    largest_bb_count = max(bb_sizes)

    ##print(' col channels: {}'.format(len(imgs[0].size())))
    batch_size = len(imgs)

    resized_imgs = []
    resized_queryMask = []
    index=0
    for img in imgs:
        if img.size(2)<max_h or img.size(3)<max_w:
            resized = torch.zeros([1,img.size(1),max_h,max_w]).type(img.type())
            diff_h = max_h-img.size(2)
            pos_r = 0#np.random.randint(0,diff_h+1)
            diff_w = max_w-img.size(3)
            pos_c = 0#np.random.randint(0,diff_w+1)
            #if len(img.size())==3:
                #    resized[:,pos_r:pos_r+img.size(1), pos_c:pos_c+img.size(2)]=img
            #else:
                #    resized[pos_r:pos_r+img.size(1), pos_c:pos_c+img.size(2)]=img
            resized[:,:,pos_r:pos_r+img.size(2), pos_c:pos_c+img.size(3)]=img
            resized_imgs.append(resized)

            if queryMask[index] is not None:
                resized_gt = torch.zeros([1,queryMask[index].size(1),max_h,max_w]).type(queryMask[index].type())
                resized_gt[:,:,pos_r:pos_r+img.size(2), pos_c:pos_c+img.size(3)]=queryMask[index]
                resized_queryMask.append(resized_gt)
        else:
            resized_imgs.append(img)
            if queryMask[index] is not None:
                resized_queryMask.append(queryMask[index])
        index+=1

            

    if largest_bb_count != 0:
        bbs = torch.zeros(batch_size, largest_bb_count, bb_dim)
    else:
        bbs=None
    for i, b in enumerate(batch):
        gt = b['responseBBs']
        if bb_sizes[i] == 0:
            continue
        bbs[i, :bb_sizes[i]] = gt


    imgs = torch.cat(resized_imgs)
    if len(resized_queryMask)==1:
        queryMask = resized_queryMask[0]
    elif len(resized_queryMask)>1:
        queryMask = torch.cat(resized_queryMask)
    else:
        queryMask = None

    ##print('collate: '+str(timeit.default_timer()-tic))
    return {
        'img': imgs,
        'responseBBs': bbs,
        "responseBB_sizes": bb_sizes,
        'queryMask': queryMask,
        "imgName": imageNames,
        "scale": scales
    }

class FormsBoxPair(torch.utils.data.Dataset):
    """
    Class for reading Forms dataset and creating quer masks from bbbs
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
        self.rotate = config['rotation'] if 'rotation' in config else True
        
        self.rescale_range=config['rescale_range']
        if 'cache_resized_images' in config:
            self.cache_resized = config['cache_resized_images']
            if self.cache_resized:
                self.cache_path = os.path.join(dirPath,'cache_'+str(self.rescale_range[1]))
                if not os.path.exists(self.cache_path):
                    os.mkdir(self.cache_path)
        else:
            self.cache_resized = False
        #self.fixedDetectorCheckpoint = config['detector_checkpoint'] if 'detector_checkpoint' in config else None
        crop_params=config['crop_params'] if 'crop_params' in config else None
        if crop_params is not None:
            self.transform = CropBoxTransform(crop_params)
        else:
            self.transform = None
        if instances is not None:
            self.instances=instances
            self.cropResize = self.__cropResizeF(patchSize,0,0)
        else:
            with open(os.path.join(dirPath,'train_valid_test_split.json')) as f:
                groupsToUse = json.loads(f.read())[split]
            self.instances=[]
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
                        for bb in annotations['textBBs']:
                            annotations['byId'][bb['id']]=bb
                        for bb in annotations['fieldBBs']:
                            annotations['byId'][bb['id']]=bb

                        #fix assumptions made in GTing
                        self.fixAnnotations(annotations)

                        #print(path)
                        for id,bb in annotations['byId'].items():
                            responseBBList = self.__getResponseBBList(id,annotations)
                            self.instances.append({
                                                'id': id,
                                                'imagePath': path,
                                                'imageName': imageName[:imageName.rfind('.')],
                                                'queryBB': bb,
                                                'responseBBList': responseBBList,
                                                'rescaled':rescale,
                                                #'helperStats': self.__getHelperStats(bbPoints, responseBBList, imH, imW)
                                            })

        


    def __len__(self):
        return len(self.instances)

    def __getitem__(self,index):
        id = self.instances[index]['id']
        imagePath = self.instances[index]['imagePath']
        imageName = self.instances[index]['imageName']
        queryBB = self.instances[index]['queryBB']
        assert(queryBB['type']!='fieldCol')
        responseBBList = self.instances[index]['responseBBList']
        rescaled = self.instances[index]['rescaled']
        #xQueryC,yQueryC,reach,x0,y0,x1,y1 = self.instances[index]['helperStats']

        np_img = cv2.imread(imagePath, 1 if self.color else 0)
        #Rescale
        scale = np.random.uniform(self.rescale_range[0], self.rescale_range[1])
        partial_rescale = scale/rescaled
        np_img = cv2.resize(np_img,(0,0),
                fx=partial_rescale,
                fy=partial_rescale,
                interpolation = cv2.INTER_CUBIC)
        #queryPoly *=scale

        if not self.color:
            np_img = np_img[:,:,None]

        response_bbs = self.getBBGT(responseBBList,scale)
        query_bb = self.getBBGT([queryBB],scale)[0,0]

        queryMask = np.zeros([np_img.shape[0],np_img.shape[1]])
        rr, cc = draw.polygon(query_bb[[1,3,5,7]], query_bb[[0,2,4,6]], queryMask.shape)
        queryMask[rr,cc]=1
        queryMask=queryMask[...,None] #add channel
        #responseMask = np.zeros([image.shape[0],image.shape[1]])
        #for poly in responsePolyList:
        #    rr, cc = draw.polygon(poly[:, 1], poly[:, 0], responseMask.shape)
        #    responseMask[rr,cc]=1

        #imageWithQuery = np.append(1-np_img/128.0,queryMask[:,:,None]),axis=2)
        #sample = self.cropResize(imageWithQuery, responseMask, xQueryC,yQueryC,reach,x0,y0,x1,y1)
        if self.transform is not None:
            out = self.transform({
                "img": np_img,
                "bb_gt": response_bbs,
                "query_bb":query_bb,
                "point_gt": None,
                "pixel_gt": queryMask,
            })
            np_img = out['img']
            response_bbs = out['bb_gt']
            if np_img.shape[2]==3:
                np_img = augmentation.apply_random_color_rotation(np_img)
            np_img = augmentation.apply_tensmeyer_brightness(np_img)
            queryMask = out['pixel_gt']
            #TODO rotate?

        t_response_bbs = convertBBs(response_bbs,self.rotate,2)
        if t_response_bbs is not None and (torch.isnan(t_response_bbs).any() or (float('inf')==t_response_bbs).any()):
            import pdb; pdb.set_trace()

        np_img = np.moveaxis(np_img,2,0)[None,...] #swap channel dim and add batch dim
        t_img = torch.from_numpy(np_img.astype(np.float32))
        t_img = 1.0 - t_img/128.0
        queryMask = np.moveaxis(queryMask,2,0)[None,...] #swap channel dim and add batch dim
        t_queryMask = torch.from_numpy(queryMask.astype(np.float32))
        return {
                'img':t_img,
                'imgName':imageName,
                'queryMask':t_queryMask,
                'scale': scale,
                'responseBBs':t_response_bbs
                }

    def getBBGT(self,useBBs,s):

        
        bbs = np.empty((1,len(useBBs), 8+8+2), dtype=np.float32) #2x4 corners, 2x4 cross-points, 2 classes
        j=0
        for bb in useBBs:
            tlX = bb['poly_points'][0][0]
            tlY = bb['poly_points'][0][1]
            trX = bb['poly_points'][1][0]
            trY = bb['poly_points'][1][1]
            brX = bb['poly_points'][2][0]
            brY = bb['poly_points'][2][1]
            blX = bb['poly_points'][3][0]
            blY = bb['poly_points'][3][1]

            field = bb['type'][:4]!='text' 

            bbs[:,j,0]=tlX*s
            bbs[:,j,1]=tlY*s
            bbs[:,j,2]=trX*s
            bbs[:,j,3]=trY*s
            bbs[:,j,4]=brX*s
            bbs[:,j,5]=brY*s
            bbs[:,j,6]=blX*s
            bbs[:,j,7]=blY*s
            #we add these for conveince to crop BBs within window
            bbs[:,j,8]=s*(tlX+blX)/2
            bbs[:,j,9]=s*(tlY+blY)/2
            bbs[:,j,10]=s*(trX+brX)/2
            bbs[:,j,11]=s*(trY+brY)/2
            bbs[:,j,12]=s*(tlX+trX)/2
            bbs[:,j,13]=s*(tlY+trY)/2
            bbs[:,j,14]=s*(brX+blX)/2
            bbs[:,j,15]=s*(brY+blY)/2
            bbs[:,j,16]=1 if not field else 0
            bbs[:,j,17]=1 if field else 0    
            j+=1
        return bbs


    def fixAnnotations(self,annotations):
        annotations['pairs']+=annotations['samePairs']
        toAdd=[]
        idsToRemove=set()

        #enumerations inside a row they are paired to should be removed
        #enumerations paired with the left row of a chained row need to be paired with the right
        pairsToRemove=[]
        pairsToAdd=[]
        for bb in annotations['textBBs']:
            if bb['type']=='textNumber':
                for pair in annotations['pairs']:
                    if bb['id'] in pair:
                        if pair[0]==bb['id']:
                            otherId=pair[1]
                        else:
                            otherId=pair[0]
                        otherBB=annotations['byId'][otherId]
                        if otherBB['type']=='fieldRow':
                            if avg_x(bb)>left_x(otherBB) and avg_x(bb)<right_x(otherBB):
                                idsToRemove.add(bb['id'])
                            #else TODO chained row case



        #remove fields we're skipping
        #reconnect para chains we broke by removing them
        #print('removing fields')
        idsToFix=[]
        for bb in annotations['fieldBBs']:
            id=bb['id']
            if self.isSkipField(bb):
                #print('remove {}'.format(id))
                idsToRemove.add(id)
                if bb['type']=='fieldP':
                    idsToFix.append(id)
            elif self.swapCircle and bb['type']=='fieldCircle':
                annotations['byId'][id]['type']='textCircle'
        
        parasLinkedTo=defaultdict(list)
        pairsToRemove=[]
        for i,pair in enumerate(annotations['pairs']):
            if pair[0] in idsToFix and annotations['byId'][pair[1]]['type'][-1]=='P':
                parasLinkedTo[pair[0]].append(pair[1])
                pairsToRemove.append(i)
            elif pair[1] in idsToFix and annotations['byId'][pair[0]]['type'][-1]=='P':
                parasLinkedTo[pair[1]].append(pair[0])
                pairsToRemove.append(i)
            elif pair[0] in idsToRemove or pair[1] in idsToRemove:
                pairsToRemove.append(i)

        pairsToRemove.sort(reverse=True)
        last=None
        for i in pairsToRemove:
            if i==last:#in case of duplicated
                continue
            #print('del pair: {}'.format(annotations['pairs'][i]))
            del annotations['pairs'][i]
            last=i
        for _,ids in parasLinkedTo.items():
            if len(ids)==2:
                if ids[0] not in idsToRemove and ids[1] not in idsToRemove:
                    #print('adding: {}'.format([ids[0],ids[1]]))
                    #annotations['pairs'].append([ids[0],ids[1]])
                    toAdd.append([ids[0],ids[1]])
            #else I don't know what's going on


        for id in idsToRemove:
            #print('deleted: {}'.format(annotations['byId'][id]))
            del annotations['byId'][id]


        #skipped link between col and enumeration when enumeration is between col header and col
        for pair in annotations['pairs']:
            notNum=num=None
            if pair[0] in annotations['byId'] and annotations['byId'][pair[0]]['type']=='textNumber':
                num=annotations['byId'][pair[0]]
                notNum=annotations['byId'][pair[1]]
            elif pair[1] in annotations['byId'] and annotations['byId'][pair[1]]['type']=='textNumber':
                num=annotations['byId'][pair[1]]
                notNum=annotations['byId'][pair[0]]

            if notNum is not None and notNum['type']!='textNumber':
                for pair2 in annotations['pairs']:
                    if notNum['id'] in pair2:
                        if notNum['id'] == pair2[0]:
                            otherId=pair2[1]
                        else:
                            otherId=pair2[0]
                        if annotations['byId'][otherId]['type']=='fieldCol' and avg_y(annotations['byId'][otherId])>avg_y(annotations['byId'][num['id']]):
                            toAdd.append([num['id'],otherId])

        #heirarchy labels.
        #for pair in annotations['samePairs']:
        #    text=textMinor=None
        #    if annotations['byId'][pair[0]]['type']=='text':
        #        text=pair[0]
        #        if annotations['byId'][pair[1]]['type']=='textMinor':
        #            textMinor=pair[1]
        #    elif annotations['byId'][pair[1]]['type']=='text':
        #        text=pair[1]
        #        if annotations['byId'][pair[0]]['type']=='textMinor':
        #            textMinor=pair[0]
        #    else:#catch case of minor-minor-field
        #        if annotations['byId'][pair[1]]['type']=='textMinor' and annotations['byId'][pair[0]]['type']=='textMinor':
        #            a=pair[0]
        #            b=pair[1]
        #            for pair2 in annotations['pairs']:
        #                if a in pair2:
        #                    if pair2[0]==a:
        #                        otherId=pair2[1]
        #                    else:
        #                        otherId=pair2[0]
        #                    toAdd.append([b,otherId])
        #                if b in pair2:
        #                    if pair2[0]==b:
        #                        otherId=pair2[1]
        #                    else:
        #                        otherId=pair2[0]
        #                    toAdd.append([a,otherId])

        #    
        #    if text is not None and textMinor is not None:
        #        for pair2 in annotations['pairs']:
        #            if textMinor in pair2:
        #                if pair2[0]==textMinor:
        #                    otherId=pair2[1]
        #                else:
        #                    otherId=pair2[0]
        #                toAdd.append([text,otherId])
        #        for pair2 in annotations['samePairs']:
        #            if textMinor in pair2:
        #                if pair2[0]==textMinor:
        #                    otherId=pair2[1]
        #                else:
        #                    otherId=pair2[0]
        #                if annotations['byId'][otherId]['type']=='textMinor':
        #                    toAddSame.append([text,otherId])

        for pair in toAdd:
            if pair not in annotations['pairs'] and [pair[1],pair[0]] not in annotations['pairs']:
                 annotations['pairs'].append(pair)
        #annotations['pairs']+=toAdd

    def isSkipField(self,bb):
        return (    (self.no_blanks and (bb['isBlank']=='blank' or bb['isBlank']==3)) or
                    (self.no_print_fields and (bb['isBlank']=='print' or bb['isBlank']==2)) or
                    #TODO no graphics
                    bb['type'] == 'fieldRow' or
                    bb['type'] == 'fieldCol' or
                    bb['type'] == 'fieldRegion'
                )
