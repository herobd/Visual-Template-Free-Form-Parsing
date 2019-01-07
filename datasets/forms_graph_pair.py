import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math
from utils.crop_transform import CropBoxTransform
from utils import augmentation
from collections import defaultdict, OrderedDict
from utils.forms_annotations import fixAnnotations, convertBBs, getBBWithPoints, getStartEndGT
import timeit

import cv2

SKIP=['174']#['193','194','197','200']
ONE_DONE=[]


def polyIntersect(poly1, poly2):
    prevPoint = poly1[-1]
    for point in poly1:
        perpVec = np.array([ -(point[1]-prevPoint[1]), point[0]-prevPoint[0] ])
        perpVec = perpVec/np.linalg.norm(perpVec)
        
        maxPoly1=np.dot(perpVec,poly1[0])
        minPoly1=maxPoly1
        for p in poly1:
            p_onLine = np.dot(perpVec,p)
            maxPoly1 = max(maxPoly1,p_onLine)
            minPoly1 = min(minPoly1,p_onLine)
        maxPoly2=np.dot(perpVec,poly2[0])
        minPoly2=maxPoly2
        for p in poly2:
            p_onLine = np.dot(perpVec,p)
            maxPoly2 = max(maxPoly2,p_onLine)
            minPoly2 = min(minPoly2,p_onLine)

        if (maxPoly1<minPoly2 or minPoly1>maxPoly2):
            return False
        prevPoint = point
    return True

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def lineIntersection(lineA, lineB, threshA_low=10, threshA_high=10, threshB_low=10, threshB_high=10, both=False):
    a1=lineA[0]
    a2=lineA[1]
    b1=lineB[0]
    b2=lineB[1]
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    point = (num / denom.astype(float))*db + b1
    #check if it is on atleast one line segment
    vecA = da/np.linalg.norm(da)
    p_A = np.dot(point,vecA)
    a1_A = np.dot(a1,vecA)
    a2_A = np.dot(a2,vecA)

    vecB = db/np.linalg.norm(db)
    p_B = np.dot(point,vecB)
    b1_B = np.dot(b1,vecB)
    b2_B = np.dot(b2,vecB)
    
    ###rint('A:{},  B:{}, int p:{}'.format(lineA,lineB,point))
    ###rint('{:.0f}>{:.0f} and {:.0f}<{:.0f}  and/or  {:.0f}>{:.0f} and {:.0f}<{:.0f} = {} {} {}'.format((p_A+threshA_low),(min(a1_A,a2_A)),(p_A-threshA_high),(max(a1_A,a2_A)),(p_B+threshB_low),(min(b1_B,b2_B)),(p_B-threshB_high),(max(b1_B,b2_B)),(p_A+threshA_low>min(a1_A,a2_A) and p_A-threshA_high<max(a1_A,a2_A)),'and' if both else 'or',(p_B+threshB_low>min(b1_B,b2_B) and p_B-threshB_high<max(b1_B,b2_B))))
    if both:
        if ( (p_A+threshA_low>min(a1_A,a2_A) and p_A-threshA_high<max(a1_A,a2_A)) and
             (p_B+threshB_low>min(b1_B,b2_B) and p_B-threshB_high<max(b1_B,b2_B)) ):
            return point
    else:
        if ( (p_A+threshA_low>min(a1_A,a2_A) and p_A-threshA_high<max(a1_A,a2_A)) or
             (p_B+threshB_low>min(b1_B,b2_B) and p_B-threshB_high<max(b1_B,b2_B)) ):
            return point
    return None

def collate(batch):
    assert(len(batch)==1)
    return batch[0]


class FormsGraphPair(torch.utils.data.Dataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        #if 'augmentation_params' in config['data_loader']:
        #    self.augmentation_params=config['augmentation_params']
        #else:
        #    self.augmentation_params=None
        self.cropToPage=config['crop_to_page']
        self.rotate = config['rotation'] if 'rotation' in config else True
        #patchSize=config['patch_size']
        if 'crop_params' in config:
            self.transform = CropBoxTransform(config['crop_params'],self.rotate)
        else:
            self.transform = None
        self.rescale_range = config['rescale_range']
        if self.rescale_range[0]==450:
            self.rescale_range[0]=0.2
        elif self.rescale_range[0]>1.0:
            self.rescale_range[0]=0.27
        if self.rescale_range[1]==800:
            self.rescale_range[1]=0.33
        elif self.rescale_range[1]>1.0:
            self.rescale_range[1]=0.27
        if 'cache_resized_images' in config:
            self.cache_resized = config['cache_resized_images']
            if self.cache_resized:
                self.cache_path = os.path.join(dirPath,'cache_'+str(self.rescale_range[1]))
                if not os.path.exists(self.cache_path):
                    os.mkdir(self.cache_path)
        else:
            self.cache_resized = False
        self.pixel_count_thresh = config['pixel_count_thresh'] if 'pixel_count_thresh' in config else 10000000
        self.max_dim_thresh = config['max_dim_thresh'] if 'max_dim_thresh' in config else 2700
        if 'only_types' in config:
            self.only_types = config['only_types']
        else:
            self.only_types=None
        #print( self.only_types)
        if 'swap_circle' in config:
            self.swapCircle = config['swap_circle']
        else:
            self.swapCircle = False
        self.color = config['color'] if 'color' in config else True

        self.simple_dataset = config['simple_dataset'] if 'simple_dataset' in config else False

        if images is not None:
            self.images=images
        else:
            if self.simple_dataset:
                splitFile = 'simple_train_valid_test_split.json'
            else:
                splitFile = 'train_valid_test_split.json'
            with open(os.path.join(dirPath,splitFile)) as f:
                #if split=='valid' or split=='validation':
                #    trainTest='train'
                #else:
                #    trainTest=split
                groupsToUse = json.loads(f.read())[split]
            self.images=[]
            groupNames = list(groupsToUse.keys())
            groupNames.sort()
            for groupName in groupNames:
                imageNames=groupsToUse[groupName]
                #print('{} {}'.format(groupName, imageNames))
                #oneonly=False
                if groupName in SKIP:
                    print('Skipped group {}'.format(groupName))
                    continue
                #    if groupName in ONE_DONE:
                #        oneonly=True
                #        with open(os.path.join(dirPath,'groups',groupName,'template'+groupName+'.json')) as f:
                #            T_annotations = json.loads(f.read())
                #    else:
                for imageName in imageNames:
                    #if oneonly and T_annotations['imageFilename']!=imageName:
                    #    #print('skipped {} {}'.format(imageName,groupName))
                    #    continue
                    #elif oneonly:
                    #    print('only {} from {}'.format(imageName,groupName))
                    org_path = os.path.join(dirPath,'groups',groupName,imageName)
                    if self.cache_resized:
                        path = os.path.join(self.cache_path,imageName)
                    else:
                        path = org_path
                    jsonPath = org_path[:org_path.rfind('.')]+'.json'
                    #print(jsonPath)
                    if os.path.exists(jsonPath):
                        rescale=1.0
                        if self.cache_resized:
                            rescale = self.rescale_range[1]
                            if not os.path.exists(path):
                                org_img = cv2.imread(org_path)
                                if org_img is None:
                                    print('WARNING, could not read {}'.format(org_img))
                                    continue
                                #target_dim1 = self.rescale_range[1]
                                #target_dim0 = int(org_img.shape[0]/float(org_img.shape[1]) * target_dim1)
                                #resized = cv2.resize(org_img,(target_dim1, target_dim0), interpolation = cv2.INTER_CUBIC)
                                resized = cv2.resize(org_img,(0,0),
                                        fx=self.rescale_range[1], 
                                        fy=self.rescale_range[1], 
                                        interpolation = cv2.INTER_CUBIC)
                                cv2.imwrite(path,resized)
                                #rescale = target_dim1/float(org_img.shape[1])
                        #elif self.cache_resized:
                            #print(jsonPath)
                            #with open(jsonPath) as f:
                            #    annotations = json.loads(f.read())
                            #imW = annotations['width']

                            #target_dim1 = self.rescale_range[1]
                            #rescale = target_dim1/float(imW)
                        #print('addint {}'.format(imageName))
                        self.images.append({'id':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':imageName[:imageName.rfind('.')]})
                    #else:
                    #    print('couldnt find {}'.format(jsonPath))
                            
                        # with open(path+'.json') as f:
                        #    annotations = json.loads(f.read())
                        #    imH = annotations['height']
                        #    imW = annotations['width']
                        #    #startCount=len(self.instances)
                        #    for bb in annotations['textBBs']:
        
        self.no_blanks = config['no_blanks'] if 'no_blanks' in config else False
        self.use_paired_class = config['use_paired_class'] if 'use_paired_class' in config else False
        if 'no_print_fields' in config:
            self.no_print_fields = config['no_print_fields']
        else:
            self.no_print_fields = False
        self.no_graphics =  config['no_graphics'] if 'no_graphics' in config else False
        self.only_opposite_pairs = config['only_opposite_pairs'] if 'only_opposite_pairs' in config else False
        self.onlyFormStuff = False
        self.errors=[]




    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        return self.getitem(index)
    def getitem(self,index,scaleP=None,cropPoint=None):
        ##ticFull=timeit.default_timer()
        imagePath = self.images[index]['imagePath']
        imageName = self.images[index]['imageName']
        annotationPath = self.images[index]['annotationPath']
        #print(annotationPath)
        rescaled = self.images[index]['rescaled']
        with open(annotationPath) as annFile:
            annotations = json.loads(annFile.read())
        fieldBBs = annotations['fieldBBs']
        fixAnnotations(self,annotations)

        ##tic=timeit.default_timer()
        np_img = cv2.imread(imagePath, 1 if self.color else 0)#/255.0
        if np_img is None or np_img.shape[0]==0:
            print("ERROR, could not open "+imagePath)
            return self.__getitem__((index+1)%self.__len__())
        ##print('imread: {}  [{}, {}]'.format(timeit.default_timer()-tic,np_img.shape[0],np_img.shape[1]))
        ##print('       channels : {}'.format(len(np_img.shape)))
        if self.cropToPage:
            print('Not implemented')
            assert(False)
            pageCorners = annotations['page_corners']
            xl = max(0,int(rescaled*min(pageCorners['tl'],pageCorners['bl'])))
            xr = min(np_img.shape[1]-1,int(rescaled*max(pageCorners['tr'],pageCorners['br'])))
            yt = max(0,int(rescaled*min(pageCorners['tl'],pageCorners['tr'])))
            yb = min(np_img.shape[0]-1,int(rescaled*max(pageCorners['bl'],pageCorners['br'])))
            np_img = np_img[yt:yb+1,xl:xr+1,:]
        #target_dim1 = int(np.random.uniform(self.rescale_range[0], self.rescale_range[1]))
        if scaleP is None:
            s = np.random.uniform(self.rescale_range[0], self.rescale_range[1])
        else:
            s = scaleP
        partial_rescale = s/rescaled
        if self.transform is None: #we're doing the whole image
            #this is a check to be sure we don't send too big images through
            pixel_count = partial_rescale*partial_rescale*np_img.shape[0]*np_img.shape[1]
            if pixel_count > self.pixel_count_thresh:
                partial_rescale = math.sqrt(partial_rescale*partial_rescale*self.pixel_count_thresh/pixel_count)
                print('{} exceed thresh: {}: {}, new {}: {}'.format(imageName,s,pixel_count,rescaled*partial_rescale,partial_rescale*partial_rescale*np_img.shape[0]*np_img.shape[1]))
                s = rescaled*partial_rescale


            max_dim = partial_rescale*max(np_img.shape[0],np_img.shape[1])
            if max_dim > self.max_dim_thresh:
                partial_rescale = partial_rescale*(self.max_dim_thresh/max_dim)
                print('{} exceed thresh: {}: {}, new {}: {}'.format(imageName,s,max_dim,rescaled*partial_rescale,partial_rescale*max(np_img.shape[0],np_img.shape[1])))
                s = rescaled*partial_rescale

        
        
        ##tic=timeit.default_timer()
        #np_img = cv2.resize(np_img,(target_dim1, target_dim0), interpolation = cv2.INTER_CUBIC)
        np_img = cv2.resize(np_img,(0,0),
                fx=partial_rescale,
                fy=partial_rescale,
                interpolation = cv2.INTER_CUBIC)
        if not self.color:
            np_img=np_img[...,None] #add 'color' channel
        ##print('resize: {}  [{}, {}]'.format(timeit.default_timer()-tic,np_img.shape[0],np_img.shape[1]))
        
        ##tic=timeit.default_timer()
        bbsToUse=[]
        ids=[]
        for id,bb in annotations['byId'].items():
            if not self.onlyFormStuff or ('paired' in bb and bb['paired']):
                bbsToUse.append(bb)
                ids.append(bb['id'])


        
        bbs = getBBWithPoints(bbsToUse,s,useBlankClass=(not self.no_blanks),usePairedClass=self.use_paired_class)
        numClasses = bbs.shape[2]-16
        #start_of_line, end_of_line = getStartEndGT(annotations['byId'].values(),s)
        #Try:
        #    table_points, table_pixels = self.getTables(
        #            fieldBBs,
        #            s, 
        #            np_img.shape[0], 
        #            np_img.shape[1],
        #            annotations['samePairs'])
        #Except Exception as inst:
        #    if imageName not in self.errors:
        #        table_points=None
        #        table_pixels=None
        #        print(inst)
        #        print('Table error on: '+imagePath)
        #        self.errors.append(imageName)


        #pixel_gt = table_pixels

        ##ticTr=timeit.default_timer()
        if self.transform is not None:
            out, cropPoint = self.transform({
                "img": np_img,
                "bb_gt": bbs,
                'bb_ids':ids,
                #"line_gt": {
                #    "start_of_line": start_of_line,
                #    "end_of_line": end_of_line
                #    },
                #"point_gt": {
                #        "table_points": table_points
                #        },
                #"pixel_gt": pixel_gt,
                
            }, cropPoint)
            np_img = out['img']
            bbs = out['bb_gt']
            ids = out['bb_ids']
            #if 'table_points' in out['point_gt']:
            #    table_points = out['point_gt']['table_points']
            #else:
            #    table_points=None
            #pixel_gt = out['pixel_gt']
            #start_of_line = out['line_gt']['start_of_line']
            #end_of_line = out['line_gt']['end_of_line']

            ##tic=timeit.default_timer()
            if np_img.shape[2]==3:
                np_img = augmentation.apply_random_color_rotation(np_img)
                np_img = augmentation.apply_tensmeyer_brightness(np_img)
            else:
                np_img = augmentation.apply_tensmeyer_brightness(np_img)
            ##print('augmentation: {}'.format(timeit.default_timer()-tic))
        ##print('transfrm: {}  [{}, {}]'.format(timeit.default_timer()-ticTr,org_img.shape[0],org_img.shape[1]))
        pairs=set()
        index1=0
        #import pdb;pdb.set_trace()
        for id in ids: #updated
            responseBBList = self.__getResponseBBList(id,annotations)
            for bb in responseBBList:
                try:
                    index2 = ids.index(bb['id'])
                    #adjMatrix[min(index1,index2),max(index1,index2)]=1
                    pairs.add((min(index1,index2),max(index1,index2)))
                except ValueError:
                    pass
            index1+=1
        #ones = torch.ones(len(pairs))
        #if len(pairs)>0:
        #    pairs = torch.LongTensor(list(pairs)).t()
        #else:
        #    pairs = torch.LongTensor(pairs)
        #adjMatrix = torch.sparse.FloatTensor(pairs,ones,(len(ids),len(ids))) # This is an upper diagonal matrix as pairings are bi-directional

        #if len(np_img.shape)==2:
        #    img=np_img[None,None,:,:] #add "color" channel and batch
        #else:
        img = np_img.transpose([2,0,1])[None,...] #from [row,col,color] to [batch,color,row,col]
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = 1.0 - img / 128.0 #ideally the median value would be 0
        #if pixel_gt is not None:
        #    pixel_gt = pixel_gt.transpose([2,0,1])[None,...]
        #    pixel_gt = torch.from_numpy(pixel_gt)

        #start_of_line = None if start_of_line is None or start_of_line.shape[1] == 0 else torch.from_numpy(start_of_line)
        #end_of_line = None if end_of_line is None or end_of_line.shape[1] == 0 else torch.from_numpy(end_of_line)
        
        bbs = convertBBs(bbs,self.rotate,numClasses)

        #if table_points is not None:
        #    table_points = None if table_points.shape[1] == 0 else torch.from_numpy(table_points)

        return {
                "img": img,
                "bb_gt": bbs,
                "adj": pairs,#adjMatrix,
                "imgName": imageName,
                "scale": s,
                "cropPoint": cropPoint
                }




    def __getResponseBBList(self,queryId,annotations):
        responseBBList=[]
        for pair in annotations['pairs']: #done already +annotations['samePairs']:
            if queryId in pair:
                if pair[0]==queryId:
                    otherId=pair[1]
                else:
                    otherId=pair[0]
                if otherId in annotations['byId'] and (not self.onlyFormStuff or ('paired' in bb and bb['paired'])):
                    responseBBList.append(annotations['byId'][otherId])
                #if not self.isSkipField(annotations['byId'][otherId]):
                #    poly = np.array(annotations['byId'][otherId]['poly_points']) #self.__getResponseBB(otherId,annotations)  
                #    responseBBList.append(poly)
        return responseBBList


def getWidthFromBB(bb):
    return (np.linalg.norm(bb[0]-bb[1]) + np.linalg.norm(bb[3]-bb[2]))/2
def getHeightFromBB(bb):
    return (np.linalg.norm(bb[0]-bb[3]) + np.linalg.norm(bb[1]-bb[2]))/2

#This iterates over each part of the columns, seeing if their seperator lines intersect the argument line
##We assume that an intersection must occur (we don't have row components that don't intersect any columns)
def getIntersectsCols(line,cols,startInd,threshLine_low=10,threshLine_high=10,threshLeft=float('inf'),threshRight=float('inf'),failed=0):
    if startInd>0:
        startInd-=1
        tryBefore=True
    else:
        tryBefore=False
    intersectionThresh=20
    intersectionBoth=True
    if failed==1:
        intersectionThresh=40
    elif failed==2:
        intersectionBoth=False
    elif failed>2:
        return [], 0, tryBefore

    #left-most boundary
    p=None
    if startInd==0:
        j=0
        for lineComponent in cols[0]:
            ###rint('first, j:{}, failed:{}'.format(j,failed))
            width = getWidthFromBB(lineComponent)
            p = lineIntersection(line,[lineComponent[0],lineComponent[3]], 
                    threshA_low=threshLeft, #float("inf"), 
                    threshA_high=width/2, 
                    threshB_low=threshLine_low if j==len(cols[0])-1 or threshLine_low!=float('inf') else 10,
                    threshB_high=threshLine_high if j==len(cols[0])-1 or threshLine_high!=float('inf') else 10,
                    both=intersectionBoth)
            j+=1
            if p is not None:
                break
        if p is None:
            if tryBefore:
                tryBefore=False
                startInd=1
                iPoints=[]
            elif failed==2:
                return [], 0, tryBefore
            else:
                return getIntersectsCols(line,cols,startInd,threshLine_low,threshLine_high,threshLeft,threshRight,failed+1)
        else:
            iPoints=[p]
            startInd=1
            #tryBefore=False
    else:
        iPoints=[]

    done=False
    i = startInd-1 #in case the for-loop doesn't run at all
    for i in range(startInd-1,len(cols)-1):
        #if i==(startInd):
            #tryBefore=False
        avgWidth_ip1=0
        for lineComponent in cols[i+1]:
            width = getWidthFromBB(lineComponent)
            avgWidth_ip1+=width
        avgWidth_ip1/=len(cols[i+1])
        avgHWidth_ip1=avgWidth_ip1/2
        pL=pR=None
        avgWidth_i=0
        j=0
        for lineComponent in cols[i]:
            ###rint('L i:{}, j:{}, failed:{}'.format(i,j,failed))
            width = getWidthFromBB(lineComponent)
            avgWidth_i+=width
            pL = lineIntersection(line,lineComponent[1:3], 
                    threshA_low=width/2, 
                    threshA_high=avgHWidth_ip1, 
                    threshB_low=threshLine_low if j==len(cols[0])-1 or threshLine_low!=float('inf') else 10,
                    threshB_high=threshLine_high if j==len(cols[0])-1 or threshLine_high!=float('inf') else 10,
                    both=intersectionBoth)
            j+=1
            if pL is not None:
                break
        avgWidth_i/=len(cols[i])
        avgHWidth_i=avgWidth_i/2
        j=0
        for lineComponent in cols[i+1]:
            ###rint('R i:{}, j:{}, failed:{}'.format(i,j,failed))
            pR = lineIntersection(line,[lineComponent[0],lineComponent[3]], 
                    threshA_low=avgHWidth_i, 
                    threshA_high=width/2, 
                    threshB_low=threshLine_low if j==len(cols[0])-1 or threshLine_low!=float('inf') else 10,
                    threshB_high=threshLine_high if j==len(cols[0])-1 or threshLine_high!=float('inf') else 10,
                    both=intersectionBoth)
            j+=1
            if pR is not None:
                break
        #print('pL {}'.format(pL))
        #print('pR {}'.format(pR))
        #print('failed {}, i={}, line={}'.format(failed,i,line))
        #assert((pL is None) == (pR is None))
        if (pL is None) and (pR is None):
            if tryBefore and i==startInd-1:
                tryBefore=False
                continue
            else:
                done=True
                break
        elif pL is None:
            iPoints.append(pR)
        elif pR is None:
            iPoints.append(pL)
        else:
            iPoints.append((pL+pR)/2.0)
    if not done:
        #right-most boundary
        j=0
        for lineComponent in cols[-1]:
            ###rint('last, j:{}, failed:{}'.format(j,failed))
            width = getWidthFromBB(lineComponent)
            p = lineIntersection(line,lineComponent[1:3], 
                    threshA_low=width/2, 
                    threshA_high=threshRight, #float('inf'), 
                    threshB_low=threshLine_low if j==len(cols[0])-1 or threshLine_low!=float('inf') else 10,
                    threshB_high=threshLine_high if j==len(cols[0])-1 or threshLine_high!=float('inf') else 10,
                    both=intersectionBoth)
            j+=1
            if p is not None:
                iPoints.append(p)
                i = len(cols)+1
                break
            else:
                i=len(cols)
    else:
        i+=1
    if len(iPoints)>0 or failed==2:
        return iPoints,i,tryBefore
    else:
        return getIntersectsCols(line,cols,startInd,threshLine_low,threshLine_high,threshLeft,threshRight,failed+1)


