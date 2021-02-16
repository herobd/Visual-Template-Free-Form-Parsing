import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math
from collections import defaultdict, OrderedDict
from utils.funsd_annotations import createLines
import timeit
from .graph_pair import GraphPairDataset

import utils.img_f as img_f

SKIP=['174']#['193','194','197','200']
ONE_DONE=[]


def collate(batch):
    assert(len(batch)==1)
    return batch[0]


class FUNSDGraphPair(GraphPairDataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(FUNSDGraphPair, self).__init__(dirPath,split,config,images)

        self.only_types=None

        self.split_to_lines = config['split_to_lines']

        if images is not None:
            self.images=images
        else:
            if 'overfit' in config and config['overfit']:
                splitFile = 'overfit_split.json'
            else:
                splitFile = 'train_valid_test_split.json'
            with open(os.path.join(dirPath,splitFile)) as f:
                #if split=='valid' or split=='validation':
                #    trainTest='train'
                #else:
                #    trainTest=split
                readFile = json.loads(f.read())
                if type(split) is str:
                    toUse = readFile[split]
                    imagesAndAnn = []
                    imageDir = os.path.join(dirPath,toUse['root'],'images')
                    annDir = os.path.join(dirPath,toUse['root'],'annotations')
                    for name in toUse['images']:
                        imagesAndAnn.append( (name+'.png',os.path.join(imageDir,name+'.png'),os.path.join(annDir,name+'.json')) )
                elif type(split) is list:
                    imagesAndAnn = []
                    for spstr in split:
                        toUse = readFile[spstr]
                        imageDir = os.path.join(dirPath,toUse['root'],'images')
                        annDir = os.path.join(dirPath,toUse['root'],'annotations')
                        for name in toUse['images']:
                            imagesAndAnn.append( (name+'.png',os.path.join(imageDir,name+'.png'),os.path.join(annDir,name+'.json')) )
                else:
                    print("Error, unknown split {}".format(split))
                    exit()
            self.images=[]
            for imageName,imagePath,jsonPath in imagesAndAnn:
                org_path = imagePath
                if self.cache_resized:
                    path = os.path.join(self.cache_path,imageName)
                else:
                    path = org_path
                if os.path.exists(jsonPath):
                    rescale=1.0
                    if self.cache_resized:
                        rescale = self.rescale_range[1]
                        if not os.path.exists(path):
                            org_img = img_f.imread(org_path)
                            if org_img is None:
                                print('WARNING, could not read {}'.format(org_img))
                                continue
                            resized = img_f.resize(org_img,(0,0),
                                    fx=self.rescale_range[1], 
                                    fy=self.rescale_range[1], 
                                    )
                            img_f.imwrite(path,resized)
                    self.images.append({'id':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':imageName[:imageName.rfind('.')]})
        self.only_types=None
        self.errors=[]

        self.classMap={
                'header':16,
                'question':17,
                'answer': 18,
                'other': 19
                }





    def parseAnn(self,annotations,s):
        #if useBlankClass:
        #    numClasses+=1
        #if usePairedClass:
        #    numClasses+=1

        numClasses=len(self.classMap)
        if self.split_to_lines:
            bbs, numNeighbors, trans, groups = createLines(annotations,self.classMap,s)
        else:
            boxes = annotations['form']
            bbs = np.empty((1,len(boxes), 8+8+numClasses), dtype=np.float32) #2x4 corners, 2x4 cross-points, n classes
            #pairs=set()
            numNeighbors=[]
            trans=[]
            for j,boxinfo in enumerate(boxes):
                lX,tY,rX,bY = boxinfo['box']
                h=bY-tY
                w=rX-lX
                if h/w>5 and self.rotate: #flip labeling, since FUNSD doesn't label verticle text correctly
                    #I don't know if it needs rotated clockwise or countercw, so I just say countercw
                    bbs[:,j,0]=lX*s
                    bbs[:,j,1]=bY*s
                    bbs[:,j,2]=lX*s
                    bbs[:,j,3]=tY*s
                    bbs[:,j,4]=rX*s
                    bbs[:,j,5]=tY*s
                    bbs[:,j,6]=rX*s
                    bbs[:,j,7]=bY*s
                    #we add these for conveince to crop BBs within window
                    bbs[:,j,8]=s*(lX+rX)/2.0
                    bbs[:,j,9]=s*bY
                    bbs[:,j,10]=s*(lX+rX)/2.0
                    bbs[:,j,11]=s*tY
                    bbs[:,j,12]=s*lX
                    bbs[:,j,13]=s*(tY+bY)/2.0
                    bbs[:,j,14]=s*rX
                    bbs[:,j,15]=s*(tY+bY)/2.0
                else:
                    bbs[:,j,0]=lX*s
                    bbs[:,j,1]=tY*s
                    bbs[:,j,2]=rX*s
                    bbs[:,j,3]=tY*s
                    bbs[:,j,4]=rX*s
                    bbs[:,j,5]=bY*s
                    bbs[:,j,6]=lX*s
                    bbs[:,j,7]=bY*s
                    #we add these for conveince to crop BBs within window
                    bbs[:,j,8]=s*lX
                    bbs[:,j,9]=s*(tY+bY)/2.0
                    bbs[:,j,10]=s*rX
                    bbs[:,j,11]=s*(tY+bY)/2.0
                    bbs[:,j,12]=s*(lX+rX)/2.0
                    bbs[:,j,13]=s*tY
                    bbs[:,j,14]=s*(rX+lX)/2.0
                    bbs[:,j,15]=s*bY
                
                bbs[:,j,16:]=0
                if boxinfo['label']=='header':
                    bbs[:,j,16]=1
                elif boxinfo['label']=='question':
                    bbs[:,j,17]=1
                elif boxinfo['label']=='answer':
                    bbs[:,j,18]=1
                elif boxinfo['label']=='other':
                    bbs[:,j,19]=1
                #for id1,id2 in boxinfo['linking']:
                #    pairs.add((min(id1,id2),max(id1,id2)))
                trans.append(boxinfo['text'])
                numNeighbors.append(len(boxinfo['linking']))
            groups = [[n] for n in range(len(boxes))]


        #self.pairs=list(pairs)
        return bbs, list(range(bbs.shape[1])), numClasses, trans, groups, {}


    def getResponseBBIdList(self,queryId,annotations):
        if self.split_to_lines:
            return annotations['linking'][queryId]
        else:
            boxes=annotations['form']
            cto=[]
            boxinfo = boxes[queryId]
            for id1,id2 in boxinfo['linking']:
                if id1==queryId:
                    cto.append(id2)
                else:
                    cto.append(id1)
            return cto




def getWidthFromBB(bb):
    return (np.linalg.norm(bb[0]-bb[1]) + np.linalg.norm(bb[3]-bb[2]))/2
def getHeightFromBB(bb):
    return (np.linalg.norm(bb[0]-bb[3]) + np.linalg.norm(bb[1]-bb[2]))/2



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

