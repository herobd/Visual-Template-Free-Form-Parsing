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
from .box_detect import BoxDetectDataset, collate
from utils.funsd_annotations import createLines
import timeit

import utils.img_f as img_f

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


class FUNSDBoxDetect(BoxDetectDataset):
    """
    Class for reading FUNSD dataset (with custom split file) 
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(FUNSDBoxDetect,self).__init__(dirPath,split,config,images)
        #if 'augmentation_params' in config['data_loader']:
        #    self.augmentation_params=config['augmentation_params']
        #else:
        #    self.augmentation_params=None
        self.split_to_lines = config['split_to_lines']
        if images is not None:
            self.images=images
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
        self.no_blanks=True #too bad


    def parseAnn(self,np_img,annotations,s,imageName):
        if self.split_to_lines:
            bbs, numNeighbors, trans, groups = createLines(annotations,self.classMap,s)
            numClasses = len(self.classMap)
            pairs=None
        else:
            boxes = annotations['form']
            numClasses=4
            #if useBlankClass:
            #    numClasses+=1
            #if usePairedClass:
            #    numClasses+=1

            bbs = np.empty((1,len(boxes), 8+8+numClasses), dtype=np.float32) #2x4 corners, 2x4 cross-points, n classes
            pairs=set()
            numNeighbors=[]
            for j,boxinfo in enumerate(boxes):
                lX,tY,rX,bY = boxinfo['box']
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
                for id1,id2 in boxinfo['linking']:
                    pairs.add((min(id1,id2),max(id1,id2)))
                numNeighbors.append(len(boxinfo['linking']))


            pairs=list(pairs)
            

        pixel_gt = None

        line_gts = None#{
                #"start_of_line": None,
                    #"end_of_line": None
                    #}
        point_gts = None#{
                        #"table_points": None
                        #}

        return bbs,line_gts,point_gts,pixel_gt,numClasses,numNeighbors, pairs





    def cluster(self,k,sample_count,outPath,use_max_width=False):
        def makePointsAndRects(h,w,r=None):
            if r is None:
                return np.array([-w/2.0,0,w/2.0,0,0,-h/2.0,0,h/2.0, 0,0, 0, h,w])
            else:
                lx= -math.cos(r)*w
                ly= -math.sin(r)*w
                rx= math.cos(r)*w
                ry= math.sin(r)*w
                tx= math.sin(r)*h
                ty= -math.cos(r)*h
                bx= -math.sin(r)*h
                by= math.cos(r)*h
                return np.array([lx,ly,rx,ry,tx,ty,bx,by, 0,0, r, h,w])
        meanH=None
        stdH=None
        meanW=None
        stdW=None
        ratios=[4.0,7.18,11.0,15.0,19.0,27.0]
        pointsAndRects=[]
        for inst in self.images:
            annotationPath = inst['annotationPath']
            #rescaled = inst['rescaled']
            with open(annotationPath) as annFile:
                annotations = json.loads(annFile.read())
            for i in range(sample_count):
                if i==0:
                    s = (self.rescale_range[0]+self.rescale_range[1])/2
                else:
                    s = np.random.uniform(self.rescale_range[0], self.rescale_range[1])
                #partial_rescale = s/rescaled
                points=[]
                if self.split_to_lines:
                    bbs, numNeighbors, trans, groups = createLines(annotations,self.classMap,s)
                    for j in range(bbs.shape[1]):
                        w = bbs[0,j,2]-bbs[0,j,0]
                        h = bbs[0,j,5]-bbs[0,j,1]
                        if use_max_width:
                            w = min(w,use_max_width)
                        p_left_x = bbs[0,j,8]
                        p_left_x = -w/2
                        p_left_y = 0
                        p_right_x = w/2
                        p_right_y = 0
                        p_top_x = 0
                        p_top_y = -h/2
                        p_bot_x = 0
                        p_bot_y = h/2
                        points.append(np.array([p_left_x,p_left_y,p_right_x,p_right_y,p_top_x,p_top_y,p_bot_x,p_bot_y,0,0,0,h,w]))
                else:
                    boxes = annotations['form']
                    for j,boxinfo in enumerate(boxes):
                        lX,tY,rX,bY = boxinfo['box']
                        w = rX-lX +1
                        h = bY-tY +1
                        w *= s
                        h *= s
                        if use_max_width:
                            w = min(w,use_max_width)
                        p_left_x = -w/2
                        p_left_y = 0
                        p_right_x = w/2
                        p_right_y = 0
                        p_top_x = 0
                        p_top_y = -h/2
                        p_bot_x = 0
                        p_bot_y = h/2
                        points.append(np.array([p_left_x,p_left_y,p_right_x,p_right_y,p_top_x,p_top_y,p_bot_x,p_bot_y,0,0,0,h,w]))
                points = np.stack(points,axis=0)
                pointsAndRects.append(points)
        pointsAndRects = np.concatenate(pointsAndRects,axis=0)
        #all_points = pointsAndRects[:,0:8]
        #all_heights = pointsAndRects[:,11]
        #all_widths = pointsAndRects[:,12]
        
        bestDistsFromMean=None
        num_attempts = 20 if k>0 else 1
        for attempt in range(num_attempts):
            if k>0:
                randomIndexes = np.random.randint(0,pointsAndRects.shape[0],(k))
                means=pointsAndRects[randomIndexes]
            else:
                #minH=5
                #minW=5
                means=[]

                ##smaller than mean
                #for step in range(5):
                #    height = minH + (meanH-minH)*(step/5.0)
                #    width = minW + (meanW-minW)*(step/5.0)
                #    for ratio in ratios:
                #        means.append(makePointsAndRects(height,ratio*height))
                #        means.append(makePointsAndRects(width/ratio,width))
                #for stddev in range(0,5):
                #    for step in range(5-stddev):
                #        height = meanH + stddev*stdH + stdH*(step/(5.0-stddev))
                #        width = meanW + stddev*stdW + stdW*(step/(5.0-stddev))
                #        for ratio in ratios:
                #            means.append(makePointsAndRects(height,ratio*height))
                #            means.append(makePointsAndRects(width/ratio,width))
                rots = [0,math.pi/2,math.pi,1.5*math.pi]
                if self.rotate:
                    for height in np.linspace(15,200,num=4):
                        for width in np.linspace(30,1200,num=4):
                            for rot in rots:
                                means.append(makePointsAndRects(height,width,rot))
                        #long boxes
                    for width in np.linspace(1600,4000,num=3):
                        #for height in np.linspace(30,100,num=3):
                        #    for rot in rots:
                        #        means.append(makePointsAndRects(height,width,rot))
                        for rot in rots:
                            means.append(makePointsAndRects(50,width,rot))
                else:
                    #rotated boxes
                    #for height in np.linspace(13,300,num=4):
                    for height in np.linspace(13,300,num=3):
                        means.append(makePointsAndRects(height,20))
                    #general boxes
                    #for height in np.linspace(15,200,num=4):
                        #for width in np.linspace(30,1200,num=4):
                    for height in np.linspace(15,200,num=2):
                        for width in np.linspace(30,1200,num=3):
                            means.append(makePointsAndRects(height,width))
                    #long boxes
                    for width in np.linspace(1600,4000,num=3):
                        #for height in np.linspace(30,100,num=3):
                        #    means.append(makePointsAndRects(height,width))
                        means.append(makePointsAndRects(50,width))

                k=len(means)
                print('K: {}'.format(k))
                means = np.stack(means,axis=0)
            #pointsAndRects [0:p_left_x, 1:p_left_y,2:p_right_x,3:p_right_y,4:p_top_x,5:p_top_y,6:p_bot_x,7:p_bot_y, 8:xc, 9:yc, 10:rot, 11:h, 12:w
            cluster_centers=means
            distsFromMean=None
            prevDistsFromMean=None
            for iteration in range(100000): #intended to break out
                print('attempt:{}/{}, bestDistsFromMean:{}, iteration:{}, bestDistsFromMean:{}'.format(attempt+1,num_attempts,bestDistsFromMean,iteration,prevDistsFromMean), end='\r')
                #means_points = means[:,0:8]
                #means_heights = means[:,11]
                #means_widths = means[:,12]
                # = groups = assignGroups(means,pointsAndRects)
                expanded_all_points = pointsAndRects[:,None,0:8]
                expanded_all_heights = pointsAndRects[:,None,11]
                expanded_all_widths = pointsAndRects[:,None,12]

                expanded_means_points = means[None,:,0:8]
                expanded_means_heights = means[None,:,11]
                expanded_means_widths = means[None,:,12]

                #expanded_all_points = expanded_all_points.expand(all_points.shape[0], all_points.shape[1], means_points.shape[1], all_points.shape[2])
                expanded_all_points = np.tile(expanded_all_points,(1,means.shape[0],1))
                expanded_all_heights = np.tile(expanded_all_heights,(1,means.shape[0]))
                expanded_all_widths = np.tile(expanded_all_widths,(1,means.shape[0]))
                #expanded_means_points = expanded_means_points.expand(means_points.shape[0], all_points.shape[0], means_points.shape[0], means_points.shape[2])
                expanded_means_points = np.tile(expanded_means_points,(pointsAndRects.shape[0],1,1))
                expanded_means_heights = np.tile(expanded_means_heights,(pointsAndRects.shape[0],1))
                expanded_means_widths = np.tile(expanded_means_widths,(pointsAndRects.shape[0],1))

                point_deltas = (expanded_all_points - expanded_means_points)
                #avg_heights = ((expanded_means_heights+expanded_all_heights)/2)
                #avg_widths = ((expanded_means_widths+expanded_all_widths)/2)
                avg_heights=avg_widths = (expanded_means_heights+expanded_all_heights+expanded_means_widths+expanded_all_widths)/4
                #print point_deltas

                normed_difference = (
                    np.linalg.norm(point_deltas[:,:,0:2],2,2)/avg_widths +
                    np.linalg.norm(point_deltas[:,:,2:4],2,2)/avg_widths +
                    np.linalg.norm(point_deltas[:,:,4:6],2,2)/avg_heights +
                    np.linalg.norm(point_deltas[:,:,6:8],2,2)/avg_heights
                    )**2
                #print normed_difference
                #import pdb; pdb.set_trace()

                groups = normed_difference.argmin(1) #this should list the mean (index) for each element of all
                distsFromMean = normed_difference.min(1).mean()
                if prevDistsFromMean is not None and distsFromMean >= prevDistsFromMean:
                    break
                prevDistsFromMean = distsFromMean

                #means = computeMeans(groups,pointsAndRects)
                #means = np.zeros(k,13)
                for ki in range(k):
                    selected = (groups==ki)[:,None]
                    numSel = float(selected.sum())
                    if (numSel==0):
                        break
                    means[ki,:] = (pointsAndRects*np.tile(selected,(1,13))).sum(0)/numSel
            if bestDistsFromMean is None or distsFromMean<bestDistsFromMean:
                bestDistsFromMean = distsFromMean
                cluster_centers=means
        #cluster_centers=means
        dH=600
        dW=3000
        draw = np.zeros([dH,dW,3],dtype=np.float)
        toWrite = []
        final_k=k
        for ki in range(k):
            pop = (groups==ki).sum().item()
            if pop>2:
                color = np.random.uniform(0.2,1,3).tolist()
                #d=math.sqrt(mean[ki,11]**2 + mean[ki,12]**2)
                #theta = math.atan2(mean[ki,11],mean[ki,12]) + mean[ki,10]
                h=cluster_centers[ki,11]
                w=cluster_centers[ki,12]
                rot=cluster_centers[ki,10]
                toWrite.append({'height':h.item(),'width':w.item(),'rot':rot.item(),'popularity':pop})
                tr = ( int(math.cos(rot)*w-math.sin(rot)*h)+dW//2,   int(math.sin(rot)*w+math.cos(rot)*h)+dH//2 )
                tl = ( int(math.cos(rot)*-w-math.sin(rot)*h)+dW//2,  int(math.sin(rot)*-w+math.cos(rot)*h)+dH//2 )
                br = ( int(math.cos(rot)*w-math.sin(rot)*-h)+dW//2,  int(math.sin(rot)*w+math.cos(rot)*-h)+dH//2 )
                bl = ( int(math.cos(rot)*-w-math.sin(rot)*-h)+dW//2, int(math.sin(rot)*-w+math.cos(rot)*-h)+dH//2 )
                
                img_f.line(draw,tl,tr,color)
                img_f.line(draw,tr,br,color)
                img_f.line(draw,br,bl,color)
                img_f.line(draw,bl,tl,color,2)
            else:
                final_k-=1
        
        #print(toWrite)
        with open(outPath.format(final_k),'w') as out:
            out.write(json.dumps(toWrite))
            print('saved '+outPath.format(final_k))
        #img_f.imshow('clusters',draw)
        #img_f.waitKey()
        img_f.imwrite('clusters.png',draw*255)


def getWidthFromBB(bb):
    return (np.linalg.norm(bb[0]-bb[1]) + np.linalg.norm(bb[3]-bb[2]))/2
def getHeightFromBB(bb):
    return (np.linalg.norm(bb[0]-bb[3]) + np.linalg.norm(bb[1]-bb[2]))/2
