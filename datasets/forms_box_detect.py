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
from utils.forms_annotations import fixAnnotations, convertBBs, getBBWithPoints, getStartEndGT, getResponseBBIdList_
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


class FormsBoxDetect(BoxDetectDataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        super(FormsBoxDetect,self).__init__(dirPath,split,config,images)
        #if 'augmentation_params' in config['data_loader']:
        #    self.augmentation_params=config['augmentation_params']
        #else:
        #    self.augmentation_params=None
        if 'only_types' in config:
            self.only_types = config['only_types']
        else:
            self.only_types=None
        #print( self.only_types)
        if 'swap_circle' in config:
            self.swapCircle = config['swap_circle']
        else:
            self.swapCircle = False

        self.special_dataset = config['special_dataset'] if 'special_dataset' in config else None
        if 'simple_dataset' in config and config['simple_dataset']:
            self.special_dataset='simple'

        if images is not None:
            self.images=images
        else:
            if self.special_dataset is not None:
                splitFile = self.special_dataset+'_train_valid_test_split.json'
            else:
                splitFile = 'train_valid_test_split.json'
            with open(os.path.join(dirPath,splitFile)) as f:
                #if split=='valid' or split=='validation':
                #    trainTest='train'
                #else:
                #    trainTest=split
                readFile = json.loads(f.read())
                if type(split) is str:
                    groupsToUse = readFile[split]
                elif type(split) is list:
                    groupsToUse = {}
                    for spstr in split:
                        newGroups = readFile[spstr]
                        groupsToUse.update(newGroups)
                else:
                    print("Error, unknown split {}".format(split))
                    exit()
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
                                print('caching {} '.format(org_img),, end='\r')
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


    def parseAnn(self,np_img,annotations,s,imageName):
        fieldBBs = annotations['fieldBBs']
        fixAnnotations(self,annotations)

        full_bbs=annotations['byId'].values()

        bbs = getBBWithPoints(full_bbs,s,useBlankClass=(not self.no_blanks),usePairedClass=self.use_paired_class)
        numClasses = bbs.shape[2]-16
        #field_bbs = getBBWithPoints(annotations['fieldBBs'],s)
        #bbs = np.concatenate([text_bbs,field_bbs],axis=1) #has batch dim
        start_of_line, end_of_line = getStartEndGT(full_bbs,s)
        try:
            table_points, table_pixels = self.getTables(
                    fieldBBs,
                    s, 
                    np_img.shape[0], 
                    np_img.shape[1],
                    annotations['pairs'])
        except Exception as inst:
            table_points=None
            table_pixels=None
            if imageName not in self.errors:
                #print(inst)
                #print('Table error on: '+imageName)
                self.errors.append(imageName)

        ##print('getStartEndGt: '+str(timeit.default_timer()-tic))

        pixel_gt = table_pixels

        line_gts = {
                    "start_of_line": start_of_line,
                    "end_of_line": end_of_line
                    }
        point_gts = {
                        "table_points": table_points
                        }
        
        numNeighbors=defaultdict(lambda:0)
        for id,bb in annotations['byId'].items():
            if not self.onlyFormStuff or ('paired' in bb and bb['paired']):
                responseIds = getResponseBBIdList_(self,id,annotations)
                for id2,bb2 in annotations['byId'].items():
                    if id!=id2:
                        pair = id2 in responseIds
                        if pair:
                            numNeighbors[id]+=1
        numNeighbors = [numNeighbors[bb['id']] for bb in full_bbs]
        #if self.pred_neighbors:
        #    bbs = torch.cat(bbs,
        idToIndex={}
        for i,bb in enumerate(full_bbs):
            idToIndex[bb['id']]=i
        pairs=[ (idToIndex[id1],idToIndex[id2]) for id1,id2 in annotations['pairs'] ]
            



        return bbs,line_gts,point_gts,pixel_gt,numClasses,numNeighbors, pairs



            #def getBBGT(self,bbs,s):

            #    useBBs=bbs
            #    #for bb in bbs:
            #    #    if fields and self.isSkipField(bb):
            #    #        continue
            #    #    else:
            #    #        useBBs.append(bb)
            #    bbs = np.empty((1,len(useBBs), 8+8+2), dtype=np.float32) #2x4 corners, 2x4 cross-points, 2 classes
            #    j=0
            #    for bb in useBBs:
            #        tlX = bb['poly_points'][0][0]
            #        tlY = bb['poly_points'][0][1]
            #        trX = bb['poly_points'][1][0]
            #        trY = bb['poly_points'][1][1]
            #        brX = bb['poly_points'][2][0]
            #        brY = bb['poly_points'][2][1]
            #        blX = bb['poly_points'][3][0]
            #        blY = bb['poly_points'][3][1]

            #        bbs[:,j,0]=tlX*s
            #        bbs[:,j,1]=tlY*s
            #        bbs[:,j,2]=trX*s
            #        bbs[:,j,3]=trY*s
            #        bbs[:,j,4]=brX*s
            #        bbs[:,j,5]=brY*s
            #        bbs[:,j,6]=blX*s
            #        bbs[:,j,7]=blY*s
            #        #we add these for conveince to crop BBs within window
            #        bbs[:,j,8]=s*(tlX+blX)/2
            #        bbs[:,j,9]=s*(tlY+blY)/2
            #        bbs[:,j,10]=s*(trX+brX)/2
            #        bbs[:,j,11]=s*(trY+brY)/2
            #        bbs[:,j,12]=s*(tlX+trX)/2
            #        bbs[:,j,13]=s*(tlY+trY)/2
            #        bbs[:,j,14]=s*(brX+blX)/2
            #        bbs[:,j,15]=s*(brY+blY)/2
            #        bbs[:,j,16]=1 if not fields else 0
            #        bbs[:,j,17]=1 if fields else 0
            #        j+=1
            #    return bbs


    def getTables(self,bbs,s, sH, sW,selfPairs):
        #find which rows and cols are part of the same tables
        groups={}
        #groupsT={}
        groupId=0
        lookup=defaultdict(lambda: None)
        #print('num bbs {}'.format(len(bbs)))
        intersect=defaultdict(list)
        for bb in bbs:
            if bb['type'] == 'fieldCol': # and bb['type'] != 'fieldRow':
                myGroup = lookup[bb['id']]
            

                for bb2 in bbs:
                    if bb2['type'] == 'fieldRow':
                        if polyIntersect(bb['poly_points'], bb2['poly_points']):
                            intersect[bb['id']].append(bb2['id'])
                            
                            otherGroup = lookup[bb2['id']]
                            #if otherGroup is None:
                            #    print('intersection! {}:{},  {}:{}'.format(bb['id'],bb['poly_points'],bb2['id'],bb2['poly_points']))
                            if otherGroup is not None:
                                if myGroup is None:
                                    groups[otherGroup].append(bb)
                                    #groupsT[otherGroup].append(bb['id'])
                                    lookup[bb['id']]=otherGroup
                                    myGroup=otherGroup
                                elif myGroup!=otherGroup:
                                    #merge groups
                                    for ele in groups[otherGroup]:
                                        lookup[ele['id']]=myGroup
                                    groups[myGroup] +=  groups[otherGroup]
                                    #groupsT[myGroup] +=  groupsT[otherGroup]
                                    del groups[otherGroup]
                                    #del groupsT[otherGroup]
                            elif myGroup is None:
                                myGroup = str(groupId)
                                groupId+=1
                                groups[myGroup] = [bb,bb2]
                                #groupsT[myGroup] = [bb['id'],bb2['id']]
                                lookup[bb['id']] = myGroup
                                lookup[bb2['id']] = myGroup
                            else:
                                groups[myGroup].append(bb2)
                                #groupsT[myGroup].append(bb2['id'])
                                lookup[bb2['id']] = myGroup
                        #print(bb['id']+'  '+bb2['id'])
                        #print('{}  {}'.format(myGroup,otherGroup))
                        #print(groupsT)
                        #input("Press Enter to continue...")
        #now we check if a group needs split into two tables (horizontally)
        final_groups=[]
        for gid, group in groups.items():
            rowsOfCol=defaultdict(list)
            colOfX={}
            for bbCol in group:
                if bbCol['type'] == 'fieldCol':
                    colOfX[bbCol['poly_points'][0][0]] = bbCol['id']
                    for bbRow in group:
                        if bbRow['type'] == 'fieldRow' and bbRow['id'] in intersect[bbCol['id']]:
                            rowsOfCol[bbCol['id']].append(bbRow['id'])



            if len(rowsOfCol)>4:
                xs=list(colOfX.keys())
                xs.sort()

                numColAnchors = math.ceil(len(rowsOfCol)*0.25)
                leftAnchors=[colOfX[x] for x in xs[0:numColAnchors]]
                rightAnchors=[colOfX[x] for x in xs[-numColAnchors:]]
                leftRows = []
                for col in leftAnchors:
                    leftRows+=rowsOfCol[col]
                rightRows = []
                for col in rightAnchors:
                    rightRows+=rowsOfCol[col]
                leftRows=set(leftRows)
                rightRows=set(rightRows)

                def addPaired(s,pairs):
                    for pair in pairs:
                        if pair[0] in s:
                            s.add(pair[1])
                        elif pair[1] in s:
                            s.add(pair[0])
                addPaired(leftRows,selfPairs)
                addPaired(rightRows,selfPairs)
                
                intersected = leftRows.intersection(rightRows)
                if len(intersected)==0:
                    #split
                    loseRowCounts=defaultdict(lambda:0)
                    leftGroup=[]#list(leftRows)
                    rightGroup=[]#list(rightRows)
                    for bb in group:
                        if bb['type'] == 'fieldCol':
                            count=0
                            addLose=[]
                            for rowId in rowsOfCol[bb['id']]:
                                if rowId in leftRows:
                                    count-=1
                                elif rowId in rightRows:
                                    count+=1
                                else:
                                    addLose.append(rowId)
                            if count<0:
                                leftGroup.append(bb)
                                for id in addLose:
                                    loseRowCounts[id]-=1
                            elif count>0:
                                rightGroup.append(bb)
                                for id in addLose:
                                    loseRowCounts[id]+=1
                            else:
                                #error, this col is ambigous
                                raise Exception("ambig col",bb)
                        if bb['type'] == 'fieldRow':
                            if bb['id'] in leftRows:
                                leftGroup.append(bb)
                            elif bb['id'] in rightRows:
                                rightGroup.append(bb)
                            else:
                                loseRowCounts[bb['id']]+=0
                    for bb in group:
                        if bb['id'] in loseRowCounts:
                            if loseRowCounts[bb['id']]<0:
                                leftGroup.append(bb)
                            elif loseRowCounts[bb['id']]>0:
                                rightGroup.append(bb)
                            else:
                                raise Exception("ambig row",bb)
                    final_groups+=[leftGroup,rightGroup]
                else:
                    final_groups.append(group)
            else:
                final_groups.append(group)
                #elif len(intersect)<len(leftRows):
                #    #error, we have a row spannign left to right, but some dont

        #print(groups)
        #parse table bbs for intersection points
        intersectionPoints = []
        pixelMap = np.zeros((sH,sW,1),dtype=np.float32)
        #print('num tables {}'.format(len(groups)))
        for tableBBs in final_groups:
            #print('tableBBs {}'.format(len(tableBBs)))
            table = {}
            for bb in tableBBs:
                bb['poly_points'] = np.array( bb['poly_points'])
                table[bb['id']]=bb
            rows=[]
            cols=[]

            #I assume that rows and columns may be represented by multiple BBs
            combIds={}
            idToComb={}
            curComb=0
            #first we figure out what these are (they are paired)
            for pair in selfPairs:
                if pair[0] in table and pair[1] in table and table[pair[0]]['type']==table[pair[1]]['type']:
                    if pair[0] in idToComb:
                        pair0Comb=idToComb[pair[0]]
                        if pair[1] in idToComb:
                            pair1Comb=idToComb[pair[1]]
                            if pair0Comb!=pair1Comb:
                                #merge
                                #print('merge {}:{} and {}:{}'.format(pair0Comb,combIds[pair0Comb],pair1Comb,combIds[pair0Comb]))
                                combIds[pair0Comb] += combIds[pair1Comb]
                                for id in combIds[pair1Comb]:
                                    idToComb[id]=pair0Comb
                                del combIds[pair1Comb]
                        else:
                            combIds[pair0Comb].append(pair[1])
                            idToComb[pair[1]]=pair0Comb
                    elif pair[1] in idToComb:
                        pair1Comb=idToComb[pair[1]]
                        combIds[pair1Comb].append(pair[0])
                        idToComb[pair[0]]=pair1Comb
                    else:
                        combIds[curComb]=[pair[0],pair[1]]
                        idToComb[pair[0]]=curComb
                        idToComb[pair[1]]=curComb
                        curComb+=1
                    #print(combIds)
                    #print(idToComb)
            #sort them in order (left->right or top->botton) and add them to our lists
            for _,ids in combIds.items():
                    typ = table[ids[0]]['type']
                    toApp=[]
                    for id in ids:
                        toApp.append(table[id]['poly_points'])
                        cv2.fillConvexPoly(pixelMap[:,:,0],(table[id]['poly_points']*s).astype(int),1)
                        del table[id]

                    if typ=='fieldRow':
                        toApp.sort(key=lambda a: a[0,0])#sort horz by top-left point
                        rows.append(toApp)
                    else:
                        toApp.sort(key=lambda a: a[0,1])#sort vert by top-left point
                        cols.append(toApp)

            #add the single BB rows and columns
            for id,bb in table.items():
                #npBB = np.array(bb['poly_points'])
                if bb['type']=='fieldRow':
                    rows.append([bb['poly_points']])
                else:
                    cols.append([bb['poly_points']])
                #print(npBB*s)
                cv2.fillConvexPoly(pixelMap[:,:,0],(bb['poly_points']*s).astype(int),1)

            rows.sort(key=lambda a: a[0][0,1])#sort vertically by top-left point
            cols.sort(key=lambda a: a[0][0,0])#sort horizontally by top-left point
            #print (len(rows))
            #print (len(cols))
            #for each row seperator line (top and bottom lines of BBs) find intersecting column sep lines
            #we must iterate over all the components of each row
            #the very top boundary (and bottom) must be handeled specially since they don't have two lines
            nextInd=0
            j=0
            for lineComponent in rows[0]:
                ###rint('row start, comp:{}'.format(j))
                height = getHeightFromBB(lineComponent)
                if j==0:
                    distFromPrev=getWidthFromBB(lineComponent)/2#float('inf')
                else:
                    distFromPrev=np.linalg.norm(lineComponent[0]-rows[0][j-1][1])
                if j==len(rows[0])-1:
                    distToNext=getWidthFromBB(lineComponent)/2#float('inf')
                else:
                    distToNext=np.linalg.norm(lineComponent[1]-rows[0][j+1][0])
                somePoints,nextInd,before = getIntersectsCols(lineComponent[0:2],
                        cols,
                        nextInd,
                        threshLine_low=height/2,#float('inf'),
                        threshLine_high=height/2,
                        threshLeft=distFromPrev,
                        threshRight=distToNext)
                if before:
                    intersectionPoints[-1] = (intersectionPoints[-1]+somePoints[0]*s)/2
                    for p in somePoints[1:]:
                        intersectionPoints.append(s*p)
                else:
                    for p in somePoints:
                        intersectionPoints.append(s*p)
                j+=1
            for i in range(len(rows)-1):
                avgHHeight_ip1=0
                for lineComponent in rows[i+1]:
                    height = getHeightFromBB(lineComponent)
                    avgHHeight_ip1+=height/2
                avgHHeight_ip1/=len(rows[i+1])
                nextInd=0
                pointsU=[] #points from the bottom line of the BB above the seperator
                avgHHeight_i=0
                j=0
                #get the upper points (bottom of above BB)
                for lineComponent in rows[i]:
                    ###rint('row U {}, comp:{}'.format(i,j))
                    height = getHeightFromBB(lineComponent)
                    avgHHeight_i+=height/2
                    if j==0:
                        distFromPrev=getWidthFromBB(lineComponent)/2#float('inf')
                    else:
                        distFromPrev=np.linalg.norm(lineComponent[3]-rows[i][j-1][2])
                    if j==len(rows[i])-1:
                        distToNext=getWidthFromBB(lineComponent)/2#float('inf')
                    else:
                        distToNext=np.linalg.norm(lineComponent[2]-rows[i][j+1][3])
                    somePoints,nextInd,before = getIntersectsCols(lineComponent[2:4],
                            cols,
                            nextInd,
                            threshLine_low=height/2,
                            threshLine_high=avgHHeight_ip1,
                            threshLeft=distFromPrev,
                            threshRight=distToNext)
                    if before:
                        if len(pointsU)>0:
                            pointsU = pointsU[:-1] + [(pointsU[-1]+somePoints[0])/2] + somePoints[1:]
                        else:
                            pointsU = somePoints[1:]
                    else:
                        pointsU+=somePoints
                    j+=1
                avgHHeight_i/=len(rows[i])
                pointsL=[] #points from the top line of the BB below the seperator
                nextInd=0
                j=0
                #get lower points (top of below BB)
                for lineComponent in rows[i+1]:
                    ###rint('row L {}, comp:{}'.format(i,j))
                    height = getHeightFromBB(lineComponent)
                    if j==0:
                        distFromPrev=getWidthFromBB(lineComponent)/2#float('inf')
                    else:
                        distFromPrev=np.linalg.norm(lineComponent[0]-rows[i+1][j-1][1])
                    if j==len(rows[i+1])-1:
                        distToNext=getWidthFromBB(lineComponent)/2#float('inf')
                    else:
                        distToNext=np.linalg.norm(lineComponent[1]-rows[i+1][j+1][0])
                    somePoints,nextInd,before = getIntersectsCols(lineComponent[0:2],
                            cols,
                            nextInd,
                            threshLine_low=avgHHeight_i,
                            threshLine_high=height/2,
                            threshLeft=distFromPrev,
                            threshRight=distToNext)
                    if before:
                        if len(pointsL)>0:
                            pointsL = pointsL[:-1] + [(pointsL[-1]+somePoints[0])/2] + somePoints[1:]
                        else:
                            pointsL = somePoints[1:]
                    else:
                        pointsL+=somePoints
                    j+=1
                #print(i)
                #print(pointsU)
                #print(pointsL)
                if len(pointsU) != len(pointsL):
                    raise Exception(i,pointsU,pointsL)
                #average the upper and lower points (and scale them)
                for pi in range(len(pointsU)):
                    intersectionPoints.append(s*(pointsU[pi]+pointsL[pi])/2.0)
                    
            #special handeling of bottom boundary
            nextInd=0
            j=0
            for lineComponent in rows[-1]:
                ###rint('row end, comp:{}'.format(j))
                if j==0:
                    distFromPrev=getWidthFromBB(lineComponent)/2#float('inf')
                else:
                    distFromPrev=np.linalg.norm(lineComponent[3]-rows[-1][j-1][2])
                if j==len(rows[-1])-1:
                    distToNext=getWidthFromBB(lineComponent)/2#float('inf')
                else:
                    distToNext=np.linalg.norm(lineComponent[2]-rows[-1][j+1][3])
                somePoints,nextInd,before = getIntersectsCols(lineComponent[2:4],
                        cols,
                        nextInd,
                        threshLine_low=height/2,
                        threshLine_high=height/2,#float('inf'),
                        threshLeft=distFromPrev,
                        threshRight=distToNext)
                j+=1
                if before:
                    intersectionPoints[-1] = (intersectionPoints[-1]+somePoints[0]*s)/2
                    for p in somePoints[1:]:
                        intersectionPoints.append(s*p)
                else:
                    for p in somePoints:
                        intersectionPoints.append(s*p)

                    #rowLines=[ [rows[0][0], rows[0][1]] ]
                    #for i in range(len(rows)-1):
                    #    rowLines.append( [(rows[i][3]+rows[i+1][0])/2, (rows[i][2]+rows[i+1][1])/2] )
                    #rowLines.append( [rows[-1][3], rows[-1][2]] )

                    #colLines=[ [cols[0][0], cols[0][3]] ]
                    #for i in range(len(cols)-1):
                    #    colLines.append( [(cols[i][1]+cols[i+1][0])/2, (cols[i][2]+cols[i+1][3])/2] )
                    #colLines.append( [cols[-1][1], cols[-1][2]] )

                    #for rowLine in rowLines:
                    #    for colLine in colLines:
                    #        p = lineIntersection(rowLine,colLine)
                    #        if p is not None:
                    #            #print('{} {} = {}'.format(rowLine,colLine,p))
                    #            intersectionPoints.append((p[0]*s,p[1]*s))

        intersectionPointsM = np.empty((1,len(intersectionPoints), 2), dtype=np.float32)
        j=0
        for x, y in intersectionPoints:
            intersectionPointsM[0,j,0]=x
            intersectionPointsM[0,j,1]=y
            j+=1
            
        return intersectionPointsM, pixelMap

    def cluster(self,k,sample_count,outPath):
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
        meanH=62.42
        stdH=87.31
        meanW=393.03
        stdW=533.53
        ratios=[4.0,7.18,11.0,15.0,19.0,27.0]
        pointsAndRects=[]
        for inst in self.images:
            annotationPath = inst['annotationPath']
            #rescaled = inst['rescaled']
            with open(annotationPath) as annFile:
                annotations = json.loads(annFile.read())
            fixAnnotations(self,annotations)
            for i in range(sample_count):
                if i==0:
                    s = (self.rescale_range[0]+self.rescale_range[1])/2
                else:
                    s = np.random.uniform(self.rescale_range[0], self.rescale_range[1])
                #partial_rescale = s/rescaled
                bbs = getBBWithPoints(annotations['byId'].values(),s)
                #field_bbs = self.getBBGT(annotations['fieldBBs'],s,fields=True)
                #bbs = np.concatenate([text_bbs,field_bbs],axis=1)
                bbs = convertBBs(bbs,self.rotate,2).numpy()[0]
                cos_rot = np.cos(bbs[:,2])
                sin_rot = np.sin(bbs[:,2])
                p_left_x = -cos_rot*bbs[:,4]
                p_left_y = -sin_rot*bbs[:,4]
                p_right_x = cos_rot*bbs[:,4]
                p_right_y = sin_rot*bbs[:,4]
                p_top_x = sin_rot*bbs[:,3]
                p_top_y = -cos_rot*bbs[:,3]
                p_bot_x = -sin_rot*bbs[:,3]
                p_bot_y = cos_rot*bbs[:,3]
                points = np.stack([p_left_x,p_left_y,p_right_x,p_right_y,p_top_x,p_top_y,p_bot_x,p_bot_y],axis=1)
                pointsAndRects.append(np.concatenate([points,bbs[:,:5]],axis=1))
        pointsAndRects = np.concatenate(pointsAndRects,axis=0)
        #all_points = pointsAndRects[:,0:8]
        #all_heights = pointsAndRects[:,11]
        #all_widths = pointsAndRects[:,12]
        
        bestDistsFromMean=None
        for attempt in range(20 if k>0 else 1):
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
                print('attempt:{}, bestDistsFromMean:{}, iteration:{}, bestDistsFromMean:{}'.format(attempt,bestDistsFromMean,iteration,prevDistsFromMean), end='\r')
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
                
                cv2.line(draw,tl,tr,color)
                cv2.line(draw,tr,br,color)
                cv2.line(draw,br,bl,color)
                cv2.line(draw,bl,tl,color,2)
            else:
                final_k-=1
        
        #print(toWrite)
        with open(outPath.format(final_k),'w') as out:
            out.write(json.dumps(toWrite))
            print('saved '+outPath.format(final_k))
        cv2.imshow('clusters',draw)
        cv2.waitKey()


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

