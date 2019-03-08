from collections import defaultdict
import numpy as np
import torch
import math

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


def convertBBs(bbs,rotate,numClasses):
    if bbs.shape[1]==0:
        return None
    new_bbs = np.empty((1,bbs.shape[1], 5+8+numClasses), dtype=np.float32) #5 params, 8 points (used in loss), n classes
    
    tlX_ = bbs[:,:,0]
    tlY_ = bbs[:,:,1]
    trX_ = bbs[:,:,2]
    trY_ = bbs[:,:,3]
    brX_ = bbs[:,:,4]
    brY_ = bbs[:,:,5]
    blX_ = bbs[:,:,6]
    blY_ = bbs[:,:,7]

    if not rotate:
        tlX = np.minimum.reduce((tlX_,blX_,trX_,brX_))
        tlY = np.minimum.reduce((tlY_,trY_,blY_,brY_))
        trX = np.maximum.reduce((tlX_,blX_,trX_,brX_))
        trY = np.minimum.reduce((tlY_,trY_,blY_,brY_))
        brX = np.maximum.reduce((tlX_,blX_,trX_,brX_))
        brY = np.maximum.reduce((tlY_,trY_,blY_,brY_))
        blX = np.minimum.reduce((tlX_,blX_,trX_,brX_))
        blY = np.maximum.reduce((tlY_,trY_,blY_,brY_))
    else:
        tlX =  tlX_
        tlY =  tlY_
        trX =  trX_
        trY =  trY_
        brX =  brX_
        brY =  brY_
        blX =  blX_
        blY =  blY_

    lX = (tlX+blX)/2.0
    lY = (tlY+blY)/2.0
    rX = (trX+brX)/2.0
    rY = (trY+brY)/2.0
    d=np.sqrt((lX-rX)**2 + (lY-rY)**2)

    hl = ((tlX-lX)*-(rY-lY) + (tlY-lY)*(rX-lX))/d #projection of half-left edge onto transpose horz run
    hr = ((brX-rX)*-(lY-rY) + (brY-rY)*(lX-rX))/d #projection of half-right edge onto transpose horz run
    h = (hl+hr)/2.0

    #tX = lX + h*-(rY-lY)/d
    #tY = lY + h*(rX-lX)/d
    #bX = lX - h*-(rY-lY)/d
    #bY = lY - h*(rX-lX)/d

    #etX =tX + rX-lX
    #etY =tY + rY-lY
    #ebX =bX + rX-lX
    #ebY =bY + rY-lY

    cX = (lX+rX)/2.0
    cY = (lY+rY)/2.0
    rot = np.arctan2(-(rY-lY),rX-lX)
    height = np.abs(h)    #this is half height
    width = d/2.0 #and half width

    topX = (tlX+trX)/2.0
    topY = (tlY+trY)/2.0
    botX = (blX+brX)/2.0
    botY = (blY+brY)/2.0
    leftX = lX
    leftY = lY
    rightX = rX
    rightY = rY

    height[ np.logical_or(np.isnan(height),height==0) ] =1
    width[ np.logical_or(np.isnan(width),width==0) ] =1

    new_bbs[:,:,0]=cX
    new_bbs[:,:,1]=cY
    new_bbs[:,:,2]=rot
    new_bbs[:,:,3]=height
    new_bbs[:,:,4]=width
    new_bbs[:,:,5]=leftX
    new_bbs[:,:,6]=leftY
    new_bbs[:,:,7]=rightX
    new_bbs[:,:,8]=rightY
    new_bbs[:,:,9]=topX
    new_bbs[:,:,10]=topY
    new_bbs[:,:,11]=botX
    new_bbs[:,:,12]=botY
    #print("{} {}, {} {}".format(new_bbs.shape,new_bbs[:,:,13:].shape,bbs.shape,bbs[:,:,-numClasses].shape))
    new_bbs[:,:,13:]=bbs[:,:,-numClasses:]

    assert(not np.isnan(new_bbs).any())


    return torch.from_numpy(new_bbs)


#This annotation corrects assumptions made during GTing, modifies the annotations for the current parameterization, and slightly changes the format
def fixAnnotations(this,annotations):
    def isSkipField(this,bb):
        return (    (this.no_blanks and (bb['isBlank']=='blank' or bb['isBlank']==3)) or
                    (this.no_print_fields and (bb['isBlank']=='print' or bb['isBlank']==2)) or
                    (this.no_graphics and bb['type']=='graphic') or
                    bb['type'] == 'fieldRow' or
                    bb['type'] == 'fieldCol' or
                    bb['type'] == 'fieldRegion'
                )



    #restructure
    annotations['byId']={}
    for bb in annotations['textBBs']:
        annotations['byId'][bb['id']]=bb
    for bb in annotations['fieldBBs']:
        annotations['byId'][bb['id']]=bb
    if 'samePairs' in annotations:
        if not this.only_opposite_pairs:
            annotations['pairs']+=annotations['samePairs']
        del annotations['samePairs']

    numPairsWithoutBB=0
    for id1,id2 in annotations['pairs']:
        if id1 not in annotations['byId'] or id2 not in annotations['byId']:
            numPairsWithoutBB+=1

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
    circleIds=[]
    for bb in annotations['fieldBBs']:
        id=bb['id']
        #print('skip:{}, type:{}'.format(isSkipField(this,bb),bb['type']))
        if isSkipField(this,bb):
            #print('remove {}'.format(id))
            idsToRemove.add(id)
            if bb['type']=='fieldP':
                idsToFix.append(id)
        elif bb['type']=='fieldCircle':
            circleIds.append(id)
            if this.swapCircle:
                annotations['byId'][id]['type']='textCircle'

    del annotations['fieldBBs']
    del annotations['textBBs']

    
    parasLinkedTo=defaultdict(list)
    pairsToRemove=[]
    for i,pair in enumerate(annotations['pairs']):
        assert(len(pair)==2)
        if pair[0] not in annotations['byId'] or pair[1] not in annotations['byId']:
            pairsToRemove.append(i)
        elif pair[0] in idsToFix and annotations['byId'][pair[1]]['type'][-1]=='P':
            parasLinkedTo[pair[0]].append(pair[1])
            pairsToRemove.append(i)
        elif pair[1] in idsToFix and annotations['byId'][pair[0]]['type'][-1]=='P':
            parasLinkedTo[pair[1]].append(pair[0])
            pairsToRemove.append(i)
        elif pair[0] in idsToRemove or pair[1] in idsToRemove:
            pairsToRemove.append(i)
        elif (this.only_opposite_pairs and 
                ( (annotations['byId'][pair[0]]['type'][:4]=='text' and 
                   annotations['byId'][pair[1]]['type'][:4]=='text') or
                  (annotations['byId'][pair[0]]['type'][:4]=='field' and 
                    annotations['byId'][pair[1]]['type'][:4]=='field') )):
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

    for pair in annotations['pairs']:
        assert(len(pair)==2)
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
        assert(len(pair)==2)
        if pair not in annotations['pairs'] and [pair[1],pair[0]] not in annotations['pairs']:
             annotations['pairs'].append(pair)
    #annotations['pairs']+=toAdd

    #handle groups of things that are intended to be circled or crossed out
    #first identify groups
    circleGroups={}
    circleGroupId=0
    #also find text-field pairings
    paired = set()
    for pair in annotations['pairs']:
        if pair[0] in circleIds and pair[1] in circleIds:
            group0=None
            group1=None
            for id,group in circleGroups.items():
                if pair[0] in group:
                    group0=id
                if pair[1] in group:
                    group1=id
            if group0 is not None:
                if group1 is None:
                    circleGroups[group0].append(pair[1])
                elif group0!=group1:
                    circleGroups[group0] += circleGroups[group1]
                    del circleGroups[group1]
            elif group1 is not None:
                circleGroups[group1].append(pair[0])
            else:
                circleGroups[circleGroupId] = pair.copy()
                circleGroupId+=1

        if pair[0] in annotations['byId'] and pair[1] in annotations['byId']:
            cls0 = annotations['byId'][pair[0]]['type'][:4]=='text'
            cls1 = annotations['byId'][pair[1]]['type'][:4]=='text'
            if cls0!=cls1:
                paired.add(pair[0])
                paired.add(pair[1])

    for pair in annotations['pairs']:
        assert(len(pair)==2)

    #what pairs to each group?
    groupPairedTo=defaultdict(list)
    for pair in annotations['pairs']:
        if pair[0] in circleIds and pair[1] not in circleIds:
            for id,group in circleGroups.items():
                if pair[0] in group:
                    groupPairedTo[id].append(pair[1])

        if pair[1] in circleIds and pair[0] not in circleIds:
            for id,group in circleGroups.items():
                if pair[1] in group:
                    groupPairedTo[id].append(pair[0])


    for pair in annotations['pairs']:
        assert(len(pair)==2)
    #add pairs
    toAdd=[]
    if not this.only_opposite_pairs:
        for gid,group in  circleGroups.items():
            for id in group:
                for id2 in group:
                    if id!=id2:
                        toAdd.append([id,id2])
                for id2 in groupPairedTo[gid]:
                    toAdd.append([id,id2])
    for pair in toAdd:
        assert(len(pair)==2)
        if pair not in annotations['pairs'] and [pair[1],pair[0]] not in annotations['pairs']:
             annotations['pairs'].append(pair)

    #mark each bb that is chained to a cross-class pairing
    while True:
        size = len(paired)
        for pair in annotations['pairs']:
            if pair[0] in paired:
                paired.add(pair[1])
            elif pair[1] in paired:
                paired.add(pair[0])
        if len(paired)<=size:
            break #at the end of every chain
    for id in paired:
        if id in annotations['byId']:
            annotations['byId'][id]['paired']=True

    for pair in annotations['pairs']:
        assert(len(pair)==2)

    return numPairsWithoutBB

def getBBWithPoints(useBBs,s,useBlankClass=False,usePairedClass=False):

    numClasses=2
    if useBlankClass:
        numClasses+=1
    if usePairedClass:
        numClasses+=1
    bbs = np.empty((1,len(useBBs), 8+8+numClasses), dtype=np.float32) #2x4 corners, 2x4 cross-points, 2 classes
    for j,bb in enumerate(useBBs):
        tlX = bb['poly_points'][0][0]
        tlY = bb['poly_points'][0][1]
        trX = bb['poly_points'][1][0]
        trY = bb['poly_points'][1][1]
        brX = bb['poly_points'][2][0]
        brY = bb['poly_points'][2][1]
        blX = bb['poly_points'][3][0]
        blY = bb['poly_points'][3][1]

            

        bbs[:,j,0]=tlX*s
        bbs[:,j,1]=tlY*s
        bbs[:,j,2]=trX*s
        bbs[:,j,3]=trY*s
        bbs[:,j,4]=brX*s
        bbs[:,j,5]=brY*s
        bbs[:,j,6]=blX*s
        bbs[:,j,7]=blY*s
        #we add these for conveince to crop BBs within window
        bbs[:,j,8]=s*(tlX+blX)/2.0
        bbs[:,j,9]=s*(tlY+blY)/2.0
        bbs[:,j,10]=s*(trX+brX)/2.0
        bbs[:,j,11]=s*(trY+brY)/2.0
        bbs[:,j,12]=s*(tlX+trX)/2.0
        bbs[:,j,13]=s*(tlY+trY)/2.0
        bbs[:,j,14]=s*(brX+blX)/2.0
        bbs[:,j,15]=s*(brY+blY)/2.0

        #classes
        if bb['type']=='detectorPrediction':
            bbs[:,j,16]=bb['textPred']
            bbs[:,j,17]=bb['fieldPred']
        else:
            field = bb['type'][:4]!='text'
            text=not field
            bbs[:,j,16]=1 if text else 0
            bbs[:,j,17]=1 if field else 0
        index = 18
        if useBlankClass:
            if bb['type']=='detectorPrediction':
                bbs[:,j,index]=bb['blankPred']
            else:
                blank = (bb['isBlank']=='blank' or bb['isBlank']==3) if 'isBlank' in bb else False
                bbs[:,j,index]=1 if blank else 0
            index+=1
        if usePairedClass:
            assert(bb['type']!='detectorPrediction')
            paired = bb['paired'] if 'paired' in bb else False
            bbs[:,j,index]=1 if paired else 0
            index+=1
    return bbs

def getStartEndGT(useBBs,s,useBlankClass=False):


    if useBlankClass:
        numClasses=3
    else:
        numClasses=2
    start_gt = np.empty((1,len(useBBs), 4+numClasses), dtype=np.float32) #x,y,r,h, x classes
    end_gt = np.empty((1,len(useBBs), 4+numClasses), dtype=np.float32) #x,y,r,h, x classes
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
        if useBlankClass and (bb['isBlank']=='blank' or bb['isBlank']==3):
            field=False
            text=False
            blank=True
        else:
            text=not field
            blank=False
            
        lX = (tlX+blX)/2.0
        lY = (tlY+blY)/2.0
        rX = (trX+brX)/2.0
        rY = (trY+brY)/2.0
        d=math.sqrt((lX-rX)**2 + (lY-rY)**2)

        hl = ((tlX-lX)*-(rY-lY) + (tlY-lY)*(rX-lX))/d #projection of half-left edge onto transpose horz run
        hr = ((brX-rX)*-(lY-rY) + (brY-rY)*(lX-rX))/d #projection of half-right edge onto transpose horz run
        h = (hl+hr)/2.0

        tX = lX + h*-(rY-lY)/d
        tY = lY + h*(rX-lX)/d
        bX = lX - h*-(rY-lY)/d
        bY = lY - h*(rX-lX)/d
        start_gt[:,j,0] = tX*s
        start_gt[:,j,1] = tY*s
        start_gt[:,j,2] = bX*s
        start_gt[:,j,3] = bY*s

        etX =tX + rX-lX
        etY =tY + rY-lY
        ebX =bX + rX-lX
        ebY =bY + rY-lY
        end_gt[:,j,0] = etX*s
        end_gt[:,j,1] = etY*s
        end_gt[:,j,2] = ebX*s
        end_gt[:,j,3] = ebY*s

        #classes
        start_gt[:,j,4]=1 if text else 0
        start_gt[:,j,5]=1 if field else 0
        if useBlankClass:
            start_gt[:,j,6]=1 if blank else 0
        end_gt[:,j,4]=1 if text else 0
        end_gt[:,j,5]=1 if field else 0
        if useBlankClass:
            end_gt[:,j,6]=1 if blank else 0
        j+=1
    return start_gt, end_gt

def getBBInfo(bb,rotate,useBlankClass=False):

    tlX_ = bb['poly_points'][0][0]
    tlY_ = bb['poly_points'][0][1]
    trX_ = bb['poly_points'][1][0]
    trY_ = bb['poly_points'][1][1]
    brX_ = bb['poly_points'][2][0]
    brY_ = bb['poly_points'][2][1]
    blX_ = bb['poly_points'][3][0]
    blY_ = bb['poly_points'][3][1]

    if not rotate:
        tlX = np.minimum.reduce((tlX_,blX_,trX_,brX_))
        tlY = np.minimum.reduce((tlY_,trY_,blY_,brY_))
        trX = np.maximum.reduce((tlX_,blX_,trX_,brX_))
        trY = np.minimum.reduce((tlY_,trY_,blY_,brY_))
        brX = np.maximum.reduce((tlX_,blX_,trX_,brX_))
        brY = np.maximum.reduce((tlY_,trY_,blY_,brY_))
        blX = np.minimum.reduce((tlX_,blX_,trX_,brX_))
        blY = np.maximum.reduce((tlY_,trY_,blY_,brY_))
    else:
        tlX =  tlX_
        tlY =  tlY_
        trX =  trX_
        trY =  trY_
        brX =  brX_
        brY =  brY_
        blX =  blX_
        blY =  blY_


    if bb['type']=='detectorPrediction':
        text=bb['textPred']
        field=bb['fieldPred']
        blank = bb['blankPred'] if 'blankPred' in bb else None
        nn = bb['nnPred'] if 'nnPred' in bb else None
    else:
        field = bb['type'][:4]!='text'
        if useBlankClass:
            blank = bb['isBlank']=='blank' or bb['isBlank']==3
        else:
            blank=None
        text=not field
        nn=None
        
    lX = (tlX+blX)/2.0
    lY = (tlY+blY)/2.0
    rX = (trX+brX)/2.0
    rY = (trY+brY)/2.0
    d=math.sqrt((lX-rX)**2 + (lY-rY)**2)

    #orthX = -(rY-lY)
    #orthY = (rX-lX)
    #origLX = blX-tlX
    #origLY = blY-tlY
    #origRX = brX-trX
    #origRY = brY-trY
    #hl = (orthX*origLX + orthY*origLY)/d
    #hr = (orthX*origRX + orthY*origRY)/d
    hl = ((tlX-lX)*-(rY-lY) + (tlY-lY)*(rX-lX))/d #projection of half-left edge onto transpose horz run
    hr = ((brX-rX)*-(lY-rY) + (brY-rY)*(lX-rX))/d #projection of half-right edge onto transpose horz run
    h = (np.abs(hl)+np.abs(hr))/2.0
    #h=0

    cX = (lX+rX)/2.0
    cY = (lY+rY)/2.0
    rot = np.arctan2(-(rY-lY),rX-lX)
    height = h*2 #use full height
    width = d

    return cX,cY,height,width,rot,text,field,blank,nn


def getResponseBBIdList_(this,queryId,annotations):
    responseBBList=[]
    for pair in annotations['pairs']: #done already +annotations['samePairs']:
        if queryId in pair:
            if pair[0]==queryId:
                otherId=pair[1]
            else:
                otherId=pair[0]
            if otherId in annotations['byId'] and (not this.onlyFormStuff or ('paired' in bb and bb['paired'])):
                #responseBBList.append(annotations['byId'][otherId])
                responseBBList.append(otherId)
    return responseBBList
