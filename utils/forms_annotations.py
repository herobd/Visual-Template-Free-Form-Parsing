from collections import defaultdict
import numpy as np
import torch

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
    
    tlX = bbs[:,:,0]
    tlY = bbs[:,:,1]
    trX = bbs[:,:,2]
    trY = bbs[:,:,3]
    brX = bbs[:,:,4]
    brY = bbs[:,:,5]
    blX = bbs[:,:,6]
    blY = bbs[:,:,7]

    if not rotate:
        tlX = np.minimum.reduce((tlX,blX,trX,brX))
        tlY = np.minimum.reduce((tlY,trY,blY,brY))
        trX = np.maximum.reduce((tlX,blX,trX,brX))
        trY = np.minimum.reduce((tlY,trY,blY,brY))
        brX = np.maximum.reduce((tlX,blX,trX,brX))
        brY = np.maximum.reduce((tlY,trY,blY,brY))
        blX = np.minimum.reduce((tlX,blX,trX,brX))
        blY = np.maximum.reduce((tlY,trY,blY,brY))

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
    rot = np.arctan2((rY-lY),rX-lX)
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


    return torch.from_numpy(new_bbs)

def fixAnnotations(this,annotations):
    def isSkipField(this,bb):
        return (    (this.no_blanks and (bb['isBlank']=='blank' or bb['isBlank']==3)) or
                    (this.no_print_fields and (bb['isBlank']=='print' or bb['isBlank']==2)) or
                    #TODO no graphics
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
    circleIds=[]
    for bb in annotations['fieldBBs']:
        id=bb['id']
        if isSkipField(this,bb):
            #print('remove {}'.format(id))
            idsToRemove.add(id)
            if bb['type']=='fieldP':
                idsToFix.append(id)
        elif bb['type']=='fieldCircle':
            circleIds.append(id)
            if this.swapCircle:
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

    #handle groups of things that are intended to be circled or crossed out
    #first identify groups
    circleGroups={}
    circleGroupId=0
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
                circleGroups[circleGroupId] = pair
                circleGroupId+=1

    #what pairs to each group?
    groupPairedTo=defaultdict(list)
    for pair in annotations['pairs']:
        if pair[0] in circleIds and pair[1] not in circleIds:
            for id,group in circleGroups.items():
                if pair[0] in group:
                    groupPairedTop[id].append(pair[1])

        if pair[1] in circleIds and pair[0] not in circleIds:
            for id,group in circleGroups.items():
                if pair[1] in group:
                    groupPairedTo[id].append(pair[0])


    #add pairs
    toAdd=[]
    for gid,group in  circleGroups.items():
        for id in group:
            for id2 in group:
                if id!=id2:
                    toAdd.append([id,id2])
            for id2 in groupPairedTo[gid]:
                toAdd.append([id,id2])
    for pair in toAdd:
        if pair not in annotations['pairs'] and [pair[1],pair[0]] not in annotations['pairs']:
             annotations['pairs'].append(pair)
