import numpy as np
from collections import defaultdict


def combineLine(classMap,line,bbs,trans,lineTrans,s,label):
    numClasses=len(classMap)
    bb = np.empty(8+8+numClasses, dtype=np.float32)
    lXL = min([w[0] for w in line])
    rXL = max([w[2] for w in line])
    tYL = min([w[1] for w in line])
    bYL = max([w[3] for w in line])
    bb[0]=lXL*s
    bb[1]=tYL*s
    bb[2]=rXL*s
    bb[3]=tYL*s
    bb[4]=rXL*s
    bb[5]=bYL*s
    bb[6]=lXL*s
    bb[7]=bYL*s
    #we add these for conveince to crop BBs within window
    bb[8]=s*lXL
    bb[9]=s*(tYL+bYL)/2.0
    bb[10]=s*rXL
    bb[11]=s*(tYL+bYL)/2.0
    bb[12]=s*(lXL+rXL)/2.0
    bb[13]=s*tYL
    bb[14]=s*(rXL+lXL)/2.0
    bb[15]=s*bYL
    
    bb[16:]=0
    bb[classMap[label]]=1
    #if boxinfo['label']=='header':
    #    bb[16]=1
    #elif boxinfo['label']=='question':
    #    bb[17]=1
    #elif boxinfo['label']=='answer':
    #    bb[18]=1
    #elif boxinfo['label']=='other':
    #    bb[19]=1
    bbs.append(bb)
    trans.append(' '.join(lineTrans))
    #nex = j<len(boxes)-1
    #numNeighbors.append(len(boxinfo['linking'])+(1 if prev else 0)+(1 if nex else 0))
    #prev=True



def createLines(annotations,classMap,scale):
    numClasses=len(classMap)
    boxes = annotations['form']
    origIdToIndexes={}
    annotations['linking']=defaultdict(list)
    groups=[]
    bbs=[]
    trans=[]
    line=[]
    lineTrans=[]

    numBBs = len(boxes)
    #new line
    line=[]
    lineTrans=[]
    for j,boxinfo in enumerate(boxes):
        prev=False
        line=[]
        lineTrans=[]
        startIdx=len(bbs)
        for word in boxinfo['words']:
            lX,tY,rX,bY = word['box']
            if len(line)==0:
                line.append(word['box']+[(lX+rX)/2,(tY+bY)/2])
                lineTrans.append(word['text'])
            else:
                difX = lX-line[-1][2]
                difY = (tY+bY)/2 - line[-1][5]
                pW = line[-1][2]-line[-1][0]
                pH = line[-1][3]-line[-1][1]
                if difX<-pW*0.25 or difY>pH*0.75:
                    combineLine(classMap,line,bbs,trans,lineTrans,scale,boxinfo['label'])
                    line=[]
                    lineTrans=[]
                line.append(word['box']+[(lX+rX)/2,(tY+bY)/2])
                lineTrans.append(word['text'])
        combineLine(classMap,line,bbs,trans,lineTrans,scale,boxinfo['label'])
        endIdx=len(bbs)
        groups.append(list(range(startIdx,endIdx)))
        for idx in range(startIdx,endIdx-1):
            annotations['linking'][idx].append(idx+1) #we link them in read order. The group supervises dense connections. Read order is how the NAF dataset is labeled.
        origIdToIndexes[j]=(startIdx,endIdx-1)

    for j,boxinfo in enumerate(boxes):
        for linkId in boxinfo['linking']:
            linkId = linkId[0] if linkId[1]==j else linkId[1]
            j_first_x = np.mean(bbs[origIdToIndexes[j][0]][0:8:2])
            j_first_y = np.mean(bbs[origIdToIndexes[j][0]][1:8:2])
            link_first_x = np.mean(bbs[origIdToIndexes[linkId][0]][0:8:2])
            link_first_y = np.mean(bbs[origIdToIndexes[linkId][0]][1:8:2])
            j_last_x = np.mean(bbs[origIdToIndexes[j][1]][0:8:2])
            j_last_y = np.mean(bbs[origIdToIndexes[j][1]][1:8:2])
            link_last_x = np.mean(bbs[origIdToIndexes[linkId][1]][0:8:2])
            link_last_y = np.mean(bbs[origIdToIndexes[linkId][1]][1:8:2])

            above = link_last_y<=j_first_y+2
            below = link_first_y>=j_last_y-2
            left = link_last_x<=j_first_x+2
            right = link_first_x>=j_last_x-2
            if above or left:
                annotations['linking'][origIdToIndexes[j][0]].append(origIdToIndexes[linkId][1])
            elif below or right:
                annotations['linking'][origIdToIndexes[j][1]].append(origIdToIndexes[linkId][0])
            else:
                print("!!!!!!!!")
                print("Print odd para align, unhandeled case.")
                print("trans:{}, ({},{}), trans:{}, ({},{})   , trans:{}, ({},{}), trans:{}, ({},{})".format(trans[origIdToIndexes[j][0]],j_first_x,j_first_y,trans[origIdToIndexes[j][1]],j_last_x,j_last_y,trans[origIdToIndexes[linkId][0]],link_first_x,link_first_y,trans[origIdToIndexes[linkId][1]],link_last_x,link_last_y))
                import pdb;pdb.set_trace()
                #annotations['linking'][origIdToIndexes[j][1]].append(origIdToIndexes[linkId][0])
    numNeighbors = [len(annotations['linking'][index]) for index in range(len(bbs))]
    bbs = np.stack(bbs,axis=0)
    bbs = bbs[None,...] #add batch dim

    return bbs, numNeighbors, trans, groups
