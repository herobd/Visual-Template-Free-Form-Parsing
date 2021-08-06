import numpy as np
import cv2
import math, os, random
import torch

def getCorners(xyrhw):
    xc=xyrhw[0].item()
    yc=xyrhw[1].item()
    r=xyrhw[2].item()
    h=xyrhw[3].item()
    w=xyrhw[4].item()
    h = min(30000,h)
    w = min(30000,w)

    tlX = int(-w*math.cos(r) -h*math.sin(r) +xc)
    tlY = int(-h*math.cos(r) +w*math.sin(r) +yc)
    trX = int( w*math.cos(r) -h*math.sin(r) +xc)
    trY = int(-h*math.cos(r) -w*math.sin(r) +yc)
    brX = int( w*math.cos(r) +h*math.sin(r) +xc)
    brY = int( h*math.cos(r) -w*math.sin(r) +yc)
    blX = int(-w*math.cos(r) +h*math.sin(r) +xc)
    blY = int( h*math.cos(r) +w*math.sin(r) +yc)
    return [[tlX,tlY],[trX,trY],[brX,brY],[blX,blY]]

def plotRect(img,color,xyrhw,lineWidth=1):
    tl,tr,br,bl=getCorners(xyrhw)

    cv2.line(img,tl,tr,np.array(color),lineWidth)
    cv2.line(img,tr,br,np.array(color),lineWidth)
    cv2.line(img,br,bl,np.array(color),lineWidth)
    cv2.line(img,bl,tl,np.array(color),lineWidth)


def draw_graph(outputBoxes,edgePred,edgeIndexes,image,pair_threshold,verbosity=1):
    if outputBoxes is not None:
        outputBoxes = outputBoxes.data.numpy()
    #image = image.cpu().numpy()
    b=0
    #image = (1-((1+np.transpose(image[b][:,:,:],(1,2,0)))/2.0))
    #if image.shape[2]==1:
    #    image = cv2.gray2rgb(image)



    to_write_text=[]
    bbs = outputBoxes


    #Draw pred groups (based on bb pred)
    groupCenters=[]
    predGroups = [[i] for i in range(len(bbs))]

    for group in predGroups:
        maxX=maxY=0
        minY=minX=99999999
        idColor = [random.random()/2+0.5 for i in range(3)]
        for j in group:
            conf = bbs[j,0]
            maxIndex =np.argmax(bbs[j,7:]) #TODO is this the right index?
            shade = conf#(conf-bb_thresh)/(1-bb_thresh)
            if maxIndex==0:
                color=(0,0,shade) #header
            elif maxIndex==1:
                color=(0,shade,shade) #question
            elif maxIndex==2:
                color=(shade,shade,0) #answer
            elif maxIndex==3:
                color=(shade,0,shade) #other
            else:
                raise NotImplementedError('Only 4 colors/classes implemented for drawing')
            lineWidth=1
            
            if verbosity>1 or len(group)==1:
                plotRect(image,color,bbs[j,1:6],lineWidth)
                x=int(bbs[j,1])
                y=int(bbs[j,2])

            tr,tl,br,bl=getCorners(outputBoxes[j,1:6])
            if verbosity>1:
                image[tl[1]:tl[1]+2,tl[0]:tl[0]+2]=idColor
                image[tr[1]:tr[1]+1,tr[0]:tr[0]+1]=idColor
                image[bl[1]:bl[1]+1,bl[0]:bl[0]+1]=idColor
                image[br[1]:br[1]+1,br[0]:br[0]+1]=idColor
            maxX=max(maxX,tr[0],tl[0],br[0],bl[0])
            minX=min(minX,tr[0],tl[0],br[0],bl[0])
            maxY=max(maxY,tr[1],tl[1],br[1],bl[1])
            minY=min(minY,tr[1],tl[1],br[1],bl[1])
        minX-=2
        minY-=2
        maxX+=2
        maxY+=2
        lineWidth=2
        #color=(0.5,0,1)
        if len(group)>1:
            cv2.line(image,(minX,minY),(maxX,minY),color,lineWidth)
            cv2.line(image,(maxX,minY),(maxX,maxY),color,lineWidth)
            cv2.line(image,(maxX,maxY),(minX,maxY),color,lineWidth)
            cv2.line(image,(minX,maxY),(minX,minY),color,lineWidth)
            if verbosity>1:
                image[minY:minY+3,minX:minX+3]=idColor
        if verbosity>1:
            image[maxY:maxY+1,minX:minX+1]=idColor
            image[maxY:maxY+1,maxX:maxX+1]=idColor
            image[minY:minY+1,maxX:maxX+1]=idColor
        groupCenters.append(((minX+maxX)//2,(minY+maxY)//2))



    #Draw pred pairings
    #draw_rel_thresh = relPred.max() * draw_rel_thresh_over
    numrelpred=0
    #hits = [False]*len(adjacency)
    edgesToDraw=[]
    if edgeIndexes is not None:
        for i,(g1,g2) in enumerate(edgeIndexes):
            
            if edgePred[i]>pair_threshold:
                x1,y1 = groupCenters[g1]
                x2,y2 = groupCenters[g2]
                edgesToDraw.append((i,x1,y1,x2,y2))


        lineColor = (0,0.8,0)
        for i,x1,y1,x2,y2 in edgesToDraw:
            cv2.line(image,(x1,y1),(x2,y2),lineColor,2)






    image*=255
    return image
    #cv2.imwrite(path,image)


