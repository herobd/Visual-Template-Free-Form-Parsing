from datasets.funsd_graph_pair import FUNSDGraphPair
from datasets import funsd_graph_pair
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch

hs=[]
ws=[]

def display(data):
    b=0

    #print (data['img'].size())
    #img = (data['img'][0].permute(1,2,0)+1)/2.0
    img = (data['img'][b].permute(1,2,0)+1)/2.0
    #print(img.shape)
    #print(data['pixel_gt']['table_pixels'].shape)
    print(data['imgName'])

    #hs.append(img.shape[0])
    #ws.append(img.shape[1])
    #return



    fig = plt.figure()

    ax_im = plt.subplot()
    ax_im.set_axis_off()
    ax_im.imshow(img[:,:,0])

    colors = {  'text_start_gt':'g-',
                'text_end_gt':'b-',
                'field_start_gt':'r-',
                'field_end_gt':'y-',
                'table_points':'co'
                }
    #print('num bb:{}'.format(data['bb_sizes'][b]))
    for i in range(data['bb_gt'].size(1)):
        xc=data['bb_gt'][b,i,0]
        yc=data['bb_gt'][b,i,1]
        rot=data['bb_gt'][b,i,2]
        h=data['bb_gt'][b,i,3]
        w=data['bb_gt'][b,i,4]
        header=data['bb_gt'][b,i,13]
        question=data['bb_gt'][b,i,14]
        answer=data['bb_gt'][b,i,15]
        other=data['bb_gt'][b,i,16]
        if header:
            color = 'r-'
        elif question:
            color = 'b-'
        elif answer:
            color = 'g-'
        elif other:
            color = 'y-'
        tr = (math.cos(rot)*w-math.sin(rot)*h +xc, math.sin(rot)*w+math.cos(rot)*h +yc)
        tl = (math.cos(rot)*-w-math.sin(rot)*h +xc, math.sin(rot)*-w+math.cos(rot)*h +yc)
        br = (math.cos(rot)*w-math.sin(rot)*-h +xc, math.sin(rot)*w+math.cos(rot)*-h +yc)
        bl = (math.cos(rot)*-w-math.sin(rot)*-h +xc, math.sin(rot)*-w+math.cos(rot)*-h +yc)
        #print([tr,tl,br,bl])

        ax_im.plot([tr[0],tl[0],bl[0],br[0],tr[0]],[tr[1],tl[1],bl[1],br[1],tr[1]],color)
    for ind1,ind2 in data['adj']:
        x1=data['bb_gt'][b,ind1,0]
        y1=data['bb_gt'][b,ind1,1]
        x2=data['bb_gt'][b,ind2,0]
        y2=data['bb_gt'][b,ind2,1]

        ax_im.plot([x1,x2],[y1,y2],'m-')
        #print('{} to {}, {} - {}'.format(ind1,ind2,(x1,y1),(x2,y2)))

    groupCenters=[]
    for group in data['gt_groups']:
        maxX=maxY=0
        minX=minY=999999999
        for i in group:
            xc=data['bb_gt'][b,i,0]
            yc=data['bb_gt'][b,i,1]
            rot=data['bb_gt'][b,i,2]
            assert(rot==0)
            h=data['bb_gt'][b,i,3]
            w=data['bb_gt'][b,i,4]
            maxX=max(maxX,xc+w)
            maxY=max(maxY,yc+h)
            minX=min(minX,xc-w)
            minY=min(minY,yc-h)
        ax_im.plot([minX,maxX,maxX,minX,minX],[minY,minY,maxY,maxY,minY],'c:')
        groupCenters.append(((minX+maxX)//2, (minY+minY)//2) )

    for g1,g2 in data['gt_groups_adj']:
        x1,y1 = groupCenters[g1]
        x2,y2 = groupCenters[g2]
        ax_im.plot([x1,x2],[y1,y2],'c-')

    plt.show()


if __name__ == "__main__":
    dirPath = sys.argv[1]
    if len(sys.argv)>2:
        start = int(sys.argv[2])
    else:
        start=0
    if len(sys.argv)>3:
        repeat = int(sys.argv[3])
    else:
        repeat=1
    data=FUNSDGraphPair(dirPath=dirPath,split='train',config={
        'color':False,
        'rescale_range':[0.8,1.2],
        '#rescale_range':[0.4,0.65],
        'crop_params':{
            "crop_size":[1000,700],
            "pad":70,
            "xxrot_degree_std_dev": 0.7}, 
        'split_to_lines': True,
})

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=funsd_graph_pair.collate)
    dataLoaderIter = iter(dataLoader)

        #if start==0:
        #display(data[0])
    for i in range(0,start):
        print(i)
        dataLoaderIter.next()
        #display(data[i])
    try:
        while True:
            #print('?')
            display(dataLoaderIter.next())
    except StopIteration:
        print('done')

    hs=np.array(hs)
    ws=np.array(ws)
    print('mean: {},{}   min: {},{}   max: {},{}'.format(hs.mean(),ws.mean(),hs.min(),ws.min,hs.max(),ws.max()))
