from datasets.forms_graph_pair import FormsGraphPair
from datasets import forms_graph_pair
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch

def display(data):
    b=0

    #print (data['img'].size())
    #img = (data['img'][0].permute(1,2,0)+1)/2.0
    img = (data['img'][b].permute(1,2,0)+1)/2.0
    #print(img.shape)
    #print(data['pixel_gt']['table_pixels'].shape)
    print(data['imgName'])



    fig = plt.figure()
    #gs = gridspec.GridSpec(1, 3)

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
        text=data['bb_gt'][b,i,13]
        field=data['bb_gt'][b,i,14]
        if text>0:
            color = 'b-'
        else:
            color = 'r-'
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

        ax_im.plot([x1,x2],[y1,y2],'g-')
        #print('{} to {}, {} - {}'.format(ind1,ind2,(x1,y1),(x2,y2)))
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
    data=FormsGraphPair(dirPath=dirPath,split='train',config={
        'color':False,
        'crop_to_page':False,
        'rescale_range':[1,1],
        'Xrescale_range':[0.4,0.65],
        'Xcrop_params':{"crop_size":[652,1608],"pad":0}, 
        'no_blanks':True,
        "swap_circle":True,
        'no_graphics':True,
        'rotation':False,
        'only_opposite_pairs':True,
        #"only_types": ["text_start_gt"]
})

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=forms_graph_pair.collate)
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
