from datasets.funsd_box_detect import FUNSDBoxDetect
from datasets import funsd_box_detect
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch
from utils import img_f

def display(data):
    batchSize = data['img'].size(0)
    for b in range(batchSize):
        img = (data['img'][b].permute(1,2,0)+1)/2.0
        print(data['imgName'][b])



        #fig = plt.figure()
        #gs = gridspec.GridSpec(1, 3)

        #ax_im = plt.subplot()
        #ax_im.set_axis_off()
        #if img.shape[2]==1:
        #    ax_im.imshow(img[0])
        #else:
        #    ax_im.imshow(img)
        if img.shape[2]==1:
            img = img_f.gray2rgb(img)

        colors = {  'text_start_gt':'g-',
                    'text_end_gt':'b-',
                    'field_start_gt':'r-',
                    'field_end_gt':'y-',
                    'table_points':'co',
                    'start_of_line':'y-',
                    'end_of_line':'c-',
                    }
        print('num bb:{}'.format(data['bb_sizes'][b]))
        for i in range(data['bb_sizes'][b]):
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
                color = (1,0,0)#'r-'
            elif question:
                color = (0,0,1)#'b-'
            elif answer:
                color = (0,1,0)#'g-'
            elif other:
                color = (1,1,0)#'y-'
            else:
                assert(False)
            tr = (math.cos(rot)*w-math.sin(rot)*h +xc, -math.sin(rot)*w-math.cos(rot)*h +yc)
            tl = (-math.cos(rot)*w-math.sin(rot)*h +xc, math.sin(rot)*w-math.cos(rot)*h +yc)
            br = (math.cos(rot)*w+math.sin(rot)*h +xc, -math.sin(rot)*w+math.cos(rot)*h +yc)
            bl = (-math.cos(rot)*w+math.sin(rot)*h +xc, math.sin(rot)*w+math.cos(rot)*h +yc)
            #print([tr,tl,br,bl])

            #ax_im.plot([tr[0],tl[0],bl[0],br[0],tr[0]],[tr[1],tl[1],bl[1],br[1],tr[1]],color)
            img_f.polylines(img,np.array([tr,tl,bl,br]),'transparent',color)
            

        #plt.show()
        img_f.imshow('page',img)
        img_f.show()
    print('batch complete')


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
    data=FUNSDBoxDetect(dirPath=dirPath,split='train',config={
        "split_to_lines": True,
        'rescale_range':[1,1],
        #'crop_params':{ "crop_size":[652,1608], 
        'crop_params':{ "crop_size":[2000,1000], 
                        "pad":0, 
                        "rot_degree_std_dev":3,
                        "flip_horz": False,
                        "flip_vert": False}, 
        'color': False
})
    data.cluster(start,repeat,'FUNSDLines_anchors_maxW100_{}.json',use_max_width=100)

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=funsd_box_detect.collate)
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
