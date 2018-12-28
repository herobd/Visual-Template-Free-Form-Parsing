from datasets.forms_box_detect import FormsBoxDetect
from datasets import forms_box_detect
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch

def display(data):
    batchSize = data['img'].size(0)
    for b in range(batchSize):
        #print (data['img'].size())
        #img = (data['img'][0].permute(1,2,0)+1)/2.0
        img = (data['img'][b].permute(1,2,0)+1)/2.0
        #print(img.shape)
        #print(data['pixel_gt']['table_pixels'].shape)
        if 'pixel_gt' in data and data['pixel_gt'] is not None:
            img[:,:,1] = data['pixel_gt'][b,0,:,:]
        print(data['imgName'][b])



        fig = plt.figure()
        #gs = gridspec.GridSpec(1, 3)

        ax_im = plt.subplot()
        ax_im.set_axis_off()
        if img.shape[2]==1:
            ax_im.imshow(img[0])
        else:
            ax_im.imshow(img)

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
            text=data['bb_gt'][b,i,13]
            field=data['bb_gt'][b,i,14]
            if text>0:
                color = 'b-'
            else:
                color = 'r-'
            tr = (math.cos(rot)*w-math.sin(rot)*h +xc, -math.sin(rot)*w-math.cos(rot)*h +yc)
            tl = (-math.cos(rot)*w-math.sin(rot)*h +xc, math.sin(rot)*w-math.cos(rot)*h +yc)
            br = (math.cos(rot)*w+math.sin(rot)*h +xc, -math.sin(rot)*w+math.cos(rot)*h +yc)
            bl = (-math.cos(rot)*w+math.sin(rot)*h +xc, math.sin(rot)*w+math.cos(rot)*h +yc)
            #print([tr,tl,br,bl])

            ax_im.plot([tr[0],tl[0],bl[0],br[0],tr[0]],[tr[1],tl[1],bl[1],br[1],tr[1]],color)
            
            if data['bb_gt'].shape[2]>15:
                blank = data['bb_gt'][b,i,15]
                if blank>0:
                    ax_im.plot(tr[0],tr[1],'mo')
                paired = data['bb_gt'][b,i,16]
                if paired>0:
                    ax_im.plot(br[0],br[1],'go')


        if 'line_gt' in data and data['line_gt'] is not None:
            for name, gt in data['line_gt'].items():
                if gt is not None: 
                    #print (gt.size())
                    for i in range(data['line_label_sizes'][name][b]):
                        x0=gt[b,i,0]
                        y0=gt[b,i,1]
                        x1=gt[b,i,2]
                        y1=gt[b,i,3]
                        #print(1,'{},{}   {},{}'.format(x0,y0,x1,y1))

                        ax_im.plot([x0,x1],[y0,y1],colors[name])


        if 'point_gt' in data and data['point_gt'] is not None:
            for name, gt in data['point_gt'].items():
                if gt is not None:
                    #print (gt.size())
                    #print(data)
                    for i in range(data['point_label_sizes'][name][b]):
                        x0=gt[b,i,0]
                        y0=gt[b,i,1]

                        ax_im.plot([x0],[y0],colors[name])
        plt.show()
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
    data=FormsBoxDetect(dirPath=dirPath,split='train',config={'crop_to_page':False,'rescale_range':[0.4,0.6],
        'crop_params':{"crop_size":[652,1608], "pad":0, "rot_degree_std_dev":2}, 
        'no_blanks':False,
        'use_paired_class':True,
        "swap_circle":True,
        'no_graphics':True,
        'rotation':True,
        #"only_types": {"boxes":True}
})
    #data.cluster(start,repeat,'anchors_rot_{}.json')

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=forms_box_detect.collate)
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
