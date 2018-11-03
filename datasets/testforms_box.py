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
        img[:,:,1] = data['pixel_gt'][b,0,:,:]
        print(data['imgName'][b])



        fig = plt.figure()
        #gs = gridspec.GridSpec(1, 3)

        ax_im = plt.subplot()
        ax_im.set_axis_off()
        ax_im.imshow(img)

        colors = {  'text_start_gt':'g-',
                    'text_end_gt':'b-',
                    'field_start_gt':'r-',
                    'field_end_gt':'y-',
                    'table_points':'co'
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
            tr = (math.cos(rot)*w-math.sin(rot)*h +xc, math.sin(rot)*w+math.cos(rot)*h +yc)
            tl = (math.cos(rot)*-w-math.sin(rot)*h +xc, math.sin(rot)*-w+math.cos(rot)*h +yc)
            br = (math.cos(rot)*w-math.sin(rot)*-h +xc, math.sin(rot)*w+math.cos(rot)*-h +yc)
            bl = (math.cos(rot)*-w-math.sin(rot)*-h +xc, math.sin(rot)*-w+math.cos(rot)*-h +yc)
            #print([tr,tl,br,bl])

            ax_im.plot([tr[0],tl[0],bl[0],br[0],tr[0]],[tr[1],tl[1],bl[1],br[1],tr[1]],color)
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
    data=FormsBoxDetect(dirPath=dirPath,split='train',config={
        'crop_to_page':True,
        'rescale_range':[0.52,0.52],
        #'crop_params':{"crop_size":512}, 
        'no_blanks':True,
        "swap_circle":True,
        'no_graphics':True,
        'rotation':False,
        #"only_types": ["text_start_gt"]
})
    #data.cluster(start,repeat,'anchors_noRot_stats.json'.format(start))

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
