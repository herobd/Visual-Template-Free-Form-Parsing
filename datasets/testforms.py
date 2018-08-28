from datasets.forms_detect import FormsDetect
from datasets import forms_detect
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
    data=FormsDetect(dirPath=dirPath,split='train',config={'crop_to_page':False,'rescale_range':[450,800],
        'crop_params':{"crop_size":512}, 
        'no_blanks':True, #"only_types": ["text_start_gt"]
})
    #display(data[0])
    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=forms_detect.collate)
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
