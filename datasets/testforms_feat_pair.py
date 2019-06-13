from datasets.forms_feature_pair import FormsFeaturePair
from datasets import forms_feature_pair
import math
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch

def display(data):
    b=0
    return data['data'].numpy()
    if False:
        #print (data['img'].size())
        #img = (data['img'][0].permute(1,2,0)+1)/2.0
        img =  cv2.imread(data['imgPath'])
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
    data=FormsFeaturePair(dirPath=dirPath,split='valid',config={
        "data_set_name": "FormsFeaturePair",
        "simple_dataset": True,
        "alternate_json_dir": "out_json/Simple18_staggerLight_NN",
        "data_dir": "../data/forms",
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 0,
        "no_blanks": True,
        "swap_circle":True,
        "no_graphics":True,
        "cache_resized_images": True,
        "rotation": False,
        "balance": True,
        "only_opposite_pairs": True,
        "corners":True
        #"only_types": ["text_start_gt"]
})

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=forms_feature_pair.collate)
    dataLoaderIter = iter(dataLoader)

        #if start==0:
        #display(data[0])
    for i in range(0,start):
        print(i)
        dataLoaderIter.next()
        #display(data[i])
    datas=[]
    try:
        while True:
            #print('?')
            data = display(dataLoaderIter.next())
            datas.append(data)
    except StopIteration:
        print('done')

    data = np.concatenate(datas,axis=0)
    
    #print(data.mean(axis=0))
    #print(data.std(axis=0))
    #toprint = ['']*data.shape[1]
    print('feat:\tmean:\tstddev:')
    for i in range(data.shape[1]):
        print('{}:\t{:.3}\t{:.3}'.format(i,data[:,i].mean(),data[:,i].std()))
