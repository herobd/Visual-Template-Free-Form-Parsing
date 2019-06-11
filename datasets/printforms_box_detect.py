from datasets.forms_box_detect import FormsBoxDetect
from datasets import forms_box_detect
import math
import sys
import os, cv2
import numpy as np
import torch

def saveBoxes(data,dest):
    batchSize = data['img'].size(0)
    for b in range(batchSize):
        #print (data['img'].size())
        #img = (data['img'][0].permute(1,2,0)+1)/2.0
        img = 255*(1-data['img'][b].permute(1,2,0))/2.0
        #print(img.shape)
        #print(data['pixel_gt']['table_pixels'].shape)
        if 'pixel_gt' in data and data['pixel_gt'] is not None:
            img[:,:,1] = data['pixel_gt'][b,0,:,:]
        imgName=(data['imgName'][b])


        img=img.numpy().astype(np.uint8)

        for i in range(data['bb_sizes'][b]):
            xc=data['bb_gt'][b,i,0]
            yc=data['bb_gt'][b,i,1]
            rot=data['bb_gt'][b,i,2]
            h=data['bb_gt'][b,i,3]
            w=data['bb_gt'][b,i,4]
            text=data['bb_gt'][b,i,13]
            field=data['bb_gt'][b,i,14]
            if text>0:
                sub = 'text'
            else:
                sub = 'field'
            tr = (math.cos(rot)*w-math.sin(rot)*h +xc, -math.sin(rot)*w-math.cos(rot)*h +yc)
            tl = (-math.cos(rot)*w-math.sin(rot)*h +xc, math.sin(rot)*w-math.cos(rot)*h +yc)
            br = (math.cos(rot)*w+math.sin(rot)*h +xc, -math.sin(rot)*w+math.cos(rot)*h +yc)
            bl = (-math.cos(rot)*w+math.sin(rot)*h +xc, math.sin(rot)*w+math.cos(rot)*h +yc)
            #print([tr,tl,br,bl])
            assert(rot==0)
            crop = img[int(tl[1]):int(br[1])+1,int(tl[0]):int(br[0])+1]
            path = os.path.join(dest,sub,'{}_b{}.png'.format(imgName,i))
            cv2.imwrite(path,crop)





if __name__ == "__main__":
    dirPath = sys.argv[1]
    if len(sys.argv)>2:
        directory = sys.argv[2]
    else:
        print('need dest dir')
        exit()
    dirText = os.path.join(directory,'text')
    dirField = os.path.join(directory,'field')
    if not os.path.exists(dirText):
        os.makedirs(dirText)
    if not os.path.exists(dirField):
        os.makedirs(dirField)
    data=FormsBoxDetect(dirPath=dirPath,split='valid',config={'crop_to_page':False,
        #'rescale_range':[0.45,0.6],
        #'rescale_range':[0.52,0.52],
        'rescale_range':[1,1],
        #'crop_params':{ "crop_size":[652,1608], 
        #                "pad":0, 
        #                "rot_degree_std_dev":1.5,
        #                "flip_horz": True,
        #                "flip_vert": True}, 
        'no_blanks':True,
        'use_paired_class':True,
        "swap_circle":True,
        'no_graphics':True,
        'rotation':False,
        "only_types": {"boxes":True}
})
    #data.cluster(start,repeat,'anchors_rot_{}.json')

    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=forms_box_detect.collate)
    dataLoaderIter = iter(dataLoader)

    try:
        while True:
            #print('?')
            saveBoxes(dataLoaderIter.next(),directory)
    except StopIteration:
        print('done')
