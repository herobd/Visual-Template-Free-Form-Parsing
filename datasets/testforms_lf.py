from datasets.forms_lf import FormsLF
from datasets import forms_detect
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch

def display(img, positions_xyxy, positions_xyrs, step_count):
    #batchSize = data['img'].size(0)
    #for b in range(batchSize):
        #print (data['img'].size())
        #img = (data['img'][0].permute(1,2,0)+1)/2.0
        #print(data['img'].shape)
        img = (img[0].permute(1,2,0)+1)/2.0
        #print(img.shape)
        print('steps= {}'.format(step_count))



        fig = plt.figure()
        #gs = gridspec.GridSpec(1, 3)

        ax_im = plt.subplot()
        ax_im.set_axis_off()
        ax_im.imshow(img)
    
        for ps in positions_xyxy:
            ps=ps.numpy()[0]
            ax_im.plot(ps[0],ps[1],'ro')

        plt.show()
    #print('batch complete')


if __name__ == "__main__":
    dirPath = sys.argv[1]
    if len(sys.argv)>2:
        start = int(sys.argv[2])
    else:
        start=0
    data=FormsLF(dirPath=dirPath,split='train',config={'no_blanks':True, "augment":True, "only_types": ['horz']#["text","field"]
})
    #display(data[0])
    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
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
            display(*dataLoaderIter.next())
    except StopIteration:
        print('done')
