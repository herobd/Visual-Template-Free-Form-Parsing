from datasets.ai2d import AI2D
from datasets.forms_pair import FormsPair
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np

def display(data):
    (q,r) =data #,xc,yc,re,x0,y0,x1,y1,id) = data
    #print(id)
    img = q[0:3:,:].transpose((1,2,0))
    qMask = q[3,:,:]

    #print(img.shape)
    #print(qMask.shape)
    #print(r.shape)

    #points = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 3)

    ax_im = plt.subplot(gs[1])
    ax_im.set_axis_off()
    ax_im.imshow(img)
    #ax_im.plot([xc],[yc],'go')
    #ax_im.add_patch(Polygon(points,facecolor=None,edgecolor='red', fill=False))

    ax_q = plt.subplot(gs[0])
    ax_q.set_axis_off()
    ax_q.imshow(qMask)
    #ax_q.plot([xc],[yc],'go')
    #ax_q.add_patch(Polygon(points,facecolor=None,edgecolor='red', fill=False))

    ax_r = plt.subplot(gs[2])
    ax_r.set_axis_off()
    ax_r.imshow(r)
    #ax_r.plot([xc],[yc],'go')
    #ax_r.add_patch(Polygon(points,facecolor=None,edgecolor='red', fill=False))

    plt.show()

if __name__ == "__main__":
    dirPath = sys.argv[1]
    if len(sys.argv)>2:
        start = int(sys.argv[2])
    else:
        start=0
    
    if 'form' in dirPath:
        data = FormsPair(dirPath=dirPath, split='train', config={"patch_size": 512,"center_jitter": 0.1,"size_jitter": 0.2}, test=False)
    else:
        data = AI2D(dirPath=dirPath, split='train', config={'data_loader':{}}, test=False)

    for i in range(start,len(data)):
        display(data[i])
