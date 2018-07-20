from skimage import color, io
import os
import numpy as np

def AI2D_printer(data, target, output, metrics, outDir, startIndex):
    #for key, value in metrics.items():
    #    print(key+': '+value)

    batchSize = data.shape[0]
    for i in range(batchSize):
        image = np.transpose(data[i][0:3,:,:],(1,2,0))
        queryMask = data[i][3,:,:]

        grayIm = color.rgb2grey(image)

        invQuery = 1-queryMask
        invTarget = 1-target[i]
        invOutput = output[i]<=0.0 #assume not sigmoided


        highlightIm = np.stack([grayIm*invOutput, grayIm*invTarget, grayIm*invQuery],axis=2)

        saveName = '{:06}'.format(startIndex+i)
        for j in range(metrics.shape[1]):
            saveName+='_m:{0:.3f}'.format(metrics[i,j])
        saveName+='.png'
        io.imsave(os.path.join(outDir,saveName),highlightIm)

