from skimage import color

def AI2D_printer(data, target output, metrics, outDir, startIndex):
    #for key, value in metrics.items():
    #    print(key+': '+value)

    batchSize = data.shape[0]
    for i in range(batchSize):
        image = data[i][0:3,:,:].transpose((1,2,0))
        queryMask = data[i][3,:,:]

        grayIm = color.rgb2grey(image)
        grayIm = np.stack([grayIm, grayIm, grayIm],axis=2)
