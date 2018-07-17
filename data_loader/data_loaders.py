import torch
import torch.utils.data
import numpy as np
from datasets.ai2d import AI2D
from torchvision import datasets, transforms
import skimage.transform as sktransform
from base import BaseDataLoader

from datasets.testai2d import display


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, config):
        super(MnistDataLoader, self).__init__(config)
        self.data_dir = config['data_loader']['data_dir']
        self.data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=256, shuffle=False)
        self.x = []
        self.y = []
        for data, target in self.data_loader:
            self.x += [i for i in data.numpy()]
            self.y += [i for i in target.numpy()]
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def __next__(self):
        batch = super(MnistDataLoader, self).__next__()
        batch = [np.array(sample) for sample in batch]
        return batch

    def _pack_data(self):
        packed = list(zip(self.x, self.y))
        return packed

    def _unpack_data(self, packed):
        unpacked = list(zip(*packed))
        unpacked = [list(item) for item in unpacked]
        return unpacked

    def _update_data(self, unpacked):
        self.x, self.y = unpacked

    def _n_samples(self):
        return len(self.x)

def getDataLoader(config,split):
        data_set_name = config['data_loader']['data_set_name']
        data_dir = config['data_loader']['data_dir']
        batch_size = config['data_loader']['batch_size']
        if 'augmentation_params' in config['data_loader']:
            aug_param = config['data_loader']['augmentation_params']
        else:
            aug_param = None
        shuffle = config['data_loader']['shuffle']
        shuffleValid = config['validation']['shuffle']
        if data_set_name=='AI2D':
            patchSize=config['data_loader']['patch_size']
            centerJitterFactor=config['data_loader']['center_jitter']
            sizeJitterFactor=config['data_loader']['size_jitter']
            dataset=AI2D(dirPath=data_dir, split=split, config=config)
            if split=='train':
                validation=torch.utils.data.DataLoader(dataset.splitValidation(config), batch_size=batch_size, shuffle=shuffleValid, collate_fn=resizeMiniBatchF(patchSize,0,0))
            else:
                validation=None
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=resizeMiniBatchF(patchSize,centerJitterFactor,sizeJitterFactor)), validation

def resizeMiniBatchF(patchSize, centerJitterFactor, sizeJitterFactor):
    """
    Returns function which crops and pads data to include all masks and
    """

    resizeImage=transforms.Resize((patchSize, patchSize))
    #resizeLabel=transforms.Resize((patchSize, patchSize))

    def resizeMiniBatch(data):
        #maxH=0
        #maxW=0
        #for image,label in data:
        #    if image.shape[1]>maxH:
        #        maxH=image.shape[1]
        #    if image.shape[2]>maxW:
        #         maxW=image.shape[2]

        #newSource = torch.zeros(len(data),4,patchSize,patchSize, dtype=torch.float32)
        #newTarget = torch.zeros(len(data),patchSize,patchSize, dtype=torch.float32)
        resizedImages = []
        resizedLabels = []
        for index, (image,label,xQueryC,yQueryC,reach,x0,y0,x1,y1,id) in enumerate(data):
            
            xc = int(min(max( xQueryC + np.random.normal(0,reach*centerJitterFactor) ,0),image.shape[2]-1))
            yc = int(min(max( yQueryC + np.random.normal(0,reach*centerJitterFactor) ,0),image.shape[1]-1))
            radius = int(reach + np.random.normal(reach*centerJitterFactor,reach*sizeJitterFactor))
            if radius<=0:
                radius=int(reach//2)
            #make radius smaller if we go off image
            if xc+radius>image.shape[2]-1:
                radius=image.shape[2]-1-xc
            if xc-radius<0:
                radius=xc
            if yc+radius+1>image.shape[1]:
                radius=image.shape[1]-yc-1
            if yc-radius<0:
                radius=yc

            if radius<0:
                print('rad neg1')
                radius=int(reach//2)

            #make radius big enough to include all masks
            if xc+radius<x1:
                radius=x1-xc
            if xc-radius>x0:
                radius = xc-x0
            if yc+radius<y1:
                radius=y1-yc
            if yc-radius>y0:
                radius = yc-y0

            if radius<0:
                print('rad neg2')
                radius=int(reach//2)
            
            #xc=int(xQueryC)
            #yc=int(yQueryC)
            #radius=int(reach)

            cropOutX0 = int(max(xc-radius,0))
            cropOutY0 = int(max(yc-radius,0))
            cropOutX1 = int(min(xc+radius+1,image.shape[2]))
            cropOutY1 = int(min(yc+radius+1,image.shape[1]))
            size = (cropOutY1-cropOutY0,cropOutX1-cropOutX0)
            print((xQueryC,yQueryC,reach,x0,y0,x1,y1))
            print((xc,yc,radius))
            print((cropOutX0, cropOutY0, cropOutX1, cropOutY1))
            print(image.shape)
            print(size)
            #imageP = image[:,cropOutY0:cropOutY1,cropOutX0:cropOutX1]
            #imagePatch = torch.from_numpy(imageP)
            #imagePath = imagePatch[:,cropOutY0:cropOutY1,cropOutX0:cropOutX1]
            #labelPatch = torch.from_numpy(label)[cropOutY0:cropOutY1,cropOutX0:cropOutX1]
            assert(size[0]<=radius*2+1 and size[1]<=radius*2+1)
            if size[0]!=radius or size[1]!=radius:

                diffH = (radius*2+1)-size[0]
                if diffH<0:
                    print('dif neg')
                    print(radius)
                    print(size)
                if diffH==0 or centerJitterFactor==0:
                    padTop=0
                else:
                    if (diffH<0):
                        print('wjat?')
                        print(radius)
                        print(size)
                    padTop = np.random.randint(0,diffH)

                diffW = (radius*2+1)-size[1]
                if diffW==0 or centerJitterFactor==0:
                    padLeft=0
                else:
                    padLeft = np.random.randint(0,diffW)

                #imagePatch = torch.zeros(image.shape[0],radius*2+1,radius*2+1, dtype=torch.float32)
                #imagePatch[:,padTop:size[0]+padTop,padLeft:size[1]+padLeft] = torch.from_numpy(image[:,cropOutY0:cropOutY1,cropOutX0:cropOutX1].copy())
                imagePatch = np.zeros((image.shape[0],radius*2+1,radius*2+1), dtype=np.float32)
                imagePatch[:,padTop:size[0]+padTop,padLeft:size[1]+padLeft] = image[:,cropOutY0:cropOutY1,cropOutX0:cropOutX1]
                #tempLabel = labelPatch
                labelPatch = np.zeros((radius*2+1,radius*2+1), dtype=np.float32)
                labelPatch[padTop:size[0]+padTop,padLeft:size[1]+padLeft] = label[cropOutY0:cropOutY1,cropOutX0:cropOutX1]

            #resizedImages.append(sktransform.resize(imagePatch.transpose((1, 2, 0)),(patchSize,patchSize)).transpose((2, 0, 1)))
            #resizedLabels.append(sktransform.resize(labelPatch, (patchSize,patchSize)))
            resizedImages.append(imagePatch)
            resizedLabels.append(labelPatch)

            #print(image.shape)
            #print(newSource.shape)
            #print((index,':',str(padTop)+':'+str(image.shape[1]+padTop),str(padLeft)+':'+str(image.shape[2]+padLeft)))
            #newSource[index][:,padTop:imagePatch.shape[1]+padTop,padLeft:imagePatch.shape[2]+padLeft] = imagePatch
            #newTarget[index][padTop:labelPatch.shape[0]+padTop,padLeft:labelPatch.shape[1]+padLeft] = labelPatch
        for i in range(len(resizedImages)):
            display((resizedImages[i],resizedLabels[i],0,0,0,0,0,0,0,id))

        return np.stack(resizedImages,axis=0), np.stack(resizedLabels,axis=0)

        #return newSource, newTarget
    return resizeMiniBatch



    

