import torch
import torch.utils.data
import numpy as np
from datasets.ai2d import AI2D
from torchvision import datasets, transforms
from base import BaseDataLoader


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
            dataset=AI2D(dirPath=data_dir, split=split, config=config)
            if split=='train':
                validation=torch.utils.data.DataLoader(dataset.splitValidation(config), batch_size=batch_size, shuffle=shuffleValid, collate_fn=padMiniBatch)
            else:
                validation=None
            patchSize=config['data_loader']['patch_size']
            centerJitterFactor=config['data_loader']['center_jitter']
            sizeJitterFactor=config['data_loader']['size_jitter']
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=resizeMiniBatchF(patchSize,centerJitterFactor,sizeJitterFactor), validation

def resizeMiniBatchF(patchSize, centerJitterFactor, sizeJitterFactor):
    """
    Returns function which crops and pads data to include all masks and
    """

    def resizeMiniBatch(data):
        #maxH=0
        #maxW=0
        #for image,label in data:
        #    if image.shape[1]>maxH:
        #        maxH=image.shape[1]
        #    if image.shape[2]>maxW:
        #         maxW=image.shape[2]

        newSource = torch.zeros(len(data),4,patchSize,patchSize, dtype=torch.float32)
        newTarget = torch.zeros(len(data),patchSize,patchSize, dtype=torch.float32)
        for index, (image,label,xQueryC,yQueryC,reach,x0,y0,x1,y1) in enumerate(data):
            xc = xQueryC + np.random.normal(0,reach*centerJitterFactor)
            xc = xQueryC + np.random.normal(0,reach*centerJitterFactor)
            radius = reach + np.random.normal(reach*centerJitterFactor,reach*sizeJitterFactor)
            #make radius smaller if we go off image
            if xc+radius>image.shape[2]-1:
                radius=image.shape[2]-1-xc
            if xc-radius<0:
                radius=xc
            if yc+radius+1>image.shape[1]:
                radius=image.shape[1]-yc-1
            if yc-radius<0:
                radius=yc

            #make radius big enough to include all masks
            if xc+radius>x1:
                radius=x1-xc
            if xc-radius<x0:
                radius = xc-x0
            if yc+radius>y1:
                radius=y1-yc
            if yc-radius<y0:
                radius = yc-y0

            cropOutX0 = max(xc-radius,0)
            cropOutY0 = max(yc-radius,0)
            cropOutX1 = min(xc+radius+1,image.shape[2])
            cropOutY1 = min(yc+radius+1,image.shape[1])

            diffH = pathSize-(cropOutX1-cropOutX0)
            if diffH==0:
                padTop=0
            else:
                padTop = np.random.randint(0,diffH)
            diffW = patchSize-(cropOutY1-cropOutY0)
            if diffW==0:
                padLeft=0
            else:
                padLeft = np.random.randint(0,diffW)

            #print(image.shape)
            #print(newSource.shape)
            #print((index,':',str(padTop)+':'+str(image.shape[1]+padTop),str(padLeft)+':'+str(image.shape[2]+padLeft)))
            newSource[index][:,padTop+cropOutY0:padTop+cropOutY1,padLeft+cropOutX0:padLeft+cropOutX1] = torch.from_numpy(image[)
            newTarget[index][padTop:image.shape[1]+padTop,padLeft:image.shape[2]+padLeft] = torch.from_numpy(label)

        return newSource, newTarget
    return resize



    

