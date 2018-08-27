import torch
import torch.utils.data
import numpy as np
from datasets.ai2d import AI2D
from datasets import forms_detect
from datasets.forms_detect import FormsDetect
from datasets.forms_pair import FormsPair
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
        if 'num_workers' in config['data_loader']:
            numDataWorkers = config['data_loader']['num_workers']
        else:
            numDataWorkers = 1
        shuffleValid = config['validation']['shuffle']

        if data_set_name=='AI2D':
            dataset=AI2D(dirPath=data_dir, split=split, config=config)
            if split=='train':
                validation=torch.utils.data.DataLoader(dataset.splitValidation(config), batch_size=batch_size, shuffle=shuffleValid, num_workers=numDataWorkers)
            else:
                validation=None
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=numDataWorkers), validation
        elif data_set_name=='FormsDetect':
            if split=='train':
                trainData = FormsDetect(dirPath=data_dir, split='train', config=config['data_loader'])
                trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=shuffle, num_workers=numDataWorkers, collate_fn=forms_detect.collate)
                validData = FormsDetect(dirPath=data_dir, split='valid', config=config['validation'])
                validLoader = torch.utils.data.DataLoader(validData, batch_size=batch_size, shuffle=shuffleValid, num_workers=numDataWorkers, collate_fn=forms_detect.collate)
                return trainLoader, validLoader
            elif split=='test':
                testData = FormsDetect(dirPath=data_dir, split='test', config=config['data_loader'])
                testLoader = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False, num_workers=numDataWorkers, collate_fn=forms_detect.collate)
                return testLoader, None
        elif data_set_name=='FormsPair':
            if split=='train':
                trainData = FormsPair(dirPath=data_dir, split='train', config=config['data_loader'])
                trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=shuffle, num_workers=numDataWorkers)
                validData = FormsPair(dirPath=data_dir, split='valid', config=config['validation'])
                validLoader = torch.utils.data.DataLoader(validData, batch_size=batch_size, shuffle=shuffleValid, num_workers=numDataWorkers)
                return trainLoader, validLoader
            elif split=='test':
                testData = FormsPair(dirPath=data_dir, split='test', config=config['data_loader'])
                testLoader = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False, num_workers=numDataWorkers)
                return testLoader, None




    

