import os
import json
import logging
import argparse
import torch
from model.model import *
from model.unet import UNet
from model.loss import *
from model.metric import *
from data_loader import getDataLoader
from utils.printers import *
from trainer import Trainer
from logger import Logger

logging.basicConfig(level=logging.INFO, format='')

def _to_tensor(gpu, data):
    if type(data) is np.ndarray:
        data = torch.FloatTensor(data.astype(np.float32))
    elif type(data) is torch.Tensor:
        data = data.type(torch.FloatTensor)
    if gpu not None:
        data = data.to(gpu)
    return data, target

def _eval_metrics(metrics, output, target):
    acc_metrics = np.zeros(len(metrics))
    output = np.argmax(output, axis=1)
    for i, metric in enumerate(metrics):
        acc_metrics[i] += metric(output, target)
    return acc_metrics

def main(resume,saveDir,numberOfImages):
    np.random.seed(1234)
    checkpoint = torch.load(resume_path)
    config = checkpoint['config']
    config['data_loader']['shuffle']=False
    config['validation']['shuffle']=False

    data_loader, valid_data_loader = getDataLoader(config,'train')
    #valid_data_loader = data_loader.split_validation()

    model = eval(config['arch'])(config['model'])
    model.eval()
    model.summary()

    if config['cuda']:
        gpu = config['gpu']
    else:
        gpu=None

    metrics = [eval(metric) for metric in config['metrics']]

    model.load_state_dict(checkpoint['state_dict'])
    trainDir = os.path.join(saveDir,'train_'+config['name'])
    validDir = os.path.join(saveDir,'valid_'+config['name'])
    if not os.path.isdir(trainDir):
        os.mkdir(trainDir)
    if not os.path.isdir(validDir):
        os.mkdir(validDir)

    saveFunc = eval(config['data_loader']['data_set_name']+'_printer')

    step=10
    batchSize = config['data_loader']['batch_size']

    #numberOfImages = numberOfImages//config['data_loader']['batch_size']

    for index in range(0,numberOfImages,step*batchSize):
        for trainIndex in range(index,index+step*batchSize, batchSize):
            if trainIndex/batchSize < len(data_loader):
                data, target = data_loader[trainIndex]
                data = _to_tensor(gpu,data)
                output = self.model(data)
                data = data.cpu().data.numpy()
                output = ouput.cpu().data.numpy()
                metricsO = _eval_metrics(metrics,output, target)
                saveFunc(data,target,output,metricsO,trainDir,trainIndex)
        for validIndex in range(index,index+step*batchSize, batchSize):
            if validIndex/batchSize < len(valid_data_loader):
                data, target = valid_data_loader[validIndex]
                data  = _to_tensor(gpu,data)
                output = self.model(data)
                data = data.cpu().data.numpy()
                output = ouput.cpu().data.numpy()
                metricsO = _eval_metrics(metrics,output, target)
                saveFunc(data,target,output,metricsO,validDir,validIndex)

    trainer = Trainer(model, loss, metrics,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Evaluator/Displayer')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--savedir', default=None, type=str,
                        help='path to directory to save result images (default: None)')
    parser.add_argument('-n', '--number', default=100, type=int,
                        help='number of images to save out (from each train and valid) (default: 100)')

    args = parser.parse_args()

    config = None
    if args.checkpoint is None or args.savedir is None:
        print('Must provide checkpoint (with -c) and save dir (with -d)')
        exit()

    main(args.checkpoint, args.savedir, args.number)
