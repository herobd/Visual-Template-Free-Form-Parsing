import os
import json
import logging
import argparse
import torch
from model.model import *
from model.unet import UNet
from model.metric import *
from data_loader import getDataLoader
from utils.printers import *

logging.basicConfig(level=logging.INFO, format='')

def _to_tensor(gpu, data):
    if type(data) is np.ndarray:
        data = torch.FloatTensor(data.astype(np.float32))
    elif type(data) is torch.Tensor:
        data = data.type(torch.FloatTensor)
    if gpu is not None:
        data = data.to(gpu)
    return data

def _eval_metrics(metrics, output, target):
    acc_metrics = np.zeros(len(metrics))
    for i, metric in enumerate(metrics):
        acc_metrics[i] += metric(output, target)
    return acc_metrics

def _eval_metrics_ind(metrics, output, target):
    acc_metrics = np.zeros((output.shape[0],len(metrics)))
    for ind in range(output.shape[0]):
        for i, metric in enumerate(metrics):
            acc_metrics[ind,i] += metric(output[ind:ind+1], target[ind:ind+1])
    return acc_metrics

def main(resume,saveDir,numberOfImages,index,gpu=None):
    np.random.seed(1234)
    checkpoint = torch.load(resume)
    config = checkpoint['config']
    config['data_loader']['shuffle']=False
    config['validation']['shuffle']=False

    data_loader, valid_data_loader = getDataLoader(config,'train')
    #valid_data_loader = data_loader.split_validation()

    model = eval(config['arch'])(config['model'])
    model.eval()
    model.summary()

    if gpu is not None:
        model = model.to(gpu)

    metrics = [eval(metric) for metric in config['metrics']]

    model.load_state_dict(checkpoint['state_dict'])

    saveFunc = eval(config['data_loader']['data_set_name']+'_printer')

    step=5
    batchSize = config['data_loader']['batch_size']

    #numberOfImages = numberOfImages//config['data_loader']['batch_size']
    train_iter = iter(data_loader)
    valid_iter = iter(valid_data_loader)

    if index is None:
        trainDir = os.path.join(saveDir,'train_'+config['name'])
        validDir = os.path.join(saveDir,'valid_'+config['name'])
        if not os.path.isdir(trainDir):
            os.mkdir(trainDir)
        if not os.path.isdir(validDir):
            os.mkdir(validDir)

        val_metrics_sum = np.zeros(len(metrics))

        curVI=0

        for index in range(0,numberOfImages,step*batchSize):
            for trainIndex in range(index,index+step*batchSize, batchSize):
                if trainIndex/batchSize < len(data_loader):
                    data, target = train_iter.next() #data_loader[trainIndex]
                    dataT = _to_tensor(gpu,data)
                    output = model(dataT)
                    data = data.cpu().data.numpy()
                    output = output.cpu().data.numpy()
                    target = target.data.numpy()
                    metricsO = _eval_metrics_ind(metrics,output, target)
                    saveFunc(data,target,output,metricsO,trainDir,trainIndex)
            
            for validIndex in range(index,index+step*batchSize, batchSize):
                if validIndex/batchSize < len(valid_data_loader):
                    data, target = valid_iter.next() #valid_data_loader[validIndex]
                    curVI+=0
                    dataT  = _to_tensor(gpu,data)
                    output = model(dataT)
                    data = data.cpu().data.numpy()
                    output = output.cpu().data.numpy()
                    target = target.data.numpy()
                    metricsO = _eval_metrics_ind(metrics,output, target)
                    saveFunc(data,target,output,metricsO,validDir,validIndex)

                    val_metrics_sum += metricsO.sum(axis=0)/metricsO.shape[0]
                    
        if gpu is not None:
            try:
                for vi in range(curVI,len(valid_data_loader)):
                    data, target = valid_iter.next() #valid_data_loader[validIndex]
                    data  = _to_tensor(gpu,data)
                    output = model(data)
                    output = output.cpu().data.numpy()
                    target = target.data.numpy()
                    metricsO = _eval_metrics(metrics,output, target)
                    val_metrics_sum += metricsO
            except StopIteration:
                print('ERROR: ran out of valid batches early. Expected {} more'.format(len(valid_data_loader)-vi))
            
            val_metrics_sum /= len(valid_data_loader)
            print('Validation metrics')
            for i in range(len(metrics)):
                print(metrics[i].__name__ + ': '+str(val_metrics_sum[i]))

    else:
        batchIndex = index//batchSize
        inBatchIndex = index%batchSize
        for i in range(batchIndex+1):
            data, target = train_iter.next()
        data, target = data[inBatchIndex:inBatchIndex+1], target[inBatchIndex:inBatchIndex+1]
        dataT = _to_tensor(gpu,data)
        output = model(dataT)
        data = data.cpu().data.numpy()
        output = output.cpu().data.numpy()
        target = target.data.numpy()
        #print (output.shape)
        #print ((output.min(), output.amin()))
        #print (target.shape)
        #print ((target.amin(), target.amin()))
        metricsO = _eval_metrics_ind(metrics,output, target)
        saveFunc(data,target,output,metricsO,saveDir,index)


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Evaluator/Displayer')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--savedir', default=None, type=str,
                        help='path to directory to save result images (default: None)')
    parser.add_argument('-i', '--index', default=None, type=int,
                        help='index on instance to process (default: None)')
    parser.add_argument('-n', '--number', default=100, type=int,
                        help='number of images to save out (from each train and valid) (default: 100)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='gpu number (default: cpu only)')

    args = parser.parse_args()

    config = None
    if args.checkpoint is None or args.savedir is None:
        print('Must provide checkpoint (with -c) and save dir (with -d)')
        exit()

    main(args.checkpoint, args.savedir, args.number, args.index, gpu=args.gpu)
