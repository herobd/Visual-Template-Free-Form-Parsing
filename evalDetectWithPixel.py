import os
import json
import logging
import argparse
import torch
from model import *
from model.metric import *
from data_loader import getDataLoader
from evaluators import *
import math
from collections import defaultdict

from datasets.forms_detect import FormsDetect
from datasets import forms_detect

logging.basicConfig(level=logging.INFO, format='')


def main(resumeDetect, resumePixel,saveDir,numberOfImages,index,gpu=None, shuffle=False, setBatch=None, configDetect=None, configPixel):
    np.random.seed(1234)
    #if gpu is None:
    #    loc = 
    checkpointDetect = torch.load(resumDetect, map_location=lambda storage, location: storage)
    checkpointPixel = torch.load(resumPixel, map_location=lambda storage, location: storage)
    if configDetect is None:
        configDetect = checkpoint['config']
    else:
        configDetect = json.load(open(configDetect))
    if configPixel is None:
        configPixel = checkpoint['config']
    else:
        configPixel = json.load(open(configPixel))
        
    #config['data_loader']['batch_size']=math.ceil(config['data_loader']['batch_size']/2)
    
    #config['data_loader']['shuffle']=shuffle
    #config['data_loader']['rot']=False
    #config['validation']['shuffle']=shuffle
    #config['validation']


    #print(config['data_loader'])
    #if setBatch is not None:
        #config['data_loader']['batch_size']=setBatch
        #config['validation']['batch_size']=setBatch
    #batchSize = config['data_loader']['batch_size']
    #if 'batch_size' in config['validation']:
    #    vBatchSize = config['validation']['batch_size']
    #else:
    #    vBatchSize = batchSize
    #data_loader, valid_data_loader = getDataLoader(config,'train')

    if 'state_dict' in checkpointDetect:
        modelDetect = eval(configDetect['arch'])(configDetect['model'])
        modelDetect.load_state_dict(checkpointDetect['state_dict'])
    else:
        modelDetect = checkpoint['model']
    modelDetect.eval()
    modelDetect.summary()

    if gpu is not None:
        modelDetect = modelDetect.to(gpu)

    #metrics = [eval(metric) for metric in config['metrics']]




    step=5

    #numberOfImages = numberOfImages//config['data_loader']['batch_size']
    #print(len(data_loader))
    valid_iter = iter(valid_data_loader)

    def __to_tensor(instance,gpu):
        data = instance['img']
        if 'bb_gt' in instance:
            targetBBs = instance['bb_gt']
            targetBBs_sizes = instance['bb_sizes']
        else:
            targetBBs = {}
            targetBBs_sizes = {}
        if 'point_gt' in instance:
            targetPoints = instance['point_gt']
            targetPoints_sizes = instance['point_label_sizes']
        else:
            targetPoints = {}
            targetPoints_sizes = {}
        if 'pixel_gt' in instance:
            targetPixels = instance['pixel_gt']
        else:
            targetPixels = None
        if type(data) is np.ndarray:
            data = torch.FloatTensor(data.astype(np.float32))
        elif type(data) is torch.Tensor:
            data = data.type(torch.FloatTensor)

        def sendToGPU(targets):
            new_targets={}
            for name, target in targets.items():
                if target is not None:
                    new_targets[name] = target.to(gpu)
                else:
                    new_targets[name] = None
            return new_targets

        if gpu is not None:
            data = data.to(gpu)
            if targetBBs is not None:
                targetBBs=targetBBs.to(gpu)
            targetPoints=sendToGPU(targetPoints)
            if targetPixels is not None:
                targetPixels=targetPixels.to(gpu)
        return data, targetBBs, targetBBs_sizes, targetPoints, targetPoints_sizes, targetPixels


        validDir = os.path.join(saveDir,'valid_detect_'+configDetect['name'])
        if not os.path.isdir(validDir):
            os.mkdir(validDir)


        for instance in valid_iter:
            imageName = instance['imgName']
            scale = instance['scale']
            print('batch: {}'.format(imageName)),end='\r')
            dataT, targetBBsT, targetBBsSizes, targetPointsT, targetPointsSizes, targetPixelsT = __to_tensor(instance,gpu)
            outputBBs, outputOffsets, outputPoints, outputPixels = modelDetector(dataT)
            #data = data.cpu().data.numpy()
            maxConf = outputBBs[:,:,0].max().item()
            threshConf = max(maxConf*0.92,0.5)
            outputBBs = non_max_sup_iou(outputBBs.cpu(),threshConf,0.4)
            for b in len(outputBBs):
                out = outputBBs[b].data.numpy()
                saveFile = os.path.join(validDir,'{}'.format(imageName[b]))
                out[:[1,2,4,5]] /= scale[b]
                np.save(saveFile,out)


    modelDetector=None
    if 'state_dict' in checkpointPixel:
        modelPixel = eval(configPixel['arch'])(configPixel['model'])
        modelPixel.load_state_dict(checkpointPixel['state_dict'])
    else:
        modelPixel = checkpoint['model']
    modelPixel.eval()
    modelPixel.summary()

    if gpu is not None:
        modelPixel = modelPixel.to(gpu)
    valid_iter = iter(valid_data_loader)
    validDir = os.path.join(saveDir,'valid_pixel_'+configPixel['name'])
    if not os.path.isdir(validDir):
        os.mkdir(validDir)

    with torch.no_grad():
        for instance in valid_iter:
            imageName = instance['imgName']


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
    parser.add_argument('-b', '--batchsize', default=None, type=int,
                        help='gpu number (default: cpu only)')
    parser.add_argument('-s', '--shuffle', default=False, type=bool,
                        help='shuffle data')
    parser.add_argument('-f', '--config', default=None, type=str,
                        help='config override')

    args = parser.parse_args()

    config = None
    if args.checkpoint is None or args.savedir is None:
        print('Must provide checkpoint (with -c) and save dir (with -d)')
        exit()

    main(args.checkpoint, args.savedir, args.number, args.index, gpu=args.gpu, shuffle=args.shuffle, setBatch=args.batchsize, config=args.config)
