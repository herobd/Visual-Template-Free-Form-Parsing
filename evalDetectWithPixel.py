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


def save(resumeDetect, resumePixel,saveDir,gpu=None, shuffle=False,configDetect=None, configPixel=None, detector_scale=16):
    np.random.seed(1234)

    if resumeDetect is not None:
        checkpointDetect = torch.load(resumDetect, map_location=lambda storage, location: storage)
        if configDetect is None:
            configDetect = checkpoint['config']
        else:
            configDetect = json.load(open(configDetect))
            

        if 'state_dict' in checkpointDetect:
            modelDetect = eval(configDetect['arch'])(configDetect['model'])
            modelDetect.load_state_dict(checkpointDetect['state_dict'])
        else:
            modelDetect = checkpoint['model']
        modelDetect.eval()
        modelDetect.summary()
        modelDetect.forPairing=True
        detector_scale=modelDetect.scale

        if gpu is not None:
            modelDetect = modelDetect.to(gpu)

        data_loader, valid_data_loader = getDataLoader(configDetect,'train')
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


        resDir = os.path.join(saveDir,'detect_res')
        if not os.path.isdir(resDir):
            os.mkdir(resDir)
        featDir = os.path.join(saveDir,'detect_feats')
        if not os.path.isdir(resDir):
            os.mkdir(resDir)


        with torch.no_grad():
            for instance in valid_iter:
                imageName = instance['imgName']
                scale = instance['scale']
                print('batch: {}'.format(imageName)),end='\r')
                dataT, targetBBsT, targetBBsSizes, targetPointsT, targetPointsSizes, targetPixelsT = __to_tensor(instance,gpu)
                outputBBs, outputOffsets, outputPoints, outputPixels = modelDetector(dataT)
                feats = modelDetector.final_features
                #data = data.cpu().data.numpy()
                maxConf = outputBBs[:,:,0].max().item()
                threshConf = max(maxConf*0.92,0.5)
                outputBBs = non_max_sup_iou(outputBBs.cpu(),threshConf,0.4)
                for b in len(outputBBs):
                    out = outputBBs[b].data.numpy()
                    outfeats = feats[b].cpu().data.numpy()
                    saveFileBB = os.path.join(resDir,'{}'.format(imageName[b]))
                    saveFileFeats = os.path.join(featDir,'{}'.format(imageName[b]))
                    out[:[1,2,4,5]] /= scale[b]
                    np.save(saveFileBB,out)
                    np.save(saveFilFeats,outfeats)


        modelDetector=None

    if resumePixel is not None:
        checkpointPixel = torch.load(resumePixel, map_location=lambda storage, location: storage)
        if configPixel is None:
            configPixel = checkpoint['config']
        else:
            configPixel = json.load(open(configPixel))

        if 'state_dict' in checkpointPixel:
            modelPixel = eval(configPixel['arch'])(configPixel['model'])
            modelPixel.load_state_dict(checkpointPixel['state_dict'])
        else:
            modelPixel = checkpoint['model']
        modelPixel.eval()
        modelPixel.summary()

        if gpu is not None:
            modelPixel = modelPixel.to(gpu)
        data_loader, valid_data_loader = getDataLoader(configPixel,'train')
        valid_iter = iter(valid_data_loader)
        pixelDir = os.path.join(saveDir,'pixel_res')
        if not os.path.isdir(pixelDir):
            os.mkdir(pixelDir)

        with torch.no_grad():
            for data, imageName, target in valid_iter:
                dataT=data.to(gpu)
                saveFileFeats = os.path.join(featDir,'{}'.format(imageName[b]))
                feats = np.load(saveFileFeats)
                padH=(detector_scale-(dataT.size(2)%detector_scale))%detector_scale
                padW=(detector_scale-(dataT.size(3)%detector_scale))%detector_scale
                if padH!=0 or padW!=0:
                     padder = torch.nn.ZeroPad2d((0,padW,0,padH))
                     dataT = padder(dataT)
                output = model(dataT,feats)
                output = output[...,:target.size(-2),:target.size(-1)].cpu()

                for b in len(output):
                    out = output[b].data.numpy()
                    saveFile = os.path.join(pixelDir,'{}'.format(imageName[b]))
                    np.save(saveFile,out)


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Evaluator/Displayer')
    parser.add_argument('-d', '--detector_checkpoint', default=None, type=str,
                        help='path to detector checkpoint (default: None)')
    parser.add_argument('-p', '--pixel_checkpoint', default=None, type=str,
                        help='path to pixel-labelers checkpoint (default: None)')
    parser.add_argument('-o', '--outdir', default=None, type=str,
                        help='path to directory to save results (default: None)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='gpu number (default: cpu only)')
    parser.add_argument('-D', '--detector_config', default=None, type=str,
                        help='detector config override')
    parser.add_argument('-P', '--pixel_config', default=None, type=str,
                        help='pixel config override')

    args = parser.parse_args()

    config = None
    if args.checkpoint is None or args.savedir is None:
        print('Must provide checkpoint (with -c) and save dir (with -d)')
        exit()

    save(args.detector_checkpoint, args.pixel_checkpoint, args.outdir, gpu=args.gpu, detector_config=args.detector_config, pixel_config=args.pixel_config)
