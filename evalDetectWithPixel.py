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
from skimage import draw
import cv2

from datasets.forms_detect import FormsDetect
from datasets import forms_detect
from evaluators.formsboxdetect_printer import plotRect

from utils.yolo_tools import non_max_sup_iou, AP_iou

logging.basicConfig(level=logging.INFO, format='')


def save(resumeDetect, resumePixel,saveDir,pair_data_loader,gpu=None, shuffle=False,configDetect=None, configPixel=None, detector_scale=16):
    configPair=None
    np.random.seed(1234)

    if resumeDetect is not None:
        checkpointDetect = torch.load(resumeDetect, map_location=lambda storage, location: storage)
        if configDetect is None:
            configDetect = checkpointDetect['config']
        else:
            configDetect = json.load(open(configDetect))
            

        if 'state_dict' in checkpointDetect:
            modelDetect = eval(configDetect['arch'])(configDetect['model'])
            modelDetect.load_state_dict(checkpointDetect['state_dict'])
        else:
            modelDetect = checkpoint['model']
        modelDetect.eval()
        modelDetect.summary()
        #modelDetect.forPairing=True
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
        if not os.path.isdir(featDir):
            os.mkdir(featDir)


        with torch.no_grad():
            for instance in valid_iter:
                imageName = instance['imgName']
                saveFileBB = os.path.join(resDir,'{}'.format(imageName[0]))
                saveFileFeats = os.path.join(featDir,'{}'.format(imageName[0]))
                if os.path.exists(saveFileBB) and  os.path.exists(saveFileFeats) and not redo:
                    continue
                scale = instance['scale']
                print('batch: {}'.format(imageName),end='\r')
                dataT, targetBBsT, targetBBsSizes, targetPointsT, targetPointsSizes, targetPixelsT = __to_tensor(instance,gpu)
                padH=(detector_scale-(dataT.size(2)%detector_scale))%detector_scale
                padW=(detector_scale-(dataT.size(3)%detector_scale))%detector_scale
                if padH!=0 or padW!=0:
                     padder = torch.nn.ZeroPad2d((0,padW,0,padH))
                     dataT = padder(dataT)
                outputBBs, outputOffsets, outputPoints, outputPixels = modelDetect(dataT)
                feats = modelDetect.final_features
                #data = data.cpu().data.numpy()
                maxConf = outputBBs[:,:,0].max().item()
                threshConf = max(maxConf*0.92,0.5)
                outputBBs = non_max_sup_iou(outputBBs.cpu(),threshConf,0.4)
                for b in range(len(outputBBs)):
                    out = outputBBs[b].data.numpy()
                    outfeats = feats[b].cpu().data.numpy()
                    saveFileBB = os.path.join(resDir,'{}'.format(imageName[b]))
                    saveFileFeats = os.path.join(featDir,'{}'.format(imageName[b]))
                    out[:,[1,2,4,5]] /= scale[b]
                    np.save(saveFileBB,out)
                    np.save(saveFileFeats,outfeats)


        modelDetect=None

    if resumePixel is not None:
        featDir = os.path.join(saveDir,'detect_feats')
        checkpointPixel = torch.load(resumePixel, map_location=lambda storage, location: storage)
        if configPixel is None:
            configPixel = checkpointPixel['config']
        else:
            configPixel = json.load(open(configPixel))

        if 'state_dict' in checkpointPixel:
            modelPixel = eval(configPixel['arch'])(configPixel['model'])
            modelPixel.load_state_dict(checkpointPixel['state_dict'])
        else:
            modelPixel = checkpoint['model']
        modelPixel.eval()
        modelPixel.summary()
        configPair=configPixel

        if gpu is not None:
            modelPixel = modelPixel.to(gpu)
        #data_loader, valid_data_loader = getDataLoader(configPixel,'train')
        #valid_iter = iter(valid_data_loader)
        pixelDir = os.path.join(saveDir,'pixel_res')
        if not os.path.isdir(pixelDir):
            os.mkdir(pixelDir)

        with torch.no_grad():
            #for data, imageName, target in valid_iter:
            i=0
            for instance in iter(pair_data_loader):
                imageName = instance['imgName'][0]
                saveFile = os.path.join(pixelDir,'{}-{}'.format(i,imageName))
                if os.path.exists(saveFile) and not redo:
                    continue
                print('batch: {}, {}/{}'.format(imageName,i,len(pair_data_loader)),end='\r')
                data = torch.cat([instance['img'],instance['queryMask']],dim=1)
                dataT=data.to(gpu)
                padH=(detector_scale-(dataT.size(2)%detector_scale))%detector_scale
                padW=(detector_scale-(dataT.size(3)%detector_scale))%detector_scale
                if padH!=0 or padW!=0:
                     padder = torch.nn.ZeroPad2d((0,padW,0,padH))
                     dataT = padder(dataT)
                
                saveFileFeats = os.path.join(featDir,'{}.npy'.format(imageName))
                feats = torch.from_numpy(np.load(saveFileFeats))[None,...].to(gpu)
                #if feats.size(-2)*detector_scale//2 < dataT.size(-2)//2:
                dataT =  dataT[...,:feats.size(-2)*detector_scale,:feats.size(-1)*detector_scale]
                output = modelPixel(dataT,feats)
                #output = output[...,:target.size(-2),:target.size(-1)].cpu()
                output = output[...,:data.size(-2)//2,:data.size(-1)//2].cpu()

                #for b in len(output):
                out = output[0].data.numpy()
                np.save(saveFile,out)
                i+=1
    return configPair

def evaluate(saveDir,pair_data_loader,draw_num=100):
    vote_thresh=0.2
    fill_thresh=0.3
    pix_thresh=0
    min_votes=5
    detectDir = os.path.join(saveDir,'detect_res')
    pixelDir = os.path.join(saveDir,'pixel_res')
    
    #data_loader, valid_data_loader = getDataLoader(configBoxPair,'train')
    aps=[]
    recalls=[]
    precisions=[]
    prevImageName=None
    if draw_num>0:
        draw_every = int(len(pair_data_loader)/draw_num)
    else:
        draw_every = None
    i=0
    for instance in iter(pair_data_loader):
        imageName = instance['imgName'][0]
        print('batch: {}, {}/{}'.format(imageName,i,len(pair_data_loader)),end='\r')
        bbs = instance['responseBBs']
        if bbs is not None:
            bbs = bbs.numpy()[0]
            bbs *= 0.5 #to account for pixel labeler predicting at half scale

        if imageName != prevImageName:
            detections = np.load(os.path.join(detectDir,'{}.npy'.format(imageName)))
            detections *= 0.5 * instance['scale'][0]
            prevImageName=imageName
        pixels = np.load(os.path.join(pixelDir,'{}-{}.npy'.format(i,imageName)))

        #threshold pixels
        pixels = pixels>pix_thresh
        total_votes= pixels.sum()

        selectedIdxs=[]
        if total_votes>min_votes:
            for bbIdx in range(detections.shape[0]):
                bb=detections[bbIdx]
                xc = bb[1]
                yc = bb[2]
                rot = bb[3]
                h = bb[4]
                w = bb[5]
                assert(rot==0)
                area=h*w
                tlX = xc-w
                tlY = yc-h
                trX = xc+w
                trY = yc-h
                blX = xc-w
                blY = yc+h
                brX = xc+w
                brY = yc+h

                rr,cc=draw.polygon([tlY,trY,brY,blY], [tlX,trX,brX,blX], pixels.shape)
                votes = pixels[rr,cc].sum()
                if (votes/float(total_votes) > vote_thresh and votes>min_votes) or votes/area>fill_thresh:
                    selectedIdxs.append(bbIdx)
        final_pred = detections[selectedIdxs]
        if bbs is not None:
            ap, precision, recall = AP_iou(torch.from_numpy(bbs),torch.from_numpy(final_pred),0.5,ignoreClasses=True)
            ap=ap[0]
            precision=precision[0]
            recall=recall[0]
        elif final_pred.shape[0]==0:
            ap=1
            precision=1
            recall=1
        else:
            ap=0
            precision=0
            recall=1
        aps.append(ap)
        precisions.append(precision)
        recalls.append(recall)

        if  draw_every is not None and i%draw_every==0:
            image = (1-np.moveaxis(instance['img'].numpy()[0],0,2))/2.0
            queryMask = instance['queryMask'].numpy()[0,0]
            #bbs *= 2
            #final_pred *= 2
            
            if image.shape[2]==1:
                image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            image[:,:,1] *= 1-queryMask
            invPix = 1-pixels
            invPix = cv2.resize(invPix.astype(np.float),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
            image[:,:,0] *= invPix
            if bbs is not None:
                for j in range(bbs.shape[0]):
                    plotRect(image,(0,1,0),bbs[j]*2)
                    nothin=None
            if final_pred.shape[0]>0:
                maxConf = final_pred[:,0].max()
                for j in range(final_pred.shape[0]):
                    conf = final_pred[j,0]
                    shade = 0.0+(conf-maxConf/2)/(maxConf-maxConf/2)
                    if final_pred[j,6] > final_pred[j,7]:
                        color=(0,0,shade) #textF
                    else:
                        color=(shade,0,0) #field
                    plotRect(image,color,final_pred[j,1:6]*2)

            saveName='{}-{}_p={:.2f}_r={:.2f}.png'.format(i,imageName,precision,recall)
            cv2.imwrite(os.path.join(saveDir,saveName),image*255)
            #print('saved '+os.path.join(saveDir,saveName))
        i+=1

    print('precision mean: {}, std: {}'.format(np.mean(precisions),np.std(precisions)))
    print('recall mean: {}, std: {}'.format(np.mean(recalls),np.std(recalls)))

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
    parser.add_argument('-n', '--number', default=100, type=int,
                        help='number of images to draw (default: 100)')

    args = parser.parse_args()

    config = None
    if args.outdir is None:
        print('Must provide out dir (with -o)')
        exit()

    #configPair=None
    configPair={
    "data_loader": {
        "data_set_name": "FormsBoxPair",
        "data_dir": "../data/forms",
        "batch_size": 2,
        "shuffle": False,
        "num_workers": 2,
        "crop_to_page":False,
        "color":False,
        "rescale_range": [0.4,0.65],
        "crop_params": {
            "crop_size":1024
        },
        "no_blanks": True,
        "swap_circle":True,
        "no_graphics":True,
        "cache_resized_images": True,
        "rotation": False,
        "use_dist_mask": True,
        "use_vdist_mask": True,
        "use_hdist_mask": True


    },
    "validation": {
        "shuffle": False,
        "crop_to_page":False,
        "color":False,
        "rescale_range": [0.52,0.52],
        "no_blanks": True,
        "swap_circle":True,
        "no_graphics":True,
        "batch_size": 1,
        "rotation": False,
        "use_dist_mask": True,
        "use_vdist_mask": True,
        "use_hdist_mask": True
    }
    }
    _train, pair_data_loader = getDataLoader(configPair,'train')

    if args.detector_checkpoint is not None or args.pixel_checkpoint is not None:
        configPair = save(args.detector_checkpoint, args.pixel_checkpoint, args.outdir,pair_data_loader, gpu=args.gpu, configDetect=args.detector_config, configPixel=args.pixel_config)
    #if configPair is None:
    #    configPair=configPair_maybe
    evaluate(args.outdir,pair_data_loader,draw_num=args.number)
