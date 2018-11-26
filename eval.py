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


def main(resume,saveDir,numberOfImages,index,gpu=None, shuffle=False, setBatch=None, config=None):
    np.random.seed(1234)
    torch.manual_seed(1234)
    #if gpu is None:
    #    loc = 
    checkpoint = torch.load(resume, map_location=lambda storage, location: storage)
    if config is None:
        config = checkpoint['config']
    else:
        config = json.load(open(config))
        
    #config['data_loader']['batch_size']=math.ceil(config['data_loader']['batch_size']/2)
    
    config['data_loader']['shuffle']=shuffle
    #config['data_loader']['rot']=False
    config['validation']['shuffle']=shuffle
    #config['validation']

    if config['data_loader']['data_set_name']=='FormsDetect':
        config['data_loader']['batch_size']=1
        del config['data_loader']["crop_params"]
        config['data_loader']["rescale_range"]= config['validation']["rescale_range"]

    #print(config['data_loader'])
    if setBatch is not None:
        #config['data_loader']['batch_size']=setBatch
        config['validation']['batch_size']=setBatch
    batchSize = config['data_loader']['batch_size']
    if 'batch_size' in config['validation']:
        vBatchSize = config['validation']['batch_size']
    else:
        vBatchSize = batchSize
    data_loader, valid_data_loader = getDataLoader(config,'train')
    #ttt=FormsDetect(dirPath='/home/ubuntu/brian/data/forms',split='train',config={'crop_to_page':False,'rescale_range':[450,800],'crop_params':{"crop_size":512},'no_blanks':True, "only_types": ["text_start_gt"], 'cache_resized_images': True})
    #data_loader = torch.utils.data.DataLoader(ttt, batch_size=16, shuffle=False, num_workers=5, collate_fn=forms_detect.collate)
    #valid_data_loader = data_loader.split_validation()

    if 'state_dict' in checkpoint:
        model = eval(config['arch'])(config['model'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = checkpoint['model']
    model.eval()
    model.summary()

    if gpu is not None:
        model = model.to(gpu)

    metrics = [eval(metric) for metric in config['metrics']]


    #if "class" in config["trainer"]:
    #    trainer_class = config["trainer"]["class"]
    #else:
    #    trainer_class = "Trainer"

    #saveFunc = eval(trainer_class+'_printer')
    saveFunc = eval(config['data_loader']['data_set_name']+'_printer')

    step=5

    #numberOfImages = numberOfImages//config['data_loader']['batch_size']
    #print(len(data_loader))
    train_iter = iter(data_loader)
    valid_iter = iter(valid_data_loader)

    with torch.no_grad():

        if index is None:


            trainDir = os.path.join(saveDir,'train_'+config['name'])
            validDir = os.path.join(saveDir,'valid_'+config['name'])
            if not os.path.isdir(trainDir):
                os.mkdir(trainDir)
            if not os.path.isdir(validDir):
                os.mkdir(validDir)

            val_metrics_sum = np.zeros(len(metrics))
            val_metrics_list = defaultdict(lambda: defaultdict(list))
            val_comb_metrics = defaultdict(list)

            if numberOfImages==0:
                for i in range(len(valid_data_loader)):
                    print('valid batch index: {}\{} (not save)'.format(i,len(valid_data_loader)),end='\r')
                    instance=valid_iter.next()
                    metricsO,_ = saveFunc(config,instance,model,gpu,metrics)

                    if type(metricsO) == dict:
                        for typ,typeLists in metricsO.items():
                            if type(typeLists) == dict:
                                for name,lst in typeLists.items():
                                    val_metrics_list[typ][name]+=lst
                                    val_comb_metrics[typ]+=lst
                            else:
                                val_comb_metrics[typ]+=typeLists
                    else:
                        val_metrics_sum += metricsO.sum(axis=0)/metricsO.shape[0]
            else:
                curVI=0

                for index in range(0,numberOfImages,step*batchSize):
                    for trainIndex in range(index,index+step*batchSize, batchSize):
                        if trainIndex/batchSize < len(data_loader):
                            print('train batch index: {}/{}'.format(trainIndex/batchSize,len(data_loader)),end='\r')
                            #data, target = train_iter.next() #data_loader[trainIndex]
                            #dataT = _to_tensor(gpu,data)
                            #output = model(dataT)
                            #data = data.cpu().data.numpy()
                            #output = output.cpu().data.numpy()
                            #target = target.data.numpy()
                            #metricsO = _eval_metrics_ind(metrics,output, target)
                            saveFunc(config,train_iter.next(),model,gpu,metrics,trainDir,trainIndex)
                    
                    for validIndex in range(index,index+step*vBatchSize, vBatchSize):
                        if validIndex/vBatchSize < len(valid_data_loader):
                            print('valid batch index: {}/{}'.format(validIndex/vBatchSize,len(valid_data_loader)),end='\r')
                            #data, target = valid_iter.next() #valid_data_loader[validIndex]
                            curVI+=1
                            #dataT  = _to_tensor(gpu,data)
                            #output = model(dataT)
                            #data = data.cpu().data.numpy()
                            #output = output.cpu().data.numpy()
                            #target = target.data.numpy()
                            #metricsO = _eval_metrics_ind(metrics,output, target)
                            metricsO,_ = saveFunc(config,valid_iter.next(),model,gpu,metrics,validDir,validIndex)
                            if type(metricsO) == dict:
                                for typ,typeLists in metricsO.items():
                                    if type(typeLists) == dict:
                                        for name,lst in typeLists.items():
                                            val_metrics_list[typ][name]+=lst
                                            val_comb_metrics[typ]+=lst
                                    else:
                                        val_comb_metrics[typ]+=typeLists
                            else:
                                val_metrics_sum += metricsO.sum(axis=0)/metricsO.shape[0]
                            
                if gpu is not None:
                    try:
                        for vi in range(curVI,len(valid_data_loader)):
                            print('valid batch index: {}\{} (not save)'.format(vi,len(valid_data_loader)),end='\r')
                            instance = valid_iter.next()
                            metricsO,_ = saveFunc(config,instance,model,gpu,metrics)
                            if type(metricsO) == dict:
                                for typ,typeLists in metricsO.items():
                                    if type(typeLists) == dict:
                                        for name,lst in typeLists.items():
                                            val_metrics_list[typ][name]+=lst
                                            val_comb_metrics[typ]+=lst
                                    else:
                                        val_comb_metrics[typ]+=typeLists
                            else:
                                val_metrics_sum += metricsO.sum(axis=0)/metricsO.shape[0]
                    except StopIteration:
                        print('ERROR: ran out of valid batches early. Expected {} more'.format(len(valid_data_loader)-vi))
                
            val_metrics_sum /= len(valid_data_loader)
            print('Validation metrics')
            for i in range(len(metrics)):
                print(metrics[i].__name__ + ': '+str(val_metrics_sum[i]))
            for typ in val_comb_metrics:
                print('{} overall mean: {}, std {}'.format(typ,np.mean(val_comb_metrics[typ],axis=0), np.std(val_comb_metrics[typ],axis=0)))
                for name, typeLists in val_metrics_list[typ].items():
                    print('{} {} mean: {}, std {}'.format(typ,name,np.mean(typeLists,axis=0),np.std(typeLists,axis=0)))

        else:
            batchIndex = index//batchSize
            inBatchIndex = index%batchSize
            for i in range(batchIndex+1):
                instance= train_iter.next()
            #data, target = data[inBatchIndex:inBatchIndex+1], target[inBatchIndex:inBatchIndex+1]
            #dataT = _to_tensor(gpu,data)
            #output = model(dataT)
            #data = data.cpu().data.numpy()
            #output = output.cpu().data.numpy()
            #target = target.data.numpy()
            #print (output.shape)
            #print ((output.min(), output.amin()))
            #print (target.shape)
            #print ((target.amin(), target.amin()))
            #metricsO = _eval_metrics_ind(metrics,output, target)
            saveFunc(config,instance,model,gpu,metrics,saveDir,batchIndex*batchSize)


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
