"""
    Copyright 2019 Brian Davis
    Visual-Template-free-Form-Parsting is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Visual-Template-free-Form-Parsting is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Visual-Template-free-Form-Parsting.  If not, see <https://www.gnu.org/licenses/>.
"""
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


logging.basicConfig(level=logging.INFO, format='')


def main(resume,saveDir,numberOfImages,index,gpu=None, shuffle=False, setBatch=None, config=None, thresh=None, addToConfig=None, test=False,verbose=2):
    np.random.seed(1234)
    torch.manual_seed(1234)
    if resume is not None:
        checkpoint = torch.load(resume, map_location=lambda storage, location: storage)
        print('loaded iteration {}'.format(checkpoint['iteration']))
        if config is None:
            config = checkpoint['config']
        else:
            config = json.load(open(config))
    else:
        checkpoint = None
        config = json.load(open(config))

    if gpu is None:
        config['cuda']=False
    else:
        config['cuda']=True
        config['gpu']=gpu
    if thresh is not None:
        config['THRESH'] = thresh
        if verbose:
            print('Threshold at {}'.format(thresh))
    addDATASET=False
    if addToConfig is not None:
        for add in addToConfig:
            addTo=config
            if verbose:
                printM='added config['
            for i in range(len(add)-2):
                addTo = addTo[add[i]]
                if verbose:
                    printM+=add[i]+']['
            value = add[-1]
            if value=="":
                value=None
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            addTo[add[-2]] = value
            if verbose:
                printM+=add[-2]+']={}'.format(value)
                print(printM)
            if (add[-2]=='useDetections' or add[-2]=='useDetect') and value!='gt':
                addDATASET=True

        
    #config['data_loader']['batch_size']=math.ceil(config['data_loader']['batch_size']/2)
    
    config['data_loader']['shuffle']=shuffle
    #config['data_loader']['rot']=False
    config['validation']['shuffle']=shuffle
    config['data_loader']['eval']=True
    config['validation']['eval']=True
    #config['validation']

    if config['data_loader']['data_set_name']=='FormsDetect':
        config['data_loader']['batch_size']=1
        del config['data_loader']["crop_params"]
        config['data_loader']["rescale_range"]= config['validation']["rescale_range"]

    #print(config['data_loader'])
    if setBatch is not None:
        config['data_loader']['batch_size']=setBatch
        config['validation']['batch_size']=setBatch
    batchSize = config['data_loader']['batch_size']
    if 'batch_size' in config['validation']:
        vBatchSize = config['validation']['batch_size']
    else:
        vBatchSize = batchSize
    if not test:
        data_loader, valid_data_loader = getDataLoader(config,'train')
    else:
        valid_data_loader, data_loader = getDataLoader(config,'test')

    if addDATASET:
        config['DATASET']=valid_data_loader.dataset

    if checkpoint is not None:
        if 'state_dict' in checkpoint:
            model = eval(config['arch'])(config['model'])
            ##DEBUG
            if 'edgeFeaturizerConv.0.0.weight' in checkpoint['state_dict']:
                keys = list(checkpoint['state_dict'].keys())
                for key in keys:
                    if 'edge' in key:
                        newKey = key.replace('edge','rel')
                        checkpoint['state_dict'][newKey] = checkpoint['state_dict'][key]
                        del checkpoint['state_dict'][key]
            ##DEBUG
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = checkpoint['model']
    else:
        model = eval(config['arch'])(config['model'])
    model.eval()
    if verbose:
        model.summary()

    if gpu is not None:
        model = model.to(gpu)
    else:
        model = model.cpu()

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
    if data_loader is not None:
        train_iter = iter(data_loader)
    valid_iter = iter(valid_data_loader)

    with torch.no_grad():

        if index is None:


            if saveDir is not None:
                trainDir = os.path.join(saveDir,'train_'+config['name'])
                validDir = os.path.join(saveDir,'valid_'+config['name'])
                if not os.path.isdir(trainDir):
                    os.mkdir(trainDir)
                if not os.path.isdir(validDir):
                    os.mkdir(validDir)
            else:
                trainDir=None
                validDir=None

            val_metrics_sum = np.zeros(len(metrics))
            val_metrics_list = defaultdict(lambda: defaultdict(list))
            val_comb_metrics = defaultdict(list)

            #if numberOfImages==0:
            #    for i in range(len(valid_data_loader)):
            #        print('valid batch index: {}\{} (not save)'.format(i,len(valid_data_loader)),end='\r')
            #        instance=valid_iter.next()
            #        metricsO,_ = saveFunc(config,instance,model,gpu,metrics)

            #        if type(metricsO) == dict:
            #            for typ,typeLists in metricsO.items():
            #                if type(typeLists) == dict:
            #                    for name,lst in typeLists.items():
            #                        val_metrics_list[typ][name]+=lst
            #                        val_comb_metrics[typ]+=lst
            #                else:
            #                    if type(typeLists) is float or type(typeLists) is int:
            #                        typeLists = [typeLists]
            #                    val_comb_metrics[typ]+=typeLists
            #        else:
            #            val_metrics_sum += metricsO.sum(axis=0)/metricsO.shape[0]
            #else:

            ####
            if 'save_nns' in config:
                nns=[]
            curVI=0

            validName='valid' if not test else 'test'

            for index in range(0,numberOfImages,step*batchSize):
            
                for validIndex in range(index,index+step*vBatchSize, vBatchSize):
                    if validIndex/vBatchSize < len(valid_data_loader):
                        print('{} batch index: {}/{}'.format(validName,validIndex/vBatchSize,len(valid_data_loader)),end='\r')
                        #data, target = valid_iter.next() #valid_data_loader[validIndex]
                        curVI+=1
                        #dataT  = _to_tensor(gpu,data)
                        #output = model(dataT)
                        #data = data.cpu().data.numpy()
                        #output = output.cpu().data.numpy()
                        #target = target.data.numpy()
                        #metricsO = _eval_metrics_ind(metrics,output, target)
                        metricsO,aux = saveFunc(config,valid_iter.next(),model,gpu,metrics,validDir,validIndex)
                        if type(metricsO) == dict:
                            for typ,typeLists in metricsO.items():
                                if type(typeLists) == dict:
                                    for name,lst in typeLists.items():
                                        val_metrics_list[typ][name]+=lst
                                        val_comb_metrics[typ]+=lst
                                else:
                                    if type(typeLists) is float or type(typeLists) is int:
                                        typeLists = [typeLists]
                                    val_comb_metrics[typ]+=typeLists
                        else:
                            val_metrics_sum += metricsO.sum(axis=0)/metricsO.shape[0]
                    
                if not test:
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
                            _,aux=saveFunc(config,train_iter.next(),model,gpu,metrics,trainDir,trainIndex)
                            if 'save_nns' in config:
                                nns+=aux[-1]

            #if gpu is not None or numberOfImages==0:
            try:
                for vi in range(curVI,len(valid_data_loader)):
                    if verbose>1:
                        print('{} batch index: {}\{} (not save)'.format(validName,vi,len(valid_data_loader)),end='\r')
                    instance = valid_iter.next()
                    metricsO,_ = saveFunc(config,instance,model,gpu,metrics)
                    if type(metricsO) == dict:
                        for typ,typeLists in metricsO.items():
                            if type(typeLists) == dict:
                                for name,lst in typeLists.items():
                                    val_metrics_list[typ][name]+=lst
                                    val_comb_metrics[typ]+=lst
                            else:
                                if type(typeLists) is float or type(typeLists) is int:
                                    typeLists = [typeLists]
                                if val_comb_metrics is not None and typeLists is not None:
                                    val_comb_metrics[typ]+=typeLists
                    else:
                        val_metrics_sum += metricsO.sum(axis=0)/metricsO.shape[0]
            except StopIteration:
                print('ERROR: ran out of valid batches early. Expected {} more'.format(len(valid_data_loader)-vi))
            ####
                
            val_metrics_sum /= len(valid_data_loader)
            print('{} metrics'.format(validName))
            for i in range(len(metrics)):
                print(metrics[i].__name__ + ': '+str(val_metrics_sum[i]))
            for typ in val_comb_metrics:
                print('{} overall mean: {}, std {}'.format(typ,np.mean(val_comb_metrics[typ],axis=0), np.std(val_comb_metrics[typ],axis=0)))
                for name, typeLists in val_metrics_list[typ].items():
                    print('{} {} mean: {}, std {}'.format(typ,name,np.mean(typeLists,axis=0),np.std(typeLists,axis=0)))

            if 'save_nns' in config:
                import pickle
                pickle.dump(nns,open(config['save_nns'],'wb'))

        elif type(index)==int:
            if index>0:
                instances = train_iter
            else:
                index*=-1
                instances = valid_iter
            batchIndex = index//batchSize
            inBatchIndex = index%batchSize
            for i in range(batchIndex+1):
                instance= instances.next()
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
        else:
            for instance in data_loader:
                if index in instance['imgName']:
                    break
            if index not in instance['imgName']:
                for instance in valid_data_loader:
                    if index in instance['imgName']:
                        break
            if index in instance['imgName']:
                saveFunc(config,instance,model,gpu,metrics,saveDir,0)
            else:
                print('{} not found! (on {})'.format(index,instance['imgName']))
                print('{} not found! (on {})'.format(index,instance['imgName']))


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Evaluator/Displayer')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--savedir', default=None, type=str,
                        help='path to directory to save result images (default: None)')
    parser.add_argument('-i', '--index', default=None, type=int,
                        help='index on instance to process (default: None)')
    parser.add_argument('-n', '--number', default=0, type=int,
                        help='number of images to save out (from each train and valid) (default: 0)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='gpu number (default: cpu only)')
    parser.add_argument('-b', '--batchsize', default=None, type=int,
                        help='gpu number (default: cpu only)')
    parser.add_argument('-s', '--shuffle', default=False, type=bool,
                        help='shuffle data')
    parser.add_argument('-f', '--config', default=None, type=str,
                        help='config override')
    parser.add_argument('-m', '--imgname', default=None, type=str,
                        help='specify image')
    parser.add_argument('-t', '--thresh', default=None, type=float,
                        help='Confidence threshold for detections')
    parser.add_argument('-a', '--addtoconfig', default=None, type=str,
                        help='Arbitrary key-value pairs to add to config of the form "k1=v1,k2=v2,...kn=vn"')
    parser.add_argument('-T', '--test', default=False, action='store_const', const=True,
                        help='Run test set')
    parser.add_argument('-v', '--verbosity', default=2, type=int,
                        help='How much stuff to print [0,1,2] (default: 2)')

    args = parser.parse_args()

    addtoconfig=[]
    if args.addtoconfig is not None:
        split = args.addtoconfig.split(',')
        for kv in split:
            split2=kv.split('=')
            addtoconfig.append(split2)

    config = None
    if args.checkpoint is None and args.config is None:
        print('Must provide checkpoint (with -c)')
        exit()

    index = args.index
    if args.index is not None and args.imgname is not None:
        print("Cannot index by number and name at same time.")
        exit()
    if args.index is None and args.imgname is not None:
        index = args.imgname
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            main(args.checkpoint, args.savedir, args.number, index, gpu=args.gpu, shuffle=args.shuffle, setBatch=args.batchsize, config=args.config, thresh=args.thresh, addToConfig=addtoconfig,test=args.test,verbose=args.verbosity)
    else:
        main(args.checkpoint, args.savedir, args.number, index, gpu=args.gpu, shuffle=args.shuffle, setBatch=args.batchsize, config=args.config, thresh=args.thresh, addToConfig=addtoconfig,test=args.test,verbose=args.verbosity)
