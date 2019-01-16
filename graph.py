import os
import sys
import signal
import json
import logging
import argparse
import torch
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format='')


def graph(log,plot=True,prefix=None):
    graphs=defaultdict(lambda:{'iters':[], 'values':[]})
    for index, entry in log.entries.items():
        iteration = entry['iteration']
        for metric, value in entry.items():
            if metric!='iteration':
                graphs[metric]['iters'].append(iteration)
                graphs[metric]['values'].append(value)
    
    print('summed')
    for metric, data in graphs.items():
        #print('{} max: {}, min {}'.format(metric,max(data['values']),min(data['values'])))
        ndata = np.array(data['values'])
        maxV = ndata.max(axis=0)
        minV = ndata.min(axis=0)
        print('{} max: {}, min {}'.format(metric,maxV,minV))

    if plot:
        import matplotlib.pyplot as plt
        i=1
        for metric, data in graphs.items():
            if (prefix is None and metric[:3]=='avg' or metric[:3]=='val') or (prefix is not None and metric[:len(prefix)]==prefix):
                plt.figure(i)
                i+=1
                plt.plot(data['iters'], data['values'], '.-')
                plt.xlabel('iterations')
                plt.ylabel(metric)
                plt.title(metric)
        plt.show()
    else:
        i=1
        for metric, data in graphs.items():
            if metric[:3]=='avg' or metric[:3]=='val':
                print(metric)
                print(data['values'])





if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='checkpoint file path (default: None)')
    parser.add_argument('-p', '--plot', default=1, type=int,
                        help='plot (default: True)')
    parser.add_argument('-o', '--only', default=None, type=str,
                        help='only stats with this prefix (default: None)')
    parser.add_argument('-e', '--extract', default=None, type=str,
                        help='instead of ploting, save a new file with only the log (default: None)')

    args = parser.parse_args()

    assert args.checkpoint is not None
    saved = torch.load(args.checkpoint,map_location=lambda storage, loc: storage)
    log = saved['logger']
    iteration = saved['iteration']
    print('loaded iteration {}'.format(iteration))
    saved=None

    if args.extract is None:
        graph(log,args.plot,args.only)
    else:
        new_save = {
                'iteration': iteration,
                'logger': log
                }
        new_file = args.extract #args.checkpoint+'.ex'
        torch.save(new_save,new_file)
        print('saved '+new_file)
