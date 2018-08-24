import os
import sys
import signal
import json
import logging
import argparse
import torch
from collections import defaultdict
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='')


def graph(log):
    graphs=defaultdict(lambda:{'iters':[], 'values':[]})
    for index, entry in log.entries.items():
        iteration = entry['iteration']
        for metric, value in entry.items():
            if metric!='iteration':
                graphs[metric]['iters'].append(iteration)
                graphs[metric]['values'].append(value)
    
    print('summed')
    i=1
    for metric, data in graphs.items():
        plt.figure(i)
        i+=1
        plt.plot(data['iters'], data['values'], '.-')
        plt.xlabel('iterations')
        plt.ylabel(metric)
        plt.title(metric)
    plt.show()





if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='checkpoint file path (default: None)')

    args = parser.parse_args()

    assert args.checkpoint is not None
    log = torch.load(args.checkpoint,map_location=lambda storage, loc: storage)['logger']
    print('loaded')

    graph(log)
