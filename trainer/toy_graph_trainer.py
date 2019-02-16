import numpy as np
import torch
from base import BaseTrainer
import timeit
#from datasets.test_random_walk import display
import random


class ToyGraphTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(ToyGraphTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False

        self.adaptLR = config['trainer']['adapt_lr'] if 'adapt_lr' in config else False
        self.adaptLR_base = config['trainer']['adapt_lr_base'] if 'adapt_lr_base' in config else 165 #roughly average number of rels
        self.adaptLR_ep = config['trainer']['adapt_lr_ep'] if 'adapt_lr_ep' in config else 15


    def _to_tensor(self, instance):
        features, adjaceny, gt, num = instance

        if self.with_cuda:
            features = features.float().to(self.gpu)
            adjaceny = adjaceny.to(self.gpu)
            gt = gt.float().to(self.gpu)
        return features, adjaceny, gt, num



    def _train_iteration(self, iteration):
        """
        Training logic for an iteration

        :param iteration: Current training iteration.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        ##tic=timeit.default_timer()
        #batch_idx = (iteration-1) % len(self.data_loader)
        try:
            thisInstance = self.data_loader_iter.next()
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            thisInstance = self.data_loader_iter.next()
        ##toc=timeit.default_timer()
        ##print('data: '+str(toc-tic))
        
        ##tic=timeit.default_timer()

        self.optimizer.zero_grad()

        ##toc=timeit.default_timer()
        ##print('for: '+str(toc-tic))
        #loss = self.loss(output, target)
        index=0
        losses={}
        ##tic=timeit.default_timer()

        features, adj, gt, num = self._to_tensor(thisInstance)
        output,_ = self.model(features,(adj,None),num)

        loss = self.loss(output,gt)
        if self.adaptLR:
            #if we only have a few relationship preds, step smaller so that we don't skew with a bad bias
            #This effects the box loss too so that it doesn't yank the detector/backbone features around
            #we actually just scale the loss, but its all the same :)
            scale = (adj._indicies().size(1)+self.adaptLR_ep)/(self.adaptLR_ep+self.adaptLR_base)
            loss *= scale
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.model.parameters(),1)
        self.optimizer.step()

        loss = loss.item()

        ##toc=timeit.default_timer()
        ##print('bac: '+str(toc-tic))


        log = {
            'loss': loss,
        }

        #if iteration%10==0:
        #image=None
        #queryMask=None
        #targetBoxes=None
        #outputBoxes=None
        #outputOffsets=None
        #loss=None
        #torch.cuda.empty_cache()


        return log#
    def _minor_log(self, log):
        ls=''
        for key,val in log.items():
            ls += key
            if type(val) is float:
                ls +=': {:.6f},\t'.format(val)
            else:
                ls +=': {},\t'.format(val)
        self.logger.info('Train '+ls)

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.logged:
                    print('iter:{} valid batch: {}/{}'.format(self.iteration,batch_idx,len(self.valid_data_loader)), end='\r')

                features, adj, gt,num = self._to_tensor(instance)
                output,_ = self.model(features,(adj,None),num)

                loss = self.loss(output,gt)

                total_loss += loss.item()
                #total_val_metrics += self._eval_metrics(output, target)
        return {
            'loss': total_loss / len(self.valid_data_loader),
        }

