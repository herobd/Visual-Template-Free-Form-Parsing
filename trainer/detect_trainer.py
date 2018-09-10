import numpy as np
import torch
from base import BaseTrainer
import timeit
from utils import util
from collections import defaultdict


class DetectTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(DetectTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        if 'loss_params' in config:
            self.loss_params=config['loss_params']
        else:
            self.loss_params={}
        if 'loss_weights' in config:
            self.loss_weight=config['loss_weights']
        else:
            self.loss_weight={'line':0.7, 'point':0.5, 'pixel':10}
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        #for i in range(self.start_iteration,
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        #self.log_step = int(np.sqrt(self.batch_size))

    def _to_tensor(self, instance):
        data = instance['img']
        if 'line_gt' in instance:
            targetLines = instance['line_gt']
            targetLines_sizes = instance['line_label_sizes']
        else:
            targetLines = {}
            targetLines_sizes = {}
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
                    new_targets[name] = target.to(self.gpu)
                else:
                    new_targets[name] = None
            return new_targets

        if self.with_cuda:
            data = data.to(self.gpu)
            targetLines=sendToGPU(targetLines)
            targetPoints=sendToGPU(targetPoints)
            if targetPixels is not None:
                targetPixels=targetPixels.to(self.gpu)
        return data, targetLines, targetLines_sizes, targetPoints, targetPoints_sizes, targetPixels

    def _eval_metrics(self, typ,name,output, target):
        if len(self.metrics[typ])>0:
            #acc_metrics = np.zeros(len(self.metrics[typ]))
            met={}
            cpu_output=[]
            for pred in output:
                cpu_output.append(output.cpu().data.numpy())
            target = target.cpu().data.numpy()
            for i, metric in enumerate(self.metrics[typ]):
                met[name+metric.__name__] = metric(cpu_output, target)
            return acc_metrics
        else:
            #return np.zeros(0)
            return {}

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
        batch_idx = (iteration-1) % len(self.data_loader)
        try:
            data, targetLines, targetLines_sizes, targetPoints, targetPoints_sizes, targetPixels= self._to_tensor(self.data_loader_iter.next())
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            data, targetLines, targetLines_sizes, targetPoints, targetPoints_sizes, targetPixels= self._to_tensor(self.data_loader_iter.next())
        ##toc=timeit.default_timer()
        ##print('data: '+str(toc-tic))
        
        ##tic=timeit.default_timer()

        self.optimizer.zero_grad()
        outputLines, outputPoints, outputPixels = self.model(data)

        ##toc=timeit.default_timer()
        ##print('for: '+str(toc-tic))
        #loss = self.loss(output, target)
        loss = 0
        index=0
        losses={}
        ##tic=timeit.default_timer()
        for name, target in targetLines.items():
            #print('line')
            predictions = util.pt_xyrs_2_xyxy(outputLines[index])
            this_loss = self.loss['line'](predictions,target,targetLines_sizes[name], **self.loss_params['line'])
            this_loss*=self.loss_weight['line']
            loss+=this_loss
            losses[name+'_loss']=this_loss.item()
            index+=1
        index=0
        for name, target in targetPoints.items():
            #print('point')
            predictions = outputPoints[index]
            this_loss = self.loss['point'](predictions,target,targetPoints_sizes[name], **self.loss_params['point'])
            this_loss*=self.loss_weight['point']
            loss+=this_loss
            losses[name+'_loss']=this_loss.item()
            index+=1
        if targetPixels is not None:
            #print('pixel')
            this_loss = self.loss['pixel'](outputPixels,targetPixels, **self.loss_params['pixel'])
            this_loss*=self.loss_weight['pixel']
            loss+=this_loss
            losses['pixel_loss']=this_loss.item()
        ##toc=timeit.default_timer()
        ##print('loss: '+str(toc-tic))
        ##tic=timeit.default_timer()
        loss.backward()
        self.optimizer.step()

        ##toc=timeit.default_timer()
        ##print('bac: '+str(toc-tic))

        #tic=timeit.default_timer()
        metrics={}
        #index=0
        #for name, target in targetLines.items():
        #    metrics = {**metrics, **self._eval_metrics('line',name,output, target)}
        #for name, target in targetPoints.items():
        #    metrics = {**metrics, **self._eval_metrics('point',name,output, target)}
        #    metrics = self._eval_metrics(name,output, target)
        #toc=timeit.default_timer()
        #print('metric: '+str(toc-tic))

        ##tic=timeit.default_timer()
        loss = loss.item()
        ##toc=timeit.default_timer()
        ##print('item: '+str(toc-tic))


        log = {
            'loss': loss,
            **metrics,
            **losses
        }


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
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        losses={}
        with torch.no_grad():
            losses = defaultdict(lambda: 0)
            for batch_idx, instance in enumerate(self.valid_data_loader):
                data, targetLines, targetLines_sizes, targetPoints, targetPoints_sizes, targetPixels = self._to_tensor(instance)

                outputLines, outputPoints, outputPixels = self.model(data)
                #loss = self.loss(output, target)
                loss = 0
                index=0
                for name, target in targetLines.items():
                    predictions = util.pt_xyrs_2_xyxy(outputLines[index])
                    this_loss = self.loss['line'](predictions,target,targetLines_sizes[name], **self.loss_params['line'])
                    loss+=this_loss*self.loss_weight['line']
                    losses['val_'+name+'_loss']+=this_loss.item()
                    index+=1
                index=0
                for name, target in targetPoints.items():
                    predictions = outputPoints[index]
                    this_loss = self.loss['point'](predictions,target,targetPoints_sizes[name], **self.loss_params['point'])
                    loss+=this_loss*self.loss_weight['point']
                    losses['val_'+name+'_loss']+=this_loss.item()
                    index+=1
                if targetPixels is not None:
                    this_loss = self.loss['pixel'](outputPixels,targetPixels, **self.loss_params['pixel'])
                    loss+=this_loss*self.loss_weight['pixel']
                    losses['val_pixel_loss']+=this_loss.item()

                total_val_loss += loss.item()
                #total_val_metrics += self._eval_metrics(output, target)
        for name in losses:
            losses[name]/=len(self.valid_data_loader)
        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist(),
            **losses
        }
