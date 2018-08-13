import numpy as np
import torch
from base import BaseTrainer
import timeit
from utils import util


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
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        #for i in range(self.start_iteration,
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        #self.log_step = int(np.sqrt(self.batch_size))

    def _to_tensor(self, instance):
        data = instance['img']
        targets = instance['sol_eol_gt']
        target_sizes = instance['label_sizes']
        if type(data) is np.ndarray:
            data = torch.FloatTensor(data.astype(np.float32))
        elif type(data) is torch.Tensor:
            data = data.type(torch.FloatTensor)
        if self.with_cuda:
            data = data.to(self.gpu)
            new_targets={}
            for name, target in targets.items():
                if target is not None:
                    new_targets[name] = target.to(self.gpu)
                else:
                    new_targets[name] = None
            return data, new_targets, target_sizes
        return data, targets, target_sizes

    def _eval_metrics(self, output, target):
        if len(self.metrics)>0:
            acc_metrics = np.zeros(len(self.metrics))
            cpu_output=[]
            for pred in output:
                cpu_output.append(output.cpu().data.numpy())
            target = target.cpu().data.numpy()
            for i, metric in enumerate(self.metrics):
                acc_metrics[i] += metric(cpu_output, target)
            return acc_metrics
        else:
            return np.zeros(0)

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
            data, targets, target_sizes = self._to_tensor(self.data_loader_iter.next())
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            data, targets, target_sizes = self._to_tensor(self.data_loader_iter.next())
        ##toc=timeit.default_timer()
        ##print('data: '+str(toc-tic))
        
        ##tic=timeit.default_timer()

        self.optimizer.zero_grad()
        output = self.model(data)

        ##toc=timeit.default_timer()
        ##print('for: '+str(toc-tic))
        #loss = self.loss(output, target)
        loss = 0
        index=0
        losses={}
        ##tic=timeit.default_timer()
        for name, target in targets.items():
            predictions = util.pt_xyrs_2_xyxy(output[index])
            this_loss = self.loss(predictions,target,target_sizes[name], **self.loss_params)
            loss+=this_loss
            losses[name+'_loss']=this_loss.item()
            index+=1
        ##toc=timeit.default_timer()
        ##print('loss: '+str(toc-tic))
        ##tic=timeit.default_timer()
        loss.backward()
        self.optimizer.step()

        ##toc=timeit.default_timer()
        ##print('bac: '+str(toc-tic))

        ##tic=timeit.default_timer()
        metrics = self._eval_metrics(output, target)
        ##toc=timeit.default_timer()
        ##print('metric: '+str(toc-tic))

        ##tic=timeit.default_timer()
        loss = loss.item()
        ##toc=timeit.default_timer()
        ##print('item: '+str(toc-tic))


        log = {
            'loss': loss,
            'metrics': metrics,
            **losses
        }


        return log

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
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = self._to_tensor(data, target)

                output = self.model(data)
                loss = self.loss(output, target)

                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
