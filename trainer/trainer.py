import numpy as np
import torch
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        #self.log_step = int(np.sqrt(self.batch_size))

    def _to_tensor(self, data, target):
        if type(data) is np.ndarray:
            data, target = torch.FloatTensor(data.astype(np.float32)), torch.FloatTensor(target.astype(np.float32))
        elif type(data) is torch.Tensor:
            data, target = data.type(torch.FloatTensor), target.type(torch.FloatTensor)
        if self.with_cuda:
            data, target = data.to(self.gpu), target.to(self.gpu)
        return data, target

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        output = output.cpu().data.numpy()
        target = target.cpu().data.numpy()
        output = np.argmax(output, axis=1)
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics

    def _train_ipoch(self, iteration):
        """
        Training logic for an ipoch (epoch of user defined size, not dataset defined size)

        :param epoch: Current training epoch.
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

        #total_loss = 0
        #total_metrics = np.zeros(len(self.metrics))
        #total_loss_since_log=0
        #tlsl_count=0
            #for batch_idx, (data, target) in enumerate(self.data_loader):
        #for iteration in range(iteration, iteration+self.ipoch_size):
        batch_idx = (iteration-1) % len(self.data_loader)
        data, target = self._to_tensor(*self.data_loader[batch_idx])
        
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss(output, target)
        loss.backward()
        self.optimizer.step()

            total_loss += loss.item()
            total_loss_since_log += loss.item()
            tlsl_count+=1
            metrics = self._eval_metrics(output, target)
            total_metrics += metrics
            total_metrics_since_log += metrics

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train iteration: {} [{}/{} ({:.0f}%)] Loss: {:.6f} avgLoss: {:.6f}'.format(
                    iteration,
                    batch_idx * self.data_loader.batch_size,
                    len(self.data_loader) * self.data_loader.batch_size,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item(),
                    total_loss_since_log.item()/tlsl_count ))
                ms=''
                for i in range(total_metrics_since_log.shape[0]):
                    ms += '{:.6f}, '.format(total_metrics_since_log[0]/tlsl_count)

                self.logger.info('     avgMetrics: '+ms)
                tlsl_count=0
                total_loss_since_log=0
                total_metric_since_log=0

        log = {
            'loss': total_loss / self.ipoch_size,
            'metrics': (total_metrics / self.ipoch_size).tolist()
        }

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _minor_log(log):
        ls=''
        for key,val in log.items:
            ks +
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
