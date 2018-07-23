import os
import math
import json
import timeit
import logging
import torch
import torch.optim as optim
import time
from utils.util import ensure_dir
from collections import defaultdict


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, resume, config, train_logger=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.name = config['name']
        self.iterations = config['trainer']['iterations']
        self.val_step = config['trainer']['val_step']
        self.save_step = config['trainer']['save_step']
        self.log_step = config['trainer']['log_step']
        self.verbosity = config['trainer']['verbosity']
        self.with_cuda = config['cuda'] and torch.cuda.is_available()
        if config['cuda'] and not torch.cuda.is_available():
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        else:
            self.gpu = torch.device('cuda:' + str(config['gpu']))
            self.model = self.model.to(self.gpu)

        self.train_logger = train_logger
        self.optimizer = getattr(optim, config['optimizer_type'])(model.parameters(),
                                                                  **config['optimizer'])
        self.lr_scheduler = getattr(
            optim.lr_scheduler,
            config['lr_scheduler_type'], None)
        if self.lr_scheduler:
            self.lr_scheduler = self.lr_scheduler(self.optimizer, **config['lr_scheduler'])
            self.lr_scheduler_step = config['lr_scheduler_step']
        self.monitor = config['trainer']['monitor']
        self.monitor_mode = config['trainer']['monitor_mode']
        #assert self.monitor_mode == 'min' or self.monitor_mode == 'max'
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
        self.start_iteration = 1
        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], self.name)
        ensure_dir(self.checkpoint_dir)
        json.dump(config, open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)
        if resume:
            self._resume_checkpoint(resume)

    def train(self):
        """
        Full training logic
        """
        sumLog=defaultdict(lambda:0)
        sumTime=0
        #for metric in self.metrics:
        #    sumLog['avg_'+metric.__name__]=0

        for iteration in range(self.start_iteration, self.iterations + 1):
            print('iteration: '+str(iteration), end='\r')

            t = timeit.default_timer()
            result = self._train_iteration(iteration)
            elapsed_time = timeit.default_timer() - t
            sumLog['sec_per_iter'] += elapsed_time
            #print('iter: '+str(elapsed_time))


            for key, value in result.items():
                if key == 'metrics':
                    for i, metric in enumerate(self.metrics):
                        sumLog['avg_'+metric.__name__] += result['metrics'][i]
                else:
                    sumLog['avg_'+key] += value

            if iteration%self.log_step==0 or iteration%self.val_step==0 or iteration % self.save_step == 0:
                log = {'iteration': iteration}

                for key, value in result.items():
                    if key == 'metrics':
                        for i, metric in enumerate(self.metrics):
                            log[metric.__name__] = result['metrics'][i]
                    else:
                        log[key] = value

            if iteration%self.log_step==0:
                print()#clear inplace text
                self._minor_log(log)
                if iteration-self.start_iteration>=self.log_step: #skip avg if started in odd spot
                    for key in sumLog:
                        sumLog[key] /= self.log_step
                    self._minor_log(sumLog)
                for key in sumLog:
                    sumLog[key] =0

            if iteration%self.val_step==0:
                val_result = self._valid_epoch()
                for key, value in val_result.items():
                    if 'metrics' in key:
                        for i, metric in enumerate(self.metrics):
                            log['val_' + metric.__name__] = val_result[key][i]
                    else:
                        log[key] = value
                        #sumLog['avg_'+key] += value

                if self.train_logger is not None:
                    if iteration%self.log_step!=0:
                        print()#clear inplace text
                    self.train_logger.add_entry(log)
                    if self.verbosity >= 1:
                        for key, value in log.items():
                            self.logger.info('    {:15s}: {}'.format(str(key), value))
                if (self.monitor_mode == 'min' and self.monitor in log and log[self.monitor] < self.monitor_best)\
                        or (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
                    self.monitor_best = log[self.monitor]
                    self._save_checkpoint(iteration, log, save_best=True)
            if iteration % self.save_step == 0:
                self._save_checkpoint(iteration, log)
                if iteration%self.log_step!=0:
                    print()#clear inplace text
                self.logger.info('Checkpoint saved for iteration '+str(iteration))
            if self.lr_scheduler and iteration % self.lr_scheduler_step == 0:
                self.lr_scheduler.step(iteration)
                lr = self.lr_scheduler.get_lr()[0]
                if iteration%self.log_step!=0:
                    print()#clear inplace text
                self.logger.info('New Learning Rate: {:.6f}'.format(lr))
            

    def _train_iteration(self, iteration):
        """
        Training logic for a single iteration

        :param iteration: Current iteration number
        """
        raise NotImplementedError

    def _save_checkpoint(self, iteration, log, save_best=False):
        """
        Saving checkpoints

        :param iteration: current iteration number
        :param log: logging information of the ipoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'iteration': iteration,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-iteration{}-loss-{:.4f}.pth.tar'
                                .format(iteration, log['loss']))
        torch.save(state, filename)
        if save_best:
            os.rename(filename, os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))
            self.logger.info("Saving current best: {} ...".format('model_best.pth.tar'))
        else:
            self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_iteration = checkpoint['iteration'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.with_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.gpu)
        self.train_logger = checkpoint['logger']
        if 'override' not in self.config or not self.config['override']:
            self.config = checkpoint['config']
        self.logger.info("Checkpoint '{}' (iteration {}) loaded".format(resume_path, self.start_iteration))
