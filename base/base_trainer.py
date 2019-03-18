import os
import math
import json, copy
import timeit
import logging
import torch
import torch.optim as optim
import time
from utils.util import ensure_dir
from collections import defaultdict
from model import *
#from ..model import PairingGraph

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
        self.logged = config['super_computer'] if 'super_computer' in config else False
        self.iterations = config['trainer']['iterations']
        self.val_step = config['trainer']['val_step']
        self.save_step = config['trainer']['save_step']
        self.save_step_minor = config['trainer']['save_step_minor'] if 'save_step_minor' in config['trainer'] else None
        self.log_step = config['trainer']['log_step']
        self.verbosity = config['trainer']['verbosity']
        self.with_cuda = config['cuda'] and torch.cuda.is_available()
        if config['cuda'] and not torch.cuda.is_available():
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        elif config['cuda']:
            self.gpu = torch.device('cuda:' + str(config['gpu']))
            self.model = self.model.to(self.gpu)
        else:
            self.gpu=None

        self.train_logger = train_logger
        if config['optimizer_type']!="none":
            self.optimizer = getattr(optim, config['optimizer_type'])(model.parameters(),
                                                                      **config['optimizer'])
        self.useLearningSchedule = config['trainer']['use_learning_schedule'] if 'use_learning_schedule' in config['trainer'] else False
        if self.useLearningSchedule=='LR_test':
            start_lr=0.000001
            slope = (1-start_lr)/self.iterations
            lr_lambda = lambda step_num: start_lr + slope*step_num
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda)
        elif self.useLearningSchedule=='cyclic':
            min_lr_mul = config['trainer']['min_lr_mul'] if 'min_lr_mul' in config['trainer'] else 0.001
            cycle_size = config['trainer']['cycle_size'] if 'cycle_size' in config['trainer'] else 500
            lr_lambda = lambda step_num: (1-(1-min_lr_mul)*((step_num-1)%cycle_size)/(cycle_size-1))
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda)
        elif self.useLearningSchedule=='cyclic-full':
            min_lr_mul = config['trainer']['min_lr_mul'] if 'min_lr_mul' in config['trainer'] else 0.25
            cycle_size = config['trainer']['cycle_size'] if 'cycle_size' in config['trainer'] else 500
            def trueCycle (step_num):
                cycle_num = step_num//cycle_size
                if cycle_num%2==0: #even, rising
                    return ((1-min_lr_mul)*((step_num)%cycle_size)/(cycle_size-1)) + min_lr_mul
                else: #odd
                    return (1-(1-min_lr_mul)*((step_num)%cycle_size)/(cycle_size-1))
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,trueCycle)
        elif self.useLearningSchedule=='1cycle':
            low_lr_mul = config['trainer']['low_lr_mul'] if 'low_lr_mul' in config['trainer'] else 0.25
            min_lr_mul = config['trainer']['min_lr_mul'] if 'min_lr_mul' in config['trainer'] else 0.0001
            cycle_size = config['trainer']['cycle_size'] if 'cycle_size' in config['trainer'] else 1000
            iters_in_trailoff = self.iterations-(2*cycle_size)
            def oneCycle (step_num):
                cycle_num = step_num//cycle_size
                if step_num<cycle_size: #rising
                    return ((1-low_lr_mul)*((step_num)%cycle_size)/(cycle_size-1)) + low_lr_mul
                elif step_num<cycle_size*2: #falling
                    return (1-(1-low_lr_mul)*((step_num)%cycle_size)/(cycle_size-1))
                else: #trail off
                    t_step_num = step_num-(2*cycle_size)
                    return low_lr_mul*(iters_in_trailoff-t_step_num)/iters_in_trailoff + min_lr_mul*t_step_num/iters_in_trailoff

            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,oneCycle)
        elif self.useLearningSchedule=='detector':
            warmup_steps = config['trainer']['warmup_steps'] if 'warmup_steps' in config['trainer'] else 1000
            lr_lambda = lambda step_num: min((step_num+1)**-0.3, (step_num+1)*warmup_steps**-1.3)
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda)
        elif self.useLearningSchedule is True:
            warmup_steps = config['trainer']['warmup_steps'] if 'warmup_steps' in config['trainer'] else 1000
            #lr_lambda = lambda step_num: min((step_num+1)**-0.3, (step_num+1)*warmup_steps**-1.3)
            lr_lambda = lambda step_num: min((max(0.000001,step_num-(warmup_steps-3))/100)**-0.1, step_num*(1.485/warmup_steps)+.01)
            #y=((x-(2000-3))/100)^-0.1 and y=x*(1.485/2000)+0.01
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda)
        elif self.useLearningSchedule:
            print('Unrecognized learning schedule: {}'.format(self.useLearningSchedule))
            exit()
        
        self.monitor = config['trainer']['monitor']
        self.monitor_mode = config['trainer']['monitor_mode']
        #assert self.monitor_mode == 'min' or self.monitor_mode == 'max'
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
        self.retry_count = config['trainer']['retry_count'] if 'retry_count' in config['trainer'] else 1
        self.start_iteration = 1
        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], self.name)
        ensure_dir(self.checkpoint_dir)
        json.dump(config, open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)
        self.swa = config['trainer']['swa'] if 'swa' in config['trainer'] else (config['trainer']['weight_averaging'] if 'weight_averaging' in config['trainer'] else False)
        if self.swa:
            self.swa_model = type(self.model)(config['model'])
            if config['cuda']:
                self.swa_model = self.swa_model.to(self.gpu)
            self.swa_start = config['trainer']['swa_start'] if 'swa_start' in config['trainer'] else config['trainer']['weight_averaging_start']
            self.swa_c_iters = config['trainer']['swa_c_iters'] if 'swa_c_iters' in config['trainer'] else config['trainer']['weight_averaging_c_iters']
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

        for self.iteration in range(self.start_iteration, self.iterations + 1):
            if not self.logged:
                print('iteration: {}'.format(self.iteration), end='\r')

            t = timeit.default_timer()
            result=None
            lastErr=None
            if self.useLearningSchedule:
                self.lr_schedule.step()
            for attempt in range(self.retry_count):
                try:
                    result = self._train_iteration(self.iteration)
                    break
                except RuntimeError as err:
                    torch.cuda.empty_cache() #this is primarily to catch rare CUDA out of memory errors
                    lastErr = err
            if result is None:
                if self.retry_count>1:
                    print('Failed all {} times!'.format(self.retry_count))
                raise lastErr

            elapsed_time = timeit.default_timer() - t
            sumLog['sec_per_iter'] += elapsed_time
            #print('iter: '+str(elapsed_time))

            #Stochastic Weight Averaging    https://github.com/timgaripov/swa/blob/master/train.py
            if self.swa and self.iteration>=self.swa_start and (self.iterations-self.swa_start)%self.swa_c_iters==0:
                swa_n = (self.iterations-self.swa_start)//self.swa_c_iters
                moving_average(self.swa_model, self.model, 1.0 / (swa_n + 1))
                #swa_n += 1


            for key, value in result.items():
                if key == 'metrics':
                    for i, metric in enumerate(self.metrics):
                        sumLog['avg_'+metric.__name__] += result['metrics'][i]
                else:
                    sumLog['avg_'+key] += value
            
            #log prep
            if (    self.iteration%self.log_step==0 or 
                    self.iteration%self.val_step==0 or 
                    self.iteration % self.save_step == 0 or 
                    (self.save_step_minor is not None and self.iteration % self.save_step_minor)
                ):
                log = {'iteration': self.iteration}

                for key, value in result.items():
                    if key == 'metrics':
                        for i, metric in enumerate(self.metrics):
                            log[metric.__name__] = result['metrics'][i]
                    else:
                        log[key] = value

            #LOG
            if self.iteration%self.log_step==0:
                #prinpt()#clear inplace text
                print('                   ', end='\r')
                if self.iteration-self.start_iteration>=self.log_step: #skip avg if started in odd spot
                    for key in sumLog:
                        sumLog[key] /= self.log_step
                    #self._minor_log(sumLog)
                    log = {**log, **sumLog}
                self._minor_log(log)
                for key in sumLog:
                    sumLog[key] =0
                if self.iteration%self.val_step!=0: #we'll do it later if we have a validation pass
                    self.train_logger.add_entry(log)

            #VALIDATION
            if self.iteration%self.val_step==0:
                val_result = self._valid_epoch()
                for key, value in val_result.items():
                    if 'metrics' in key:
                        for i, metric in enumerate(self.metrics):
                            log['val_' + metric.__name__] = val_result[key][i]
                    else:
                        log[key] = value
                        #sumLog['avg_'+key] += value
                if self.swa and self.iteration>=self.swa_start:
                    temp_model = self.model
                    self.model = self.swa_model
                    val_result = self._valid_epoch()
                    self.model = temp_model
                    for key, value in val_result.items():
                        if 'metrics' in key:
                            for i, metric in enumerate(self.metrics):
                                log['swa_val_' + metric.__name__] = val_result[key][i]
                        else:
                            log['swa_'+key] = value

                if self.train_logger is not None:
                    if self.iteration%self.log_step!=0:
                        print('                   ', end='\r')
                    #    print()#clear inplace text
                    self.train_logger.add_entry(log)
                    if self.verbosity >= 1:
                        for key, value in log.items():
                            if self.verbosity>=2 or 'avg' in key or 'val' in key:
                                self.logger.info('    {:15s}: {}'.format(str(key), value))
                if (self.monitor_mode == 'min' and self.monitor in log and log[self.monitor] < self.monitor_best)\
                        or (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
                    self.monitor_best = log[self.monitor]
                    self._save_checkpoint(self.iteration, log, save_best=True)

            #SAVE
            if self.iteration % self.save_step == 0:
                self._save_checkpoint(self.iteration, log)
                if self.iteration%self.log_step!=0:
                    print('                   ', end='\r')
                #    print()#clear inplace text
                self.logger.info('Checkpoint saved for iteration '+str(self.iteration))
            elif self.iteration % self.save_step_minor == 0:
                self._save_checkpoint(self.iteration, log, minor=True)
                if self.iteration%self.log_step!=0:
                    print('                   ', end='\r')
                #    print()#clear inplace text
                #self.logger.info('Minor checkpoint saved for iteration '+str(self.iteration))

            

    def _train_iteration(self, iteration):
        """
        Training logic for a single iteration

        :param iteration: Current iteration number
        """
        raise NotImplementedError

    def save(self):
        self._save_checkpoint(self.iteration, None)

    def _save_checkpoint(self, iteration, log, save_best=False, minor=False):
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
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        if 'save_mode' not in self.config or self.config['save_mode']=='state_dict':
            state_dict = self.model.state_dict()
            for k,v in state_dict.items():
                state_dict[k]=v.cpu()
            state['state_dict']= state_dict
            if self.swa:
                swa_state_dict = self.swa_model.state_dict()
                for k,v in swa_state_dict.items():
                    swa_state_dict[k]=v.cpu()
                state['swa_state_dict']= swa_state_dict
        else:
            state['model'] = self.model.cpu()
            if self.swa:
                state['swa_model'] = self.swa_model.cpu()
        #if self.swa:
        #    state['swa_n']=self.swa_n
        torch.cuda.empty_cache() #weird gpu memory issue when calling torch.save()
        if not minor:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint-iteration{}.pth.tar'
                                    .format(iteration))
        else:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint-latest.pth.tar')
                            
        #print(self.module.state_dict().keys())
        torch.save(state, filename)
        if not minor:
            #remove minor as this is the latest
            filename_late = os.path.join(self.checkpoint_dir, 'checkpoint-latest.pth.tar')
            try:
                os.remove(filename_late)
            except FileNotFoundError:
                pass
            #os.link(filename,filename_late) #this way checkpoint-latest always does have the latest
            torch.save(state, filename_late) #something is wrong with thel inkgin

        if save_best:
            os.rename(filename, os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))
            self.logger.info("Saved current best: {} ...".format('model_best.pth.tar'))
        else:
            self.logger.info("Saved checkpoint: {} ...".format(filename))


        ######DEBUG
        #checkpoint = torch.load(filename)
        #model_dict=self.model.state_dict()
        #for name in checkpoint['state_dict']:
            #if (checkpoint['state_dict'][name]!=model_dict[name]).any():
                #        print('state not equal at: '+name)
        #        import pdb; pdb.set_trace()

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        if 'override' not in self.config or not self.config['override']:
            self.config = checkpoint['config']
        self.start_iteration = checkpoint['iteration'] + 1
        self.monitor_best = checkpoint['monitor_best']
        #print(checkpoint['state_dict'].keys())
        if ('save_mode' not in self.config or self.config['save_mode']=='state_dict') and 'state_dict' in checkpoint:
            ##DEBUG
            if 'edgeFeaturizerConv.0.0.weight' in checkpoint['state_dict']:
                keys = list(checkpoint['state_dict'].keys())
                for key in keys:
                    if 'edge' in key:
                        newKey = key.replace('edge','rel')
                        checkpoint['state_dict'][newKey] = checkpoint['state_dict'][key]
                        del checkpoint['state_dict'][key]
            ##DEBUG

            self.model.load_state_dict(checkpoint['state_dict'])
            if self.swa:
                self.swa_model.load_state_dict(checkpoint['swa_state_dict'])
        else:
            self.model = checkpoint['model']
            if self.swa:
                self.swa_model = checkpoint['swa_model']
        #if self.swa:
        #    self.swa_n = checkpoint['swa_n']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.with_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.gpu)
        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (iteration {}) loaded".format(resume_path, self.start_iteration))

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha
