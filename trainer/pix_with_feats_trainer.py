import numpy as np
import torch
from .trainer import Trainer
import timeit
from model import *
import torch.optim as optim


class PixWithFeatsTrainer(Trainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(PixWithFeatsTrainer, self).__init__(model, loss, metrics, resume, config, data_loader, valid_data_loader, train_logger)
        checkpoint = torch.load(config["trainer"]['detector_checkpoint'])
        detector_config = config["trainer"]['detector_config'] if 'detector_config' in config else checkpoint['config']['model']
        if 'state_dict' in checkpoint:
            self.detector = eval(checkpoint['config']['arch'])(detector_config)
            self.detector.load_state_dict(checkpoint['state_dict'])
        else:
            self.detector = checkpoint['model']
        self.detector.forPairing=True
        self.detector = self.detector.to(self.gpu)
        #self.model.build(self.detector.last_channels,self.detector.scale)
        #self.model = self.model.to(self.gpu)
        #self.optimizer = getattr(optim, config['optimizer_type_late'])(self.model.parameters(),
        #                                                                          **config['optimizer'])
        for param in self.detector.parameters():
            param.requires_grad=False
        self.storedImageName=None
        self.imgChs = 3 if 'color' not in detector_config or detector_config['color'] else 1

    #def _to_tensor(self, data, target):
    #    return self._to_tensor_individual(data), _to_tensor_individual(target)

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

        #tic=timeit.default_timer()
        batch_idx = (iteration-1) % len(self.data_loader)
        try:
            image, imageName, target = self._to_tensor(*self.data_loader_iter.next())
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            image, imageName, target = self._to_tensor(*self.data_loader_iter.next())
        #toc=timeit.default_timer()
        #print('data: '+str(toc-tic))
        
        #tic=timeit.default_timer()
        padH=(self.detector.scale-(image.size(2)%self.detector.scale))%self.detector.scale
        padW=(self.detector.scale-(image.size(3)%self.detector.scale))%self.detector.scale
        if padH!=0 or padW!=0:
            padder = torch.nn.ZeroPad2d((0,padW,0,padH))
            image = padder(image)
        self.detector(image[:,:self.imgChs])
        final_features=self.detector.final_features

        self.optimizer.zero_grad()
        output = self.model(image,final_features)
        output = output[:,:target.size(1),:target.size(2)]
        loss = self.loss(output, target)
        loss.backward()
        self.optimizer.step()

        #toc=timeit.default_timer()
        #print('for/bac: '+str(toc-tic))

        #tic=timeit.default_timer()
        metrics = self._eval_metrics(output, target)
        #toc=timeit.default_timer()
        #print('metric: '+str(toc-tic))

        #tic=timeit.default_timer()
        loss = loss.item()
        #toc=timeit.default_timer()
        #print('item: '+str(toc-tic))


        log = {
            'loss': loss,
            'metrics': metrics
        }


        return log

    def _minor_log(self, log):
        ls=''
        for key,val in log.items():
            ls += key
            if type(val) is float:
                ls +=': {:.6f}, '.format(val)
            else:
                ls +=': {}, '.format(val)
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
            for batch_idx, (image, imageName, target) in enumerate(self.valid_data_loader):
                print('iter:{} valid batch: {}/{}'.format(self.iteration,batch_idx,len(self.valid_data_loader)), end='\r')
                image, target = self._to_tensor(image, target)
                padH=(self.detector.scale-(image.size(2)%self.detector.scale))%self.detector.scale
                padW=(self.detector.scale-(image.size(3)%self.detector.scale))%self.detector.scale
                if padH!=0 or padW!=0:
                    padder = torch.nn.ZeroPad2d((0,padW,0,padH))
                    image = padder(image)
                    #padH=(self.detector.scale-((image.size(2)//2)%self.detector.scale))%self.detector.scale
                    #padW=(self.detector.scale-((image.size(3)//2)%self.detector.scale))%self.detector.scale
                    #padder = torch.nn.ZeroPad2d((0,padW,0,padH))
                    #target = padder(target)
                if self.storedImageName is not None and imageName==self.storedImageName:
                    #offsetPredictionsD=self.storedOffsetPredictionsD
                    final_features=self.storedFinal_features
                else:
                    self.detector(image[:,:self.imgChs])
                    final_features=self.detector.final_features

                    self.storedFinal_features=final_features
                    self.storedImageName=imageName

                output = self.model(image,final_features)
                output = output[...,:target.size(1),:target.size(2)]
                loss = self.loss(output, target)

                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
