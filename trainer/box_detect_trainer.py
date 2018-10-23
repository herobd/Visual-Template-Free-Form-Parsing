import numpy as np
import torch
from base import BaseTrainer
import timeit
from utils import util
from collections import defaultdict
from evaluators import FormsBoxDetect_printer
from utils.yolo_tools import non_max_sup_iou, AP_iou


class BoxDetectTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(BoxDetectTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        if 'loss_params' in config:
            self.loss_params=config['loss_params']
        else:
            self.loss_params={}
        self.loss['box'] = self.loss['box'](**self.loss_params['box'], 
                num_classes=model.numBBTypes, 
                rotation=model.rotation, 
                scale=model.scale,
                anchors=model.anchors)
        if 'loss_weights' in config:
            self.loss_weight=config['loss_weights']
        else:
            self.loss_weight={'box':0.7, 'point':0.5, 'pixel':10}
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        #for i in range(self.start_iteration,
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        #self.log_step = int(np.sqrt(self.batch_size))
        #lr schedule from "Attention is all you need"
        #base_lr=config['optimizer']['lr']
        warmup_steps = config['warmup_steps'] if 'warmup_steps' in config else 1000
        lr_lambda = lambda step_num: min((step_num+1)**-0.3, (step_num+1)*warmup_steps**-1.3)
        self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda)

        self.thresh_conf = config['thresh_conf'] if 'thresh_conf' in config else 0.92
        self.thresh_intersect = config['thresh_intersect'] if 'thresh_intersect' in config else 0.4

    def _to_tensor(self, instance):
        data = instance['img']
        if 'bb_gt' in instance:
            targetBoxes = instance['bb_gt']
            targetBoxes_sizes = instance['bb_sizes']
        else:
            targetBoxes = None
            targetBoxes_sizes = []
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
            if targetBoxes is not None:
                targetBoxes=targetBoxes.to(self.gpu)
            targetPoints=sendToGPU(targetPoints)
            if targetPixels is not None:
                targetPixels=targetPixels.to(self.gpu)
        return data, targetBoxes, targetBoxes_sizes, targetPoints, targetPoints_sizes, targetPixels

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
        self.lr_schedule.step()

        ##tic=timeit.default_timer()
        batch_idx = (iteration-1) % len(self.data_loader)
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
        loss = 0
        index=0
        losses={}
        ##tic=timeit.default_timer()
        #predictions = util.pt_xyrs_2_xyxy(outputBoxes)
        if self.iteration % self.save_step == 0:
            targetPoints={}
            targetPixels=None
            _,lossC=FormsBoxDetect_printer(None,thisInstance,self.model,self.gpu,self._eval_metrics,self.checkpoint_dir,self.iteration,self.loss['box'])
            this_loss, position_loss, conf_loss, class_loss, recall, precision = lossC
        else:
            data, targetBoxes, targetBoxes_sizes, targetPoints, targetPoints_sizes, targetPixels = self._to_tensor(thisInstance)
            outputBoxes, outputOffsets, outputPoints, outputPixels = self.model(data)
            this_loss, position_loss, conf_loss, class_loss, recall, precision = self.loss['box'](outputOffsets,targetBoxes,targetBoxes_sizes)
        this_loss*=self.loss_weight['box']
        loss+=this_loss
        losses['box_loss']=this_loss.item()

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
        #what is grads?
        #minGrad=9999999999
        #maxGrad=-9999999999
        #for p in filter(lambda p: p.grad is not None, self.model.parameters()):
        #    minGrad = min(minGrad,p.min())
        #    maxGrad = max(maxGrad,p.max())
        torch.nn.utils.clip_grad_value_(self.model.parameters(),1)
        self.optimizer.step()

        ##toc=timeit.default_timer()
        ##print('bac: '+str(toc-tic))

        #tic=timeit.default_timer()
        metrics={}
        #index=0
        #for name, target in targetBoxes.items():
        #    metrics = {**metrics, **self._eval_metrics('box',name,output, target)}
        #for name, target in targetPoints.items():
        #    metrics = {**metrics, **self._eval_metrics('point',name,output, target)}
        #    metrics = self._eval_metrics(name,output, target)
        #toc=timeit.default_timer()
        #print('metric: '+str(toc-tic))

        ##tic=timeit.default_timer()
        loss = loss.item()
        ##toc=timeit.default_timer()
        ##print('item: '+str(toc-tic))
        #perAnchor={}
        #for i in range(avg_conf_per_anchor.size(0)):
        #    perAnchor['anchor{}'.format(i)]=avg_conf_per_anchor[i]

        log = {
            'loss': loss,
            'recall':recall,
            #'precision':precision,
            #'position_loss':position_loss,
            'conf_loss':conf_loss,
            'class_loss':class_loss,
            #'minGrad':minGrad,
            #'maxGrad':maxGrad,
            #'cor_conf_loss':cor_conf_loss,
            #'conf_loss':conf_loss,
            #'conf':confs.mean(),
            #'bad_conf':bad_confs.mean(),
            #**perAnchor,

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
        mAP = np.zeros(self.model.numBBTypes)
        mRecall = np.zeros(self.model.numBBTypes)
        mPrecision = np.zeros(self.model.numBBTypes)
        with torch.no_grad():
            losses = defaultdict(lambda: 0)
            for batch_idx, instance in enumerate(self.valid_data_loader):
                data, targetBoxes, targetBoxes_sizes, targetPoints, targetPoints_sizes, targetPixels = self._to_tensor(instance)

                outputBoxes,outputOffsets, outputPoints, outputPixels = self.model(data)
                #loss = self.loss(output, target)
                loss = 0
                index=0
                
                this_loss, position_loss, conf_loss, class_loss, recall, precision = self.loss['box'](outputOffsets,targetBoxes,targetBoxes_sizes)
                loss+=this_loss*self.loss_weight['box']
                losses['val_box_loss']+=this_loss.item()
                
                threshConf = self.thresh_conf*outputBoxes[:,:,0].max()
                outputBoxes = non_max_sup_iou(outputBoxes.cpu(),self.thresh_conf,self.thresh_intersect)
                targetBoxes = targetBoxes.cpu()
                for b in range(len(outputBoxes)):
                    if targetBoxes is not None:
                        target_for_b = targetBoxes[b,:targetBoxes_sizes[b],:]
                    else:
                        target_for_b = torch.empty(0)
                    ap_5, prec_5, recall_5 =AP_iou(target_for_b,outputBoxes[b],0.5,self.model.numBBTypes)
                    mAP += np.array(ap_5)/len(outputBoxes)
                    mRecall += np.array(recall_5)/len(outputBoxes)
                    mPrecision += np.array(prec_5)/len(outputBoxes)
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
            'val_recall':(mRecall/len(self.valid_data_loader)).tolist(),
            'val_precision':(mPrecision/len(self.valid_data_loader)).tolist(),
            'val_mAP':(mAP/len(self.valid_data_loader)).tolist(),
            'val_position_loss':position_loss,
            'val_conf_loss':conf_loss,
            'val_class_loss':class_loss,
            **losses
        }
