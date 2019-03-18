import numpy as np
import torch
from base import BaseTrainer
import timeit
from utils import util
from collections import defaultdict
from evaluators import FormsBoxDetect_printer
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist
from datasets.testforms_box import display


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
        if 'box' in self.loss:
            self.loss['box'] = self.loss['box'](**self.loss_params['box'], 
                    num_classes=model.numBBTypes, 
                    rotation=model.rotation, 
                    scale=model.scale,
                    anchors=model.anchors)
        if 'line' in self.loss and self.loss['line'] is not None:
            if 'line' in self.loss_params:
                params = self.loss_params['line']
            else:
                params = {}
            self.loss['line'] = self.loss['line'](**params, 
                    num_classes=model.numBBTypes, 
                    scale=model.scale,
                    anchor_h=model.meanH)
        if 'loss_weights' in config:
            self.loss_weight=config['loss_weights']
        else:
            self.loss_weight={'box':0.6, 'line': 0.4, 'point':0.4, 'pixel':8}
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        #for i in range(self.start_iteration,
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        #self.log_step = int(np.sqrt(self.batch_size))
        #lr schedule from "Attention is all you need"
        #base_lr=config['optimizer']['lr']

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
        if 'num_neighbors' in instance:
            target_num_neighbors = instance['num_neighbors']
        else:
            target_num_neighbors = None
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
            if targetBoxes is not None:
                targetBoxes=targetBoxes.to(self.gpu)
            targetLines=sendToGPU(targetLines)
            targetPoints=sendToGPU(targetPoints)
            if targetPixels is not None:
                targetPixels=targetPixels.to(self.gpu)
            if target_num_neighbors is not None:
                target_num_neighbors=target_num_neighbors.to(self.gpu)
        return data, targetBoxes, targetBoxes_sizes, targetLines, targetLines_sizes, targetPoints, targetPoints_sizes, targetPixels, target_num_neighbors

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
        #self.model.eval()
        #print('WARNING EVAL')

        ##tic=timeit.default_timer()
        batch_idx = (iteration-1) % len(self.data_loader)
        try:
            thisInstance = self.data_loader_iter.next()
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            thisInstance = self.data_loader_iter.next()
        if not self.model.predNumNeighbors:
            del thisInstance['num_neighbors']
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
        #if self.iteration % self.save_step == 0:
        #    targetPoints={}
        #    targetPixels=None
        #    _,lossC=FormsBoxDetect_printer(None,thisInstance,self.model,self.gpu,self._eval_metrics,self.checkpoint_dir,self.iteration,self.loss['box'])
        #    this_loss, position_loss, conf_loss, class_loss, recall, precision = lossC
        #else:
        data, targetBoxes, targetBoxes_sizes, targetLines, targetLines_sizes, targetPoints, targetPoints_sizes, targetPixels,target_num_neighbors = self._to_tensor(thisInstance)
        outputBoxes, outputOffsets, outputLines, outputOffsetLines, outputPoints, outputPixels = self.model(data)

        if 'box' in self.loss:
            this_loss, position_loss, conf_loss, class_loss, nn_loss, recall, precision = self.loss['box'](outputOffsets,targetBoxes,targetBoxes_sizes,target_num_neighbors)

            this_loss*=self.loss_weight['box']
            loss+=this_loss
            losses['box_loss']=this_loss.item()
            #print('boxLoss:{}'.format(this_loss))
            #display(thisInstance)
        else:
            position_loss=0
            conf_loss=0
            class_loss=0

        index=0
        for name, target in targetLines.items():
            #print('line')
            predictions = outputOffsetLines[index]
            this_loss, line_pos_loss, line_conf_loss, line_class_loss = self.loss['line'](predictions,target,targetLines_sizes[name])
            this_loss*=self.loss_weight['line']
            loss+=this_loss
            losses[name+'_loss']=this_loss.item()
            losses[name+'_pos_loss']=line_pos_loss
            losses[name+'_conf_loss']=line_conf_loss
            losses[name+'_class_loss']=line_class_loss
            index+=1
        index=0
        for name, target in targetPoints.items():
            #print('point')
            predictions = outputPoints[index]
            this_loss, this_position_loss, this_conf_loss, this_class_loss = self.loss['point'](predictions,target,targetPoints_sizes[name], **self.loss_params['point'])
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
            #'recall':recall,
            #'precision':precision,
            'position_loss':position_loss,
            'conf_loss':conf_loss,
            'class_loss':class_loss,
            'num_neighbor_loss':nn_loss,
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
        total_position_loss =0
        total_conf_loss =0
        tota_class_loss =0
        tota_num_neighbor_loss =0
        total_val_metrics = np.zeros(len(self.metrics))
        losses={}
        mAP = 0
        mAP_count = 0
        numClasses = self.model.numBBTypes
        if 'no_blanks' in self.config['validation'] and not self.config['data_loader']['no_blanks']:
            numClasses-=1
        if self.model.predNumNeighbors:
            extraPreds=1
        else:
            extraPreds=0
        mRecall = np.zeros(numClasses)
        mPrecision = np.zeros(numClasses)

        with torch.no_grad():
            losses = defaultdict(lambda: 0)
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.model.predNumNeighbors:
                    del instance['num_neighbors']
                data, targetBoxes, targetBoxes_sizes, targetLines, targetLines_sizes, targetPoints, targetPoints_sizes, targetPixels,target_num_neighbors = self._to_tensor(instance)
                batchSize = data.size(0)
                #print('data: {}'.format(data.size()))
                outputBoxes,outputOffsets, outputLines, outputOffsetsLines, outputPoints, outputPixels = self.model(data)
                #loss = self.loss(output, target)
                loss = 0
                index=0
                
                if 'box' in self.loss:
                    this_loss, position_loss, conf_loss, class_loss, nn_loss, recall, precision = self.loss['box'](outputOffsets,targetBoxes,targetBoxes_sizes,target_num_neighbors)
                    loss+=this_loss*self.loss_weight['box']
                    total_position_loss+=position_loss
                    total_conf_loss+=conf_loss
                    tota_class_loss+=class_loss
                    tota_num_neighbor_loss+=nn_loss
                    losses['val_box_loss']+=this_loss.item()
                
                    threshConf = max(self.thresh_conf*outputBoxes[:,:,0].max().item(),0.5)
                    if self.model.rotation:
                        outputBoxes = non_max_sup_dist(outputBoxes.cpu(),threshConf,1.2/self.thresh_intersect)
                    else:
                        outputBoxes = non_max_sup_iou(outputBoxes.cpu(),threshConf,self.thresh_intersect)
                    if targetBoxes is not None:
                        targetBoxes = targetBoxes.cpu()
                    for b in range(batchSize):
                        #if self.model.predNumNeighbors:
                        #    useOutputBoxes=torch.cat((outputBoxes[b][:,0:6],outputBoxes[b][:,7:]),dim=1)
                        #else:
                        #    useOutputBoxes=outputBoxes[b]
                        if targetBoxes is not None:
                            target_for_b = targetBoxes[b,:targetBoxes_sizes[b],:]
                        else:
                            target_for_b = torch.empty(0)

                        if self.model.rotation:
                            ap_5, prec_5, recall_5 =AP_dist(target_for_b,outputBoxes[b],0.9,numClasses,beforeCls=extraPreds)
                        else:
                            ap_5, prec_5, recall_5 =AP_iou(target_for_b,outputBoxes[b],0.5,numClasses,beforeCls=extraPreds)
                        if ap_5 is not None:
                            mAP+=ap_5
                            mAP_count+=1
                        mRecall += np.array(recall_5,dtype=np.float)#/len(outputBoxes)
                        mPrecision += np.array(prec_5,dtype=np.float)#/len(outputBoxes)
                index=0
                for name, target in targetLines.items():
                    #print('line')
                    predictions = outputOffsetsLines[index]
                    this_loss, line_pos_loss, line_conf_loss, line_class_loss = self.loss['line'](predictions,target,targetLines_sizes[name])
                    this_loss*=self.loss_weight['line']
                    loss+=this_loss
                    losses['val_'+name+'_loss']=this_loss.item()
                    losses['val_'+name+'_pos_loss']=line_pos_loss
                    losses['val_'+name+'_conf_loss']=line_conf_loss
                    losses['val_'+name+'_class_loss']=line_class_loss
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
                loss=None
                this_loss=None
                data=None
                targetBoxes=None
                targetPoints=None
                outputBoxes=None
                outputOffsets=None
                outputPoints=None
                outputPixels=None
                #total_val_metrics += self._eval_metrics(output, target)
        for name in losses:
            losses[name]/=len(self.valid_data_loader)
        if mAP_count==0:
            mAP_count=1
        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist(),
            'val_recall':(mRecall/(batchSize*len(self.valid_data_loader))).tolist(),
            'val_precision':(mPrecision/(batchSize*len(self.valid_data_loader))).tolist(),
            'val_Fm':(mPrecision.mean()+mRecall.mean())/(2*batchSize*len(self.valid_data_loader)),
            'val_mAP':mAP/mAP_count,
            'val_position_loss':total_position_loss / len(self.valid_data_loader),
            'val_conf_loss':total_conf_loss / len(self.valid_data_loader),
            'val_class_loss':tota_class_loss / len(self.valid_data_loader),
            'val_num_neighbor_loss':tota_num_neighbor_loss / len(self.valid_data_loader),
            **losses
        }
