import numpy as np
import torch
from torch.nn import functional as F
#from base import BaseTrainer
from .trainer import Trainer
import timeit
from utils import util
from collections import defaultdict
from evaluators import FormsBoxDetect_printer
from utils.yolo_tools import non_max_sup_iou, AP_iou, computeAP


class FeaturePairTrainer(Trainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(FeaturePairTrainer, self).__init__(model, loss, metrics, resume, config,
                data_loader, valid_data_loader, train_logger)
        #self.config = config
        #self.batch_size = data_loader.batch_size
        #self.data_loader = data_loader
        #self.data_loader_iter = iter(data_loader)
        #self.valid_data_loader = valid_data_loader
        #self.valid = True if self.valid_data_loader is not None else False

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
        #self.lr_schedule.step()

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
        index=0
        losses={}
        ##tic=timeit.default_timer()

        #if self.iteration % self.save_step == 0:
        #    targetPoints={}
        #    targetPixels=None
        #    _,lossC=FormsBoxPair_printer(None,thisInstance,self.model,self.gpu,self._eval_metrics,self.checkpoint_dir,self.iteration,self.loss['box'])
        #    loss, position_loss, conf_loss, class_loss, recall, precision = lossC
        #else:
        data,label = self._to_tensor(thisInstance['data'],thisInstance['label'])
        output = self.model(data)
        outputRel = output[:,0]
        if output.size(1)==3:
            outputNN = output[:,1:]
            gtNN = self._to_tensor(thisInstance['numNeighbors'])
            lossNN = F.mse_loss(outputNN,gtNN[0])
        else:
            lossNN=0
        lossRel = self.loss(outputRel,label)

        loss = lossRel+lossNN

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
        #import pdb; pdb.set_trace()
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
        lossRel=lossRel.item()
        if type(lossNN) is not int:
            lossNN=lossNN.item()
        ##toc=timeit.default_timer()
        ##print('item: '+str(toc-tic))
        #perAnchor={}
        #for i in range(avg_conf_per_anchor.size(0)):
        #    perAnchor['anchor{}'.format(i)]=avg_conf_per_anchor[i]

        log = {
            'loss': loss,
            'lossRel':lossRel,
            'lossNN':lossNN,

            **metrics,
            **losses
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


    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_lossRel = 0
        total_val_lossNN = 0
        total_val_metrics = np.zeros(len(self.metrics))

        tp_image=defaultdict(lambda:0)
        fp_image=defaultdict(lambda:0)
        tn_image=defaultdict(lambda:0)
        fn_image=defaultdict(lambda:0)
        images=set()
        scores=defaultdict(list)

        with torch.no_grad():
            losses = defaultdict(lambda: 0)
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.logged:
                    print('iter:{} valid batch: {}/{}'.format(self.iteration,batch_idx,len(self.valid_data_loader)), end='\r')

                data,label = self._to_tensor(instance['data'],instance['label'])
                output = self.model(data)
                outputRel = output[:,0]
                if output.size(1)==3:
                    outputNN = output[:,1:]
                    gtNN = self._to_tensor(instance['numNeighbors'])
                    lossNN = F.mse_loss(outputNN,gtNN[0])
                else:
                    lossNN=0
                lossRel = self.loss(outputRel,label)

                loss = lossRel+lossNN
                
                
                for b in range(len(output)):
                    image = instance['imgName'][b]
                    images.add(image)
                    scores[image].append( (outputRel[b],label[b]) )
                    if outputRel[b]<0.5:
                        if label[b]==0:
                            tn_image[image]+=1
                        else:
                            fn_image[image]+=1
                    else:
                        if label[b]==0:
                            fp_image[image]+=1
                        else:
                            tp_image[image]+=1

                total_val_loss += loss.item()
                total_val_lossRel += lossRel.item()
                if type(lossNN) is not int:
                    lossNN=lossNN.item()
                total_val_lossNN += lossNN

        mRecall=0
        mPrecision=0
        mAP=0
        mAP_count=0
        
        for image in images:
            ap = computeAP(scores[image])
            if ap is not None:
                mAP+=ap
                mAP_count+=1
            if tp_image[image]+fn_image[image]>0:
                mRecall += tp_image[image]/(tp_image[image]+fn_image[image])
            else:
                mRecall += 1
            if tp_image[image]+fp_image[image]>0:
                mPrecision += tp_image[image]/(tp_image[image]+fp_image[image])
            else:
                mPrecision += 1
        mRecall /= len(images)
        mPrecision /= len(images)
        if mAP_count>0:
            mAP /= mAP_count

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_lossRel': total_val_lossRel / len(self.valid_data_loader),
            'val_lossNN': total_val_lossNN / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist(),
            'val_recall*':mRecall,
            'val_precision*':mPrecision,
            'val_mAP*': mAP
        }
