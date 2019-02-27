import numpy as np
import torch
from base import BaseTrainer
import timeit
from utils import util
from collections import defaultdict
from evaluators import FormsBoxDetect_printer
from utils.yolo_tools import non_max_sup_iou, AP_iou, non_max_sup_dist, AP_dist, getTargIndexForPreds_iou, getTargIndexForPreds_dist, computeAP
from datasets.testforms_graph_pair import display
import random


class GraphPairTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(GraphPairTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        if 'loss_params' in config:
            self.loss_params=config['loss_params']
        else:
            self.loss_params={}
        self.lossWeights = config['loss_weights'] if 'loss_weights' in config else {"box": 1, "rel":1}
        self.loss['box'] = self.loss['box'](**self.loss_params['box'], 
                num_classes=model.numBBTypes, 
                rotation=model.rotation, 
                scale=model.scale,
                anchors=model.anchors)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        #for i in range(self.start_iteration,
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        #self.log_step = int(np.sqrt(self.batch_size))
        #lr schedule from "Attention is all you need"
        #base_lr=config['optimizer']['lr']
        self.useLearningSchedule = config['trainer']['use_learning_schedule'] if 'use_learning_schedule' in config['trainer'] else False
        if self.useLearningSchedule=='cyclic':
            min_lr_mul = config['trainer']['min_lr_mul'] if 'min_lr_mul' in config['trainer'] else 0.001
            cycle_size = config['trainer']['cycle_size'] if 'cycle_size' in config['trainer'] else 500
            lr_lambda = lambda step_num: (1-(1-min_lr_mul)*((step_num-1)%cycle_size)/(cycle_size-1))
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda)
        elif self.useLearningSchedule is True:
            warmup_steps = config['trainer']['warmup_steps'] if 'warmup_steps' in config['trainer'] else 1000
            #lr_lambda = lambda step_num: min((step_num+1)**-0.3, (step_num+1)*warmup_steps**-1.3)
            lr_lambda = lambda step_num: min((max(0.000001,step_num-(warmup_steps-3))/100)**-0.1, step_num*(1.485/warmup_steps)+.01)
            #y=((x-(2000-3))/100)^-0.1 and y=x*(1.485/2000)+0.01
            self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda)
        else:
            print('Unrecognized learning schedule: {}'.format(self.useLearningSchedule))
            exit()


        #default is unfrozen, can be frozen by setting 'start_froze' in the PairingGraph models params
        self.unfreeze_detector = config['trainer']['unfreeze_detector'] if 'unfreeze_detector' in config['trainer'] else None

        self.thresh_conf = config['trainer']['thresh_conf'] if 'thresh_conf' in config['trainer'] else 0.92
        self.thresh_intersect = config['trainer']['thresh_intersect'] if 'thresh_intersect' in config['trainer'] else 0.4
        self.thresh_rel = config['trainer']['thresh_rel'] if 'thresh_rel' in config['trainer'] else 0.5

        #we iniailly train the pairing using GT BBs, but eventually need to fine-tune the pairing using the networks performance
        self.stop_from_gt = config['trainer']['stop_from_gt'] if 'stop_from_gt' in config['trainer'] else None
        self.partial_from_gt = config['trainer']['partial_from_gt'] if 'partial_from_gt' in config['trainer'] else None
        self.max_use_pred = config['trainer']['max_use_pred'] if 'max_use_pred' in config['trainer'] else 0.9

        self.conf_thresh_init = config['trainer']['conf_thresh_init'] if 'conf_thresh_init' in config['trainer'] else 0.9
        self.conf_thresh_change_iters = config['trainer']['conf_thresh_change_iters'] if 'conf_thresh_change_iters' in config['trainer'] else 5000

        self.train_hard_detect_limit = config['trainer']['train_hard_detect_limit'] if 'train_hard_detect_limit' in config else 100
        self.val_hard_detect_limit = config['trainer']['val_hard_detect_limit'] if 'val_hard_detect_limit' in config else 300

        self.useBadBBPredForRelLoss = config['trainer']['use_bad_bb_pred_for_rel_loss'] if 'use_bad_bb_pred_for_rel_loss' in config else False

        self.adaptLR = config['trainer']['adapt_lr'] if 'adapt_lr' in config else False
        self.adaptLR_base = config['trainer']['adapt_lr_base'] if 'adapt_lr_base' in config else 165 #roughly average number of rels
        self.adaptLR_ep = config['trainer']['adapt_lr_ep'] if 'adapt_lr_ep' in config else 15

        #Name change
        if 'edge' in self.lossWeights:
            self.lossWeights['rel'] = self.lossWeights['edge']
        if 'edge' in self.loss:
            self.loss['rel'] = self.loss['edge']

    def _to_tensor(self, instance):
        image = instance['img']
        bbs = instance['bb_gt']
        adjaceny = instance['adj']
        num_neighbors = instance['num_neighbors']

        if self.with_cuda:
            image = image.to(self.gpu)
            if bbs is not None:
                bbs = bbs.to(self.gpu)
            if num_neighbors is not None:
                num_neighbors = num_neighbors.to(self.gpu)
            #adjacenyMatrix = adjacenyMatrix.to(self.gpu)
        return image, bbs, adjaceny, num_neighbors

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

    def useGT(self,iteration):
        if self.stop_from_gt is not None and iteration>=self.stop_from_gt:
            return random.random()>self.max_use_pred #I think it's best to always have some GT examples
        elif self.partial_from_gt is not None and iteration>=self.partial_from_gt:
            return random.random()> self.max_use_pred*(iteration-self.partial_from_gt)/(self.stop_from_gt-self.partial_from_gt)
        else:
            return True

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
        if self.unfreeze_detector is not None and iteration>=self.unfreeze_detector:
            self.model.unfreeze()
        self.model.train()
        #self.model.eval()
        #print("WARNING EVAL")
        if self.useLearningSchedule:
            self.lr_schedule.step()

        ##tic=timeit.default_timer()
        batch_idx = (iteration-1) % len(self.data_loader)
        try:
            thisInstance = self.data_loader_iter.next()
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            thisInstance = self.data_loader_iter.next()
        if not self.model.detector.predNumNeighbors:
            thisInstance['num_neighbors']=None
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
        if self.conf_thresh_change_iters > iteration:
            threshIntur = 1 - iteration/self.conf_thresh_change_iters
        else:
            threshIntur = None
        image, targetBoxes, adj, target_num_neighbors = self._to_tensor(thisInstance)
        useGT = self.useGT(iteration)
        if useGT:
            outputBoxes, outputOffsets, relPred, relIndexes, bbPred = self.model(image,targetBoxes,target_num_neighbors,True,
                    otherThresh=self.conf_thresh_init, otherThreshIntur=threshIntur, hard_detect_limit=self.train_hard_detect_limit)
            #_=None
            #gtPairing,predPairing = self.prealignedEdgePred(adj,relPred)
            predPairingShouldBeTrue,predPairingShouldBeFalse, eRecall,ePrec,fullPrec,ap = self.prealignedEdgePred(adj,relPred,relIndexes)
            if self.model.predNN or self.model.predClass:
                if target_num_neighbors is not None:
                    alignedNN_use = target_num_neighbors[0]
                bbPred_use = bbPred
        else:
            outputBoxes, outputOffsets, relPred, relIndexes, bbPred = self.model(image,
                    otherThresh=self.conf_thresh_init, otherThreshIntur=threshIntur, hard_detect_limit=self.train_hard_detect_limit)
            #gtPairing,predPairing = self.alignEdgePred(targetBoxes,adj,outputBoxes,relPred)
            predPairingShouldBeTrue,predPairingShouldBeFalse, eRecall,ePrec,fullPrec,ap, bbAlignment, bbNoIntersections = self.alignEdgePred(targetBoxes,adj,outputBoxes,relPred,relIndexes)
            if (self.model.predNN or self.model.predClass) and bbPred is not None:
                #create aligned GT
                #first, remove unmatched predicitons that didn't overlap (weren't close) to any targets
                toKeep = 1-((bbNoIntersections==1) * (bbAlignment==-1))
                bbPred_use = bbPred[toKeep]
                #alignedNN_use = alignedNN[toKeep]
                bbAlignment_use = bbAlignment[toKeep]
                #becuase we used -1 to indicate no match (in bbAlignment), we add 0 as the last position in the GT, as unmatched 
                if target_num_neighbors is not None:
                    target_num_neighbors_use = torch.cat((target_num_neighbors[0].float(),torch.zeros(1).to(target_num_neighbors.device)),dim=0)
                else:
                    target_num_neighbors_use = torch.zeros(1).to(bbPred.device)
                alignedNN_use = target_num_neighbors_use[bbAlignment_use]
            else:
                bbPred_use = None
        if relPred is not None:
            numEdgePred = relPred.size(0)
            if predPairingShouldBeTrue is not None:
                lenTrue = predPairingShouldBeTrue.size(0)
            else:
                lenTrue = 0
            if predPairingShouldBeFalse is not None:
                lenFalse = predPairingShouldBeFalse.size(0)
            else:
                lenFalse = 0
        else:
            numEdgePred = lenTrue = lenFalse = 0
        numBoxPred = outputBoxes.size(0)
        #if iteration>25:
        #    import pdb;pdb.set_trace()
        #if len(predPairing.size())>0 and predPairing.size(0)>0:
        #    relLoss = self.loss['rel'](predPairing,gtPairing)
        #else:
        #    relLoss = torch.tensor(0.0,requires_grad=True).to(image.device)
        #relLoss = torch.tensor(0.0).to(image.device)
        relLoss = None
        if predPairingShouldBeTrue is not None and predPairingShouldBeTrue.size(0)>0:
            ones = torch.ones_like(predPairingShouldBeTrue).to(image.device)
            relLoss = self.loss['rel'](predPairingShouldBeTrue,ones)
            debug_avg_relTrue = predPairingShouldBeTrue.mean().item()
        else:
            debug_avg_relTrue =0 
        if predPairingShouldBeFalse is not None and predPairingShouldBeFalse.size(0)>0:
            zeros = torch.zeros_like(predPairingShouldBeFalse).to(image.device)
            relLossFalse = self.loss['rel'](predPairingShouldBeFalse,zeros)
            if relLoss is None:
                relLoss=relLossFalse
            else:
                relLoss+=relLossFalse
            debug_avg_relFalse = predPairingShouldBeFalse.mean().item()
        else:
            debug_avg_relFalse = 0
        if relLoss is not None:
            relLoss *= self.lossWeights['rel']



        if not self.model.detector_frozen:
            if targetBoxes is not None:
                targSize = targetBoxes.size(1)
            else:
                targSize =0 
            #import pdb;pdb.set_trace()
            boxLoss, position_loss, conf_loss, class_loss, nn_loss, recall, precision = self.loss['box'](outputOffsets,targetBoxes,[targSize],target_num_neighbors)
            boxLoss *= self.lossWeights['box']
            if relLoss is not None:
                loss = relLoss + boxLoss
            else:
                loss = boxLoss
        else:
            loss = relLoss


        if self.model.predNN and bbPred_use is not None and bbPred_use.size(0)>0:
            nn_loss_final = self.loss['nn'](bbPred_use[:,0],alignedNN_use)
            nn_loss_final *= self.lossWeights['nn']

            loss += nn_loss_final
            nn_loss_final = nn_loss_final.item()
        else:
            nn_loss_final=0

        if self.model.predClass and bbPred_use is not None:
            raise NotImplemented('havent implemented loss for graph pred of classes')
            
        ##toc=timeit.default_timer()
        ##print('loss: '+str(toc-tic))
        ##tic=timeit.default_timer()
        predPairingShouldBeTrue= predPairingShouldBeFalse=outputBoxes=outputOffsets=relPred=image=targetBoxes=relLossFalse=None
        if relLoss is not None:
            relLoss = relLoss.item()
        else:
            relLoss = 0
        if not self.model.detector_frozen:
            boxLoss = boxLoss.item()
        else:
            boxLoss = 0
        if loss is not None:
            if self.adaptLR:
                #if we only have a few relationship preds, step smaller so that we don't skew with a bad bias
                #This effects the box loss too so that it doesn't yank the detector/backbone features around
                #we actually just scale the loss, but its all the same :)
                scale = (numEdgePred+self.adaptLR_ep)/(self.adaptLR_ep+self.adaptLR_base)
                loss *= scale
            loss.backward()

            torch.nn.utils.clip_grad_value_(self.model.parameters(),1)
            self.optimizer.step()

            loss = loss.item()
        else:
            loss=0

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

        #perAnchor={}
        #for i in range(avg_conf_per_anchor.size(0)):
        #    perAnchor['anchor{}'.format(i)]=avg_conf_per_anchor[i]

        log = {
            'loss': loss,
            'boxLoss': boxLoss,
            'relLoss': relLoss,
            'nn_loss_final': nn_loss_final,
            'edgePredLens':np.array([numEdgePred,numBoxPred,numEdgePred+numBoxPred,-1],dtype=np.float),
            'rel_recall':eRecall,
            #'rel_prec': ePrec,
            'rel_fullPrec':fullPrec,
            'rel_F': (eRecall+fullPrec)/2,
            #'debug_avg_relTrue': debug_avg_relTrue,
            #'debug_avg_relFalse': debug_avg_relFalse,

            **metrics,
        }
        if ap is not None:
            log['rel_AP']=ap

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
        total_val_loss = 0
        total_box_loss =0
        total_rel_loss =0
        total_rel_recall=0
        total_rel_prec=0
        total_rel_fullPrec=0
        total_AP=0
        AP_count=0
        total_val_metrics = np.zeros(len(self.metrics))
        nn_loss_final_total=0

        numClasses = self.model.numBBTypes
        if 'no_blanks' in self.config['validation'] and not self.config['data_loader']['no_blanks']:
            numClasses-=1
        mAP = 0
        mAP_count = 0
        mRecall = np.zeros(numClasses)
        mPrecision = np.zeros(numClasses)

        with torch.no_grad():
            losses = defaultdict(lambda: 0)
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.model.detector.predNumNeighbors:
                    instance['num_neighbors']=None
                if not self.logged:
                    print('iter:{} valid batch: {}/{}'.format(self.iteration,batch_idx,len(self.valid_data_loader)), end='\r')

                image, targetBoxes, adjM, target_num_neighbors = self._to_tensor(instance)

                outputBoxes, outputOffsets, relPred, relIndexes, bbPred = self.model(image, hard_detect_limit=self.val_hard_detect_limit)
                #loss = self.loss(output, target)
                loss = 0
                index=0
                
                predPairingShouldBeTrue,predPairingShouldBeFalse, recall,prec,fullPrec, ap, bbAlignment, bbNoIntersections = self.alignEdgePred(targetBoxes,adjM,outputBoxes,relPred,relIndexes)
                total_rel_recall+=recall
                total_rel_prec+=prec
                total_rel_fullPrec+=fullPrec
                if ap is not None:
                    total_AP+=ap
                    AP_count+=1
                #relLoss = torch.tensor(0.0,requires_grad=True).to(image.device)
                relLoss=None
                if predPairingShouldBeTrue is not None and predPairingShouldBeTrue.size(0)>0:
                    relLoss = self.loss['rel'](predPairingShouldBeTrue,torch.ones_like(predPairingShouldBeTrue).to(image.device))
                if predPairingShouldBeFalse is not None  and predPairingShouldBeFalse.size(0)>0:
                    relFalseLoss = self.loss['rel'](predPairingShouldBeFalse,torch.zeros_like(predPairingShouldBeFalse).to(image.device))
                    if relLoss is not None:
                        relLoss += relFalseLoss
                    else:
                        relLoss = relFalseLoss
                if relLoss is None:
                    relLoss = torch.tensor(0.0)
                else:
                    relLoss = relLoss.cpu()
                if not self.model.detector_frozen:
                    boxLoss, position_loss, conf_loss, class_loss, nn_loss, recallX, precisionX = self.loss['box'](outputOffsets,targetBoxes,[targetBoxes.size(1)],target_num_neighbors)
                    loss = relLoss*self.lossWeights['rel'] + boxLoss.cpu()*self.lossWeights['box']
                else:
                    boxLoss=torch.tensor(0.0)
                    loss = relLoss*self.lossWeights['rel']
                total_box_loss+=boxLoss.item()
                total_rel_loss+=relLoss.item()
                if self.model.predNN and bbPred is not None:
                    #create aligned GT
                    #first, remove unmatched predicitons that didn't overlap (weren't close) to any targets
                    toKeep = 1-((bbNoIntersections==1) * (bbAlignment==-1))
                    bbPred_use = bbPred[toKeep]
                    bbAlignment_use = bbAlignment[toKeep]
                    #becuase we used -1 to indicate no match (in bbAlignment), we add 0 as the last position in the GT, as unmatched 
                    if target_num_neighbors is not None:
                        target_num_neighbors_use = torch.cat((target_num_neighbors[0].float(),torch.zeros(1).to(target_num_neighbors.device)),dim=0)
                    else:
                        target_num_neighbors_use = torch.zeros(1).to(bbPred.device)
                    alignedNN_use = target_num_neighbors_use[bbAlignment_use]
                    
                    nn_loss_final = self.loss['nn'](bbPred_use[:,0],alignedNN_use)
                    nn_loss_final *= self.lossWeights['nn']

                    loss += nn_loss_final.to(loss.device)
                    nn_loss_final = nn_loss_final.item()
                    nn_loss_final_total += nn_loss_final
                else:
                    nn_loss_final=0
                
                if self.model.detector.predNumNeighbors:
                    outputBoxes=torch.cat((outputBoxes[:,0:6],outputBoxes[:,7:]),dim=1) #throw away NN pred
                if targetBoxes is not None:
                    targetBoxes = targetBoxes.cpu()
                if targetBoxes is not None:
                    target_for_b = targetBoxes[0]
                else:
                    target_for_b = torch.empty(0)
                if self.model.rotation:
                    ap_5, prec_5, recall_5 =AP_dist(target_for_b,outputBoxes,0.9,numClasses)
                else:
                    ap_5, prec_5, recall_5 =AP_iou(target_for_b,outputBoxes,0.5,numClasses)

                #import pdb;pdb.set_trace()
                if ap_5 is not None:
                    mAP+=ap_5
                    mAP_count+=1
                mRecall += np.array(recall_5)
                mPrecision += np.array(prec_5)

                total_val_loss += loss.item()
                loss=relFalseLoss=relLoss=boxLoss=None
                instance=predPairingShouldBeTrue= predPairingShouldBeFalse=outputBoxes=outputOffsets=relPred=image=targetBoxes=relLossFalse=None
                #total_val_metrics += self._eval_metrics(output, target)
        toRet= {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_box_loss': total_box_loss / len(self.valid_data_loader),
            'val_rel_loss': total_rel_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist(),
            'val_bb_recall':(mRecall/len(self.valid_data_loader)).tolist(),
            'val_bb_precision':(mPrecision/len(self.valid_data_loader)).tolist(),
            #'val_bb_F':(( (mRecall+mPrecision)/2 )/len(self.valid_data_loader)).tolist(),
            'val_bb_F_avg':(( (mRecall+mPrecision)/2 )/len(self.valid_data_loader)).mean(),
            'val_bb_mAP':(mAP/mAP_count),
            'val_rel_recall':total_rel_recall/len(self.valid_data_loader),
            'val_rel_prec':total_rel_prec/len(self.valid_data_loader),
            'val_rel_F':(total_rel_prec+total_rel_recall)/(2*len(self.valid_data_loader)),
            'val_rel_fullPrec':total_rel_fullPrec/len(self.valid_data_loader),
            'val_rel_mAP': total_AP/AP_count
            #'val_position_loss':total_position_loss / len(self.valid_data_loader),
            #'val_conf_loss':total_conf_loss / len(self.valid_data_loader),
            #'val_class_loss':tota_class_loss / len(self.valid_data_loader),
        }
        if self.model.predNN:
            toRet['val_nn_loss_final']=nn_loss_final_total/len(self.valid_data_loader)
        return toRet


    def alignEdgePred(self,targetBoxes,adj,outputBoxes,relPred,relIndexes):
        if relPred is None or targetBoxes is None:
            if targetBoxes is None:
                if relPred is not None and (relPred>self.thresh_rel).any():
                    prec=0
                    ap=0
                else:
                    prec=1
                    ap=1
                recall=1
                targIndex = -torch.ones(outputBoxes.size(0))
            elif relPred is None:
                if targetBoxes is not None:
                    recall=0
                    ap=0
                else:
                    recall=1
                    ap=1
                prec=1
                targIndex = None

            return torch.tensor([]),torch.tensor([]),recall,prec,prec,ap, targIndex, torch.ones(outputBoxes.size(0))
        targetBoxes = targetBoxes.cpu()
        #decide which predicted boxes belong to which target boxes
        #should this be the same as AP_?
        numClasses = 2

        if self.useBadBBPredForRelLoss == 'fixed':
            #fullHit = predsWithNoIntersection
            if self.model.rotation:
                targIndex, fullHit = getTargIndexForPreds_dist(targetBoxes[0],outputBoxes,1.1,numClasses,hard_thresh=False)
            else:
                targIndex, fullHit = getTargIndexForPreds_iou(targetBoxes[0],outputBoxes,0.4,numClasses,hard_thresh=False)
        else:
            if self.model.rotation:
                targIndex, predsWithNoIntersection = getTargIndexForPreds_dist(targetBoxes[0],outputBoxes,1.1,numClasses)
            else:
                targIndex, predsWithNoIntersection = getTargIndexForPreds_iou(targetBoxes[0],outputBoxes,0.4,numClasses)

        #Create gt vector to match relPred.values()

        rels = relIndexes #relPred._indices().cpu()
        predsAll = relPred #relPred._values()
        sigPredsAll = torch.sigmoid(predsAll)
        #newGT = []#torch.tensor((rels.size(0)))
        predsPos = []
        predsNeg = []
        scores = []
        matches=0
        #for i in range(rels.size(0)):
        truePred=falsePred=badPred=0
        for i,(n0,n1) in enumerate(rels):
            #n0 = rels[i,0]
            #n1 = rels[i,1]
            t0 = targIndex[n0].item()
            t1 = targIndex[n1].item()
            if t0>=0 and t1>=0:
                #newGT.append( int((t0,t1) in adj) )#adjM[ min(t0,t1), max(t0,t1) ])
                #preds.append(predsAll[i])
                if (min(t0,t1),max(t0,t1)) in adj:
                    if self.useBadBBPredForRelLoss!='fixed' or (fullHit[n0] and fullHit[n1]):
                        matches+=1
                        predsPos.append(predsAll[i])
                        scores.append( (sigPredsAll[i],True) )
                        if sigPredsAll[i]>self.thresh_rel:
                            truePred+=1
                    else:
                        scores.append( (sigPredsAll[i],False) ) #for the sake of scoring, this is a bad relationship
                else:
                    predsNeg.append(predsAll[i])
                    scores.append( (sigPredsAll[i],False) )
                    if sigPredsAll[i]>self.thresh_rel:
                        falsePred+=1
            else:
                if self.useBadBBPredForRelLoss=='fixed' or (self.useBadBBPredForRelLoss and (predsWithNoIntersection[n0] or predsWithNoIntersection[n1])):
                    predsNeg.append(predsAll[i])
                scores.append( (sigPredsAll[i],False) )
                if sigPredsAll[i]>self.thresh_rel:
                    badPred+=1
        #Add score 0 for instances we didn't predict
        for i in range(len(adj)-matches):
            scores.append( (float('nan'),True) )
        #if len(preds)==0:
        #    return torch.tensor([]),torch.tensor([])
        #newGT = torch.tensor(newGT).float().to(relPred[1].device)
        #preds = torch.cat(preds).to(relPred[1].device)
    
        #return newGT, preds
        if len(predsPos)>0:
            predsPos = torch.cat(predsPos).to(relPred.device)
        else:
            predsPos = None
        if len(predsNeg)>0:
            predsNeg = torch.cat(predsNeg).to(relPred.device)
        else:
            predsNeg = None

        if len(adj)>0:
            recall = truePred/len(adj)
        else:
            recall = 1
        if falsePred>0:
            prec = truePred/(truePred+falsePred)
        else:
            prec = 1
        if falsePred+badPred>0:
            fullPrec = truePred/(truePred+falsePred+badPred)
        else:
            fullPrec = 1
        return predsPos,predsNeg, recall, prec ,fullPrec, computeAP(scores), targIndex, predsWithNoIntersection


    def prealignedEdgePred(self,adj,relPred,relIndexes):
        if relPred is None:
            #assert(adj is None or len(adj)==0) this is a failure of the heuristic pairing
            if adj is not None and len(adj)>0:
                recall=0
                ap=0
            else:
                recall=1
                ap=1
            prec=1

            return torch.tensor([]),torch.tensor([]),recall,prec,prec,ap
        rels = relIndexes #relPred._indices().cpu().t()
        predsAll = relPred
        sigPredsAll = torch.sigmoid(predsAll)

        #gt = torch.empty(len(rels))#rels.size(0))
        predsPos = []
        predsNeg = []
        scores = []
        truePred=falsePred=0
        for i,(n0,n1) in enumerate(rels):
            #n0 = rels[i,0]
            #n1 = rels[i,1]
            #gt[i] = int((n0,n1) in adj) #(adjM[ n0, n1 ])
            if (n0,n1) in adj:
                predsPos.append(predsAll[i])
                scores.append( (sigPredsAll[i],True) )
                if sigPredsAll[i]>self.thresh_rel:
                    truePred+=1
            else:
                predsNeg.append(predsAll[i])
                scores.append( (sigPredsAll[i],False) )
                if sigPredsAll[i]>self.thresh_rel:
                    falsePred+=1
    
        #return gt.to(relPred.device), relPred._values().view(-1).view(-1)
        #return gt.to(relPred[1].device), relPred[1].view(-1)
        if len(predsPos)>0:
            predsPos = torch.cat(predsPos).to(relPred.device)
        else:
            predsPos = None
        if len(predsNeg)>0:
            predsNeg = torch.cat(predsNeg).to(relPred.device)
        else:
            predsNeg = None
        if len(adj)>0:
            recall = truePred/len(adj)
        else:
            recall = 1
        if falsePred>0:
            prec = truePred/(truePred+falsePred)
        else:
            prec = 1
        return predsPos,predsNeg, recall, prec, prec, computeAP(scores)
