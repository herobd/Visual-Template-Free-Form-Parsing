import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from utils.yolo_tools import allIOU, allDist

class YoloLoss (nn.Module):
    def __init__(self, num_classes, rotation, scale, anchors, ignore_thresh=0.5,use_special_loss=False,bad_conf_weight=1.25):
        super(YoloLoss, self).__init__()
        self.ignore_thresh=ignore_thresh
        self.num_classes=num_classes
        self.rotation=rotation
        self.scale=scale
        self.use_special_loss=use_special_loss
        self.bad_conf_weight=bad_conf_weight
        self.anchors=anchors
        self.num_anchors=len(anchors)
        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCEWithLogitsLoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self,prediction, target, target_sizes ):

        nA = self.num_anchors
        nB = prediction.size(0)
        nH = prediction.size(2)
        nW = prediction.size(3)
        stride=self.scale

        FloatTensor = torch.cuda.FloatTensor if prediction.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if prediction.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if prediction.is_cuda else torch.ByteTensor

        x = prediction[..., 1]  # Center x
        y = prediction[..., 2]  # Center y
        w = prediction[..., 5]  # Width
        h = prediction[..., 4]  # Height
        #r = prediction[..., 3]  # Rotation (not used here)
        pred_conf = prediction[..., 0]  # Conf 
        pred_cls = prediction[..., 6:]  # Cls pred.

        grid_x = torch.arange(nW).repeat(nH, 1).view([1, 1, nH, nW]).type(FloatTensor).to(prediction.device)
        grid_y = torch.arange(nH).repeat(nW, 1).t().view([1, 1, nH, nW]).type(FloatTensor).to(prediction.device)
        scaled_anchors = FloatTensor([(a['width'] / stride, a['height']/ stride) for a in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1)).to(prediction.device)
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1)).to(prediction.device)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = torch.tanh(x.data)+0.5 + grid_x
        pred_boxes[..., 1] = torch.tanh(y.data)+0.5 + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        if target is not None:
            target[:,:,[0,1,3,4]] /= self.scale

        nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls, distances, ious = build_targets(
            pred_boxes=pred_boxes.cpu().data,
            pred_conf=pred_conf.cpu().data,
            pred_cls=pred_cls.cpu().data,
            target=target.cpu().data if target is not None else None,
            target_sizes=target_sizes,
            anchors=scaled_anchors.cpu().data,
            num_anchors=nA,
            num_classes=self.num_classes,
            grid_sizeH=nH,
            grid_sizeW=nW,
            ignore_thres=self.ignore_thresh,
            scale=self.scale,
            calcIOUAndDist=self.use_special_loss
        )

        nProposals = int((pred_conf > 0).sum().item())
        recall = float(nCorrect / nGT) if nGT else 1
        if nProposals>0:
            precision = float(nCorrect / nProposals)
        else:
            precision = 1

        # Handle masks
        mask = (mask.type(ByteTensor))
        conf_mask = (conf_mask.type(ByteTensor))

        # Handle target variables
        tx = tx.type(FloatTensor).to(prediction.device)
        ty = ty.type(FloatTensor).to(prediction.device)
        tw = tw.type(FloatTensor).to(prediction.device)
        th = th.type(FloatTensor).to(prediction.device)
        tconf = tconf.type(FloatTensor).to(prediction.device)
        tcls = tcls.type(LongTensor).to(prediction.device)

        # Get conf mask where gt and where there is no gt
        conf_mask_true = mask
        conf_mask_false = conf_mask - mask

        #import pdb; pdb.set_trace()

        # Mask outputs to ignore non-existing objects
        if self.use_special_loss:
            loss_conf = weighted_bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false],distances[conf_mask_false],ious[conf_mask_false],nB)
            distances=None
            ious=None
        else:
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false])
        loss_conf *= self.bad_conf_weight
        if target is not None and nGT>0:
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss_conf += self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            return (
                loss,
                loss_x.item()+loss_y.item()+loss_w.item()+loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )
        else:
            return (
                loss_conf,
                0,
                loss_conf.item(),
                0,
                recall,
                precision,
            )

def weighted_bce_loss(pred,gt,distances,ious,batch_size):
    #remove any good predictions
    keep = ious<0.6
    #pred=pred[keep]
    #gt=gt[keep]
    distances=distances[keep]
    if batch_size>1:
        max_per_batch = distances.view(batch_size,-1).max(dim=1)[0][:,None,None,None]
        sum_per_batch = distances.view(batch_size,-1).sum(dim=1)[0][:,None,None,None]
        epsilon = distances.mean(dim=1)
        count_per = keep.sum(dim=1)
    else:
        max_per_batch = distances.max()
        sum_per_batch = distances.sum()
        epsilon = distances.mean()
        count_per = keep.sum()
    distance_weights = (max_per_batch-distances+epsilon)/(sum_per_batch+count_per.float()*epsilon)
    lossByBatch= distance_weights.to(pred.device)*F.binary_cross_entropy_with_logits(pred[keep],gt[keep],reduction='none')
    if batch_size>1:
        lossByBatch=lossByBatch.sum(dim=1)
    #lossByBatch= (-distance_weights*(gt*torch.log(pred) + (1-gt)*torch.log(1-pred))).sum(dim=1)
    distance_weights=None
    return lossByBatch.mean()

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        #I assume H and W are half
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] , box1[:, 0] + box1[:, 2] 
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] , box1[:, 1] + box1[:, 3] 
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] , box2[:, 0] + box2[:, 2] 
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] , box2[:, 1] + box2[:, 3] 
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def inv_tanh(y):
    if y<=-1:
        return -2
    elif y >=1:
        return 2
    return 0.5*(math.log((1+y)/(1-y)))

def build_targets(
    pred_boxes, pred_conf, pred_cls, target, target_sizes, anchors, num_anchors, num_classes, grid_sizeH, grid_sizeW, ignore_thres, scale, calcIOUAndDist=False
):
    nB = pred_boxes.size(0)
    nA = num_anchors
    nC = num_classes
    nH = grid_sizeH
    nW = grid_sizeW
    mask = torch.zeros(nB, nA, nH, nW)
    conf_mask = torch.ones(nB, nA, nH, nW)
    tx = torch.zeros(nB, nA, nH, nW)
    ty = torch.zeros(nB, nA, nH, nW)
    tw = torch.zeros(nB, nA, nH, nW)
    th = torch.zeros(nB, nA, nH, nW)
    tconf = torch.ByteTensor(nB, nA, nH, nW).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nH, nW, nC).fill_(0)
    if calcIOUAndDist:
        distances = torch.ones(nB,nA, nH, nW) #distance to closest target
        ious = torch.zeros(nB,nA, nH, nW) #max iou to target
    else:
        distances=None
        ious=None

    nGT = 0
    nCorrect = 0
    #import pdb; pdb.set_trace()
    for b in range(nB):
        if calcIOUAndDist and target_sizes[b]>0:
            flat_pred = pred_boxes[b].view(-1,pred_boxes.size(-1))
            #flat_target = target[b,:target_sizes[b]].view(-1,target.size(-1))
            iousB = allIOU(flat_pred,target[b,:target_sizes[b]], boxes1XYWH=[0,1,2,3])
            iousB = iousB.view(nA, nH, nW,-1)
            ious[b] = iousB.max(dim=-1)[0]
            distancesB = allDist(flat_pred,target[b,:target_sizes[b]])
            distances[b] = distancesB.min(dim=-1)[0].view(nA, nH, nW)
            #import pdb;pdb.set_trace()
        
        for t in range(target_sizes[b]): #range(target.shape[1]):
            #if target[b, t].sum() == 0:
            #    continue
            # Convert to position relative to box
            gx = target[b, t, 0] #/ scale
            gy = target[b, t, 1] #/ scale
            gw = target[b, t, 4] #/ scale
            gh = target[b, t, 3] #/ scale
        
            if gw==0 or gh==0:
                continue
            nGT += 1
            # Get grid box indices
            gi = max(min(int(gx),conf_mask.size(3)-1),0)
            gj = max(min(int(gy),conf_mask.size(2)-1),0)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes) #these are at half their size, but IOU is the same
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1 #why not just set this to 0?
            # Coordigates
            tx[b, best_n, gj, gi] = inv_tanh(gx - (gi+0.5))
            ty[b, best_n, gj, gi] = inv_tanh(gy - (gj+0.5))
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            #target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi] = target[b, t,13:]
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            if calcIOUAndDist:
                #iou = ious[best_n*(nH*nW) + gj*(nW) + gi,t]
                iou = iousB[best_n, gj, gi, t]
            else:
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            #import pdb; pdb.set_trace()
            if iou > 0.5 and pred_label == torch.argmax(target[b,t,13:]) and score > 0:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls, distances, ious




class YoloDistLoss (nn.Module):
    def __init__(self, num_classes, rotation, scale, anchors, ignore_thresh=0.5,bad_conf_weight=1.25):
        super(YoloDistLoss, self).__init__()
        self.ignore_thresh=ignore_thresh
        self.num_classes=num_classes
        self.rotation=rotation
        self.scale=scale
        self.bad_conf_weight=bad_conf_weight
        self.anchors=anchors
        self.num_anchors=len(anchors)
        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCEWithLogitsLoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

        #make anchor points from anchors
        o_r = torch.FloatTensor([a['rot'] for a in anchors])
        o_h = torch.FloatTensor([a['height'] for a in anchors])
        o_w = torch.FloatTensor([a['width'] for a in anchors])
        cos_rot = torch.cos(o_r)
        sin_rot = torch.sin(o_r)
        p_left_x =  -cos_rot*o_w
        p_left_y =  -sin_rot*o_w
        p_right_x = cos_rot*o_w
        p_right_y = sin_rot*o_w
        p_top_x =   sin_rot*o_h
        p_top_y =   -cos_rot*o_h
        p_bot_x =   -sin_rot*o_h
        p_bot_y =   cos_rot*o_h
        self.anchor_points=torch.stack([p_left_x,p_left_y,p_right_x,p_right_y,p_top_x,p_top_y,p_bot_x,p_bot_y],dim=1)
        self.anchor_hws= (o_h+o_w)/2.0

    def forward(self,prediction, target, target_sizes ):

        nA = self.num_anchors
        nB = prediction.size(0)
        nH = prediction.size(2)
        nW = prediction.size(3)
        stride=self.scale

        FloatTensor = torch.cuda.FloatTensor if prediction.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if prediction.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if prediction.is_cuda else torch.ByteTensor

        x = prediction[..., 1]  # Center x
        y = prediction[..., 2]  # Center y
        w = prediction[..., 5]  # Width
        h = prediction[..., 4]  # Height
        r = prediction[..., 3]  # Rotation
        pred_conf = prediction[..., 0]  # Conf 
        pred_cls = prediction[..., 6:]  # Cls pred.

        grid_x = torch.arange(nW).repeat(nH, 1).view([1, 1, nH, nW]).type(FloatTensor)
        grid_y = torch.arange(nH).repeat(nW, 1).t().view([1, 1, nH, nW]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a['width'] / stride, a['height']/ stride, a['rot']) for a in self.anchors])
        scaled_anchor_points = self.anchor_points/stride
        scaled_anchor_hws = self.anchor_hws/stride
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1)).to(prediction.device)
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1)).to(prediction.device)
        anchor_r = scaled_anchors[:, 2:3].view((1, nA, 1, 1)).to(prediction.device)

        # Add offset and scale with anchors
        #pred_boxes = FloatTensor(prediction[..., :bbParams].shape)
        #pred_boxes[..., 0] = x.data + grid_x
        #pred_boxes[..., 1] = y.data + grid_y
        #pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        #pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        #pred_boxes[..., 4] = r.data

        #Create points from predicted boxes
        o_x = torch.tanh(x)+0.5 + grid_x
        o_y = torch.tanh(y)+0.5 + grid_y
        o_w = torch.exp(w) * anchor_w
        o_h = torch.exp(h) * anchor_h
        o_r =  (math.pi/2)*torch.tanh(r) + anchor_r

        cos_rot = torch.cos(o_r)
        sin_rot = torch.sin(o_r)
        p_left_x = o_x-cos_rot*o_w
        p_left_y = o_y-sin_rot*o_w
        p_right_x = o_x+cos_rot*o_w
        p_right_y = o_y+sin_rot*o_w
        p_top_x = o_x+sin_rot*o_h
        p_top_y = o_y-cos_rot*o_h
        p_bot_x = o_x-sin_rot*o_h
        p_bot_y = o_y+cos_rot*o_h
        pred_points = torch.stack([p_left_x,p_left_y,p_right_x,p_right_y,p_top_x,p_top_y,p_bot_x,p_bot_y],dim=4)

        if target is not None:
            target[:,:,[0,1,3,4]] /= self.scale
            target[:,:,5:13] /= self.scale

        nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tr, tconf, tcls = build_targets_dist(
            pred_points=pred_points.cpu().data,
            pred_hws=((o_h+o_w)/2.0).cpu().data,
            pred_conf=pred_conf.cpu().data,
            pred_cls=pred_cls.cpu().data,
            target=target.cpu().data if target is not None else None,
            target_sizes=target_sizes,
            anchors=scaled_anchors.cpu().data,
            anchor_points=scaled_anchor_points.cpu().data,
            anchor_hws=scaled_anchor_hws.cpu().data,
            num_anchors=nA,
            num_classes=self.num_classes,
            grid_sizeH=nH,
            grid_sizeW=nW,
            ignore_thres=self.ignore_thresh,
            scale=self.scale
        )

        nProposals = int((pred_conf > 0).sum().item())
        recall = float(nCorrect / nGT) if nGT else 1
        if nProposals>0:
            precision = float(nCorrect / nProposals)
        else:
            precision = 1

        # Handle masks
        mask = (mask.type(ByteTensor))
        conf_mask = (conf_mask.type(ByteTensor))

        # Handle target variables
        tx = tx.type(FloatTensor)
        ty = ty.type(FloatTensor)
        tw = tw.type(FloatTensor)
        th = th.type(FloatTensor)
        tr = tr.type(FloatTensor)
        tconf = tconf.type(FloatTensor)
        tcls = tcls.type(LongTensor)

        # Get conf mask where gt and where there is no gt
        conf_mask_true = mask
        conf_mask_false = conf_mask - mask

        # Mask outputs to ignore non-existing objects
        loss_conf = self.bad_conf_weight*self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false])
        if target is not None and nGT>0:
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_r = self.mse_loss(r[mask], tr[mask])
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss_conf += self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
            loss = loss_x + loss_y + loss_w + loss_h + loss_r + loss_conf + loss_cls
            return (
                loss,
                loss_x.item()+loss_y.item()+loss_w.item()+loss_h.item()+loss_r.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )
        else:
            return (
                loss_conf,
                0,
                loss_conf.item(),
                0,
                recall,
                precision,
            )



def build_targets_dist(
    pred_points, pred_hws, pred_conf, pred_cls, target, target_sizes, anchors, anchor_points, anchor_hws, num_anchors, num_classes, grid_sizeH, grid_sizeW, ignore_thres, scale
):
    nB = pred_points.size(0)
    nA = num_anchors
    nC = num_classes
    nH = grid_sizeH
    nW = grid_sizeW
    mask = torch.zeros(nB, nA, nH, nW)
    conf_mask = torch.ones(nB, nA, nH, nW)
    tx = torch.zeros(nB, nA, nH, nW)
    ty = torch.zeros(nB, nA, nH, nW)
    tw = torch.zeros(nB, nA, nH, nW)
    th = torch.zeros(nB, nA, nH, nW)
    tr = torch.zeros(nB, nA, nH, nW)
    tconf = torch.ByteTensor(nB, nA, nH, nW).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nH, nW, nC).fill_(0)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target_sizes[b]): #range(target.shape[1]):
            #if target[b, t].sum() == 0:
            #    continue

            # Convert to position relative to box
            gx = target[b, t, 0] #/ scale
            gy = target[b, t, 1] #/ scale
            gw = target[b, t, 4] #/ scale
            gh = target[b, t, 3] #/ scale
            gr = target[b, t, 2]
            if gw==0 or gh==0:
                continue
            nGT += 1
            # Get grid box indices
            gi = max(min(int(gx),conf_mask.size(3)-1),0)
            gj = max(min(int(gy),conf_mask.size(2)-1),0)
            # Get shape of gt box
            gt_points = target[b,t,5:13]
            gt_points[[0,2,4,6]]-=gx #center the points about the origin instead of BB location
            gt_points[[1,3,5,7]]-=gy
            # Get shape of anchor box
            #anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_dists = bbox_dist(gt_points, (gh+gw)/2.0, anchor_points, anchor_hws)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_dists < ignore_thres, gj, gi] = 0
            # Find the best matching anchor box
            best_n = np.argmin(anch_dists)
            # Get ground truth box
            gt_points = target[b,t,5:13] #/ scale
            #gt_points[[0,2,4,6]]+=gx
            #gt_points[[1,3,5,7]]+=gy
            # Get the best prediction
            pred_point = pred_points[b, best_n, gj, gi]#.unsqueeze(0)
            pred_hw = pred_hws[b, best_n, gj, gi]#.unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = inv_tanh(gx - (gi+0.5))
            ty[b, best_n, gj, gi] = inv_tanh(gy - (gj+0.5))
            # Rotation
            rot_diff = gr-anchors[best_n][2]
            if rot_diff>math.pi:
                rot_diff-=2*math.pi
            elif rot_diff<-math.pi:
                rot_diff+=2*math.pi
            tr[b, best_n, gj, gi] = inv_tanh(rot_diff/(math.pi/2))
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            #target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi] = target[b, t,13:]
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            dist = bbox_dist(gt_points, (gh+gw)/2.0, pred_point, pred_hw)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if dist < 0.85 and pred_label == torch.argmax(target[b,t,13:]) and score > 0.0:
                nCorrect += 1
    #nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tr, tconf, tcls
    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tr, tconf, tcls


def bbox_dist(box1, box1H, box2, box2H):
    """
    Returns the point distance of bounding boxes
    the boxes are [leftX,Y,rightX,Y,topX,Y,botX,Y]
    """
    if len(box2.size())>1 or len(box1.size())>1:
        if len(box1.size())==1:
            box1=box1[None,:]
            box1H=torch.tensor([box1H])
            flat1=True
        else:
            flat1=False
        if len(box2.size())==1:
            box2=box2[None,:]
            box2H=torch.tensor([box2H])
            flat2=True
        else:
            flat2=False
        expanded1 = box1[:,None,:].expand(box1.size(0),box2.size(0),8)
        expanded1H = box1H[:,None].expand(box1.size(0),box2.size(0))
        expanded2 = box2[None,:,:].expand(box1.size(0),box2.size(0),8)
        expanded2H = box2H[None,:].expand(box1.size(0),box2.size(0))

        normalization = (expanded1H+expanded2H)/2.0

        deltas = expanded1-expanded2
        dist = ((
                torch.norm(deltas[:,:,0:2],2,2) +
                torch.norm(deltas[:,:,0:2],2,2) +
                torch.norm(deltas[:,:,0:2],2,2) +
                torch.norm(deltas[:,:,0:2],2,2) 
               )/normalization)**2
        if flat1:
            assert(dist.size(0)==1)
            dist=dist[0]
        if flat2:
            if flat1:
                assert(dist.size(0)==1)
                dist=dist[0]
            else:
                assert(dist.size(1)==1)
                dist=dist[:,0]
    else:
        diff = box1-box2
        normalizer = (box1H+box2H)/2.0
        dist = ((torch.norm(diff[0:2])+torch.norm(diff[2:4])+torch.norm(diff[4:6])+torch.norm(diff[6:8]))/normalizer)**2
    return dist





class LineLoss (nn.Module):
    def __init__(self, num_classes, scale,  anchor_h,bad_conf_weight=1.25):
        super(YoloDistLoss, self).__init__()
        self.ignore_thresh=ignore_thresh
        self.num_classes=num_classes
        self.scale=scale
        self.bad_conf_weight=bad_conf_weight
        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCEWithLogitsLoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss
        self.anchor_h = anchor_h/scale

    def forward(self,prediction, target, target_sizes ):

        nB = prediction.size(0)
        nH = prediction.size(2)
        nW = prediction.size(3)
        stride=self.scale

        FloatTensor = torch.cuda.FloatTensor if prediction.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if prediction.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if prediction.is_cuda else torch.ByteTensor

        x = prediction[..., 1]  # Center x
        y = prediction[..., 2]  # Center y
        h = prediction[..., 4]  # Height
        r = prediction[..., 3]  # Rotation
        pred_conf = prediction[..., 0]  # Conf 
        pred_cls = prediction[..., 5:]  # Cls pred.

        grid_x = torch.arange(nW).repeat(nH, 1).view([1, nH, nW]).type(FloatTensor)
        grid_y = torch.arange(nH).repeat(nW, 1).t().view([1, nH, nW]).type(FloatTensor)

        #Create points from predicted boxes
        o_x = torch.tanh(x)+0.5 + grid_x
        o_y = torch.tanh(y)+0.5 + grid_y
        o_h = torch.exp(h) * self.anchor_h #half
        o_r =  (math.pi/2)*torch.tanh(r)

        x1 = -o_h*torch.sin(math.pi-o_r) + o_x
        y1 = o_h*torch.cos(math.pi-o_r) + o_y
        x2 = o_h*torch.sin(math.pi-o_r) + o_x
        y2 = -o_h*torch.cos(math.pi-o_r) + o_y

        pred = torch.stack([o_x,o_y,o_r,o_h],dim=3)

        if target is not None: #target is x1,y1,x2,y2
            target /= self.scale

        nGT, mask, conf_mask, tx1, ty1, tx2, ty2, tconf, tcls = self.build_targets_lines(
            pred=pred.cpu().data,
            pred_conf=pred_conf.cpu().data,
            pred_cls=pred_cls.cpu().data,
            target=target.cpu().data if target is not None else None,
            target_sizes=target_sizes,
            num_classes=self.num_classes,
            grid_sizeH=nH,
            grid_sizeW=nW,
            ignore_thres=self.ignore_thresh,
            scale=self.scale
        )

        #nProposals = int((pred_conf > 0).sum().item())
        #recall = float(nCorrect / nGT) if nGT else 1
        #if nProposals>0:
        #    precision = float(nCorrect / nProposals)
        #else:
        #    precision = 1

        # Handle masks
        mask = (mask.type(ByteTensor))
        conf_mask = (conf_mask.type(ByteTensor))

        # Handle target variables
        tx1 = tx1.type(FloatTensor)
        ty1 = ty1.type(FloatTensor)
        tx1 = tx1.type(FloatTensor)
        ty1 = ty1.type(FloatTensor)
        tconf = tconf.type(FloatTensor)
        tcls = tcls.type(LongTensor)

        # Get conf mask where gt and where there is no gt
        conf_mask_true = mask
        conf_mask_false = conf_mask - mask

        # Mask outputs to ignore non-existing objects
        loss_conf = self.bad_conf_weight*self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false])
        if target is not None and nGT>0:
            loss_x1 = self.mse_loss(x1[mask], tx1[mask])
            loss_y1 = self.mse_loss(y1[mask], ty1[mask])
            loss_x2 = self.mse_loss(x2[mask], tx2[mask])
            loss_y2 = self.mse_loss(y2[mask], ty2[mask])
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss_conf += self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
            loss = loss_x1 + loss_y1 + loss_x2 + loss_y2 + loss_conf + loss_cls
            return (
                loss,
                loss_x1.item()+loss_y1.item()+loss_x2.item()+loss_y2.item(),
                loss_conf.item(),
                loss_cls.item(),
                #recall,
                #precision,
            )
        else:
            return (
                loss_conf,
                0,
                loss_conf.item(),
                0,
                #recall,
                #precision,
            )



    def build_targets_lines(self,
        pred, pred_conf, pred_cls, target, target_sizes, grid_sizeH, grid_sizeW
    ):
        nB = pred_points.size(0)
        nC = self.num_classes
        nH = grid_sizeH
        nW = grid_sizeW
        mask = torch.zeros(nB, nH, nW)
        conf_mask = torch.ones(nB, nH, nW)
        tx = torch.zeros(nB, nH, nW)
        ty = torch.zeros(nB, nH, nW)
        th = torch.zeros(nB, nH, nW)
        tr = torch.zeros(nB, nH, nW)
        tconf = torch.ByteTensor(nB, nH, nW).fill_(0)
        tcls = torch.ByteTensor(nB, nH, nW, nC).fill_(0)

        nGT = 0
        for b in range(nB):
            for t in range(target_sizes[b]): #range(target.shape[1]):
                #if target[b, t].sum() == 0:
                #    continue

                # Convert to position relative to box
                gx1 = target[b, t, 0] 
                gy1 = target[b, t, 1]
                gx2 = target[b, t, 2] 
                gy2 = target[b, t, 3]
                gx = (gx1+gx2)/2.0
                gh = (gy1+gy2)/2.0
                #if gh==0:
                #    continue
                nGT += 1
                # Get grid box indices
                gi = max(min(int(gx),conf_mask.size(3)-1),0)
                gj = max(min(int(gy),conf_mask.size(2)-1),0)
                # Masks
                mask[b, gj, gi] = 1
                conf_mask[b, gj, gi] = 1
                # Coordinates
                tx1[b, gj, gi] = gx1 - (gi+0.5) #inv_tanh(gx1 - (gi+0.5))
                ty1[b, gj, gi] = gy1 - (gj+0.5) #inv_tanh(gy1 - (gj+0.5))
                tx2[b, gj, gi] = gx2 - (gi+0.5) #inv_tanh(gx2 - (gi+0.5))
                ty2[b, gj, gi] = gy2 - (gj+0.5) #inv_tanh(gy2 - (gj+0.5))
                # One-hot encoding of label
                #target_label = int(target[b, t, 0])
                tcls[b, gj, gi] = target[b, t,5:]
                tconf[b, gj, gi] = 1

                # Calculate iou between ground truth and best matching prediction
                #dist = bbox_dist(gt_points, (gh+gw)/2.0, pred_point, pred_hw)
                #dist = 
                #pred_label = torch.argmax(pred_cls[b, gj, gi])
                #score = pred_conf[b, gj, gi]
                #if dist < 0.85 and pred_label == torch.argmax(target[b,t,13:]) and score > 0.0:
                #    nCorrect += 1
        #nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tr, tconf, tcls
        return nGT, mask, conf_mask, tx1, ty1, tx2, ty2, tconf, tcls
