import torch.nn as nn
import torch
import numpy as np
import math

class YoloLoss (nn.Module):
    def __init__(self, num_classes, rotation, scale, anchors, ignore_thresh=0.5):
        super(YoloLoss, self).__init__()
        self.ignore_thresh=ignore_thresh
        self.num_classes=num_classes
        self.rotation=rotation
        self.scale=scale
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

        grid_x = torch.arange(nW).repeat(nH, 1).view([1, 1, nH, nW]).type(FloatTensor)
        grid_y = torch.arange(nH).repeat(nW, 1).t().view([1, 1, nH, nW]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a['width'] / stride, a['height']/ stride) for a in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = torch.tanh(x.data)+0.5 + grid_x
        pred_boxes[..., 1] = torch.tanh(y.data)+0.5 + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
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
        tconf = tconf.type(FloatTensor)
        tcls = tcls.type(LongTensor)

        # Get conf mask where gt and where there is no gt
        conf_mask_true = mask
        conf_mask_false = conf_mask - mask

        # Mask outputs to ignore non-existing objects
        loss_x = self.mse_loss(x[mask], tx[mask])
        loss_y = self.mse_loss(y[mask], ty[mask])
        loss_w = self.mse_loss(w[mask], tw[mask])
        loss_h = self.mse_loss(h[mask], th[mask])
        loss_conf = 2*self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
            pred_conf[conf_mask_true], tconf[conf_mask_true]
        )
        loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        return (
            loss,
            loss_x.item()+loss_y.item()+loss_w.item()+loss_h.item(),
            loss_conf.item(),
            loss_cls.item(),
            recall,
            precision,
        )

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
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

def build_targets(
    pred_boxes, pred_conf, pred_cls, target, target_sizes, anchors, num_anchors, num_classes, grid_sizeH, grid_sizeW, ignore_thres, scale
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

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target_sizes[b]): #range(target.shape[1]):
            #if target[b, t].sum() == 0:
            #    continue
            nGT += 1
            # Convert to position relative to box
            gx = target[b, t, 0] / scale
            gy = target[b, t, 1] / scale
            gw = target[b, t, 4] / scale
            gh = target[b, t, 3] / scale
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
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
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            #target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi] = target[b, t,13:]
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == torch.argmax(target[b,t,13:]) and score > 0:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls




class YoloDistLoss (nn.Module):
    def __init__(self, num_classes, rotation, scale, anchors, ignore_thresh=0.5):
        super(YoloRotLoss, self).__init__()
        self.ignore_thresh=ignore_thresh
        self.num_classes=num_classes
        self.rotation=rotation
        self.scale=scale
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
        w = prediction[..., 4]  # Width
        h = prediction[..., 3]  # Height
        clsIdx=5
        bbParams=4
        if self.rotation:
            r = prediction[..., 5]
            clsIdx=6
            bbParams=5
        pred_conf = prediction[..., 0]  # Conf 
        pred_cls = prediction[..., clsIdx:]  # Cls pred.

        grid_x = torch.arange(nW).repeat(nH, 1).view([1, 1, nH, nW]).type(FloatTensor)
        grid_y = torch.arange(nH).repeat(nW, 1).t().view([1, 1, nH, nW]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a['width'] / stride, a['height']/ stride) for a in self.anchors])
        scaled_anchor_points = self.achor_points/stride
        scaled_anchor_hws = self.achor_hws/stride
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        #pred_boxes = FloatTensor(prediction[..., :bbParams].shape)
        #pred_boxes[..., 0] = x.data + grid_x
        #pred_boxes[..., 1] = y.data + grid_y
        #pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        #pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        #pred_boxes[..., 4] = r.data
        o_x = x + grid_x
        o_y = y.data + grid_y
        o_w = torch.exp(w) * anchor_w
        o_h = torch.exp(h) * anchor_h
        o_r = r + anchor_r

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
        pred_points = torch.stack([p_left_x,p_left_y,p_right_x,p_right_y,p_top_x,p_top_y,p_bot_x,p_bot_y],dim=2)

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
        loss_x = self.mse_loss(x[mask], tx[mask])
        loss_y = self.mse_loss(y[mask], ty[mask])
        loss_w = self.mse_loss(w[mask], tw[mask])
        loss_h = self.mse_loss(h[mask], th[mask])
        loss_r = self.mse_loss(r[mask], tr[mask])
        loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
            pred_conf[conf_mask_true], tconf[conf_mask_true]
        )
        loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        return (
            loss,
            loss_x.item()+loss_y.item()+loss_w.item()+loss_h.item(),
            loss_conf.item(),
            loss_cls.item(),
            recall,
            precision,
        )




def build_targets_rot(
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
    tconf = torch.ByteTensor(nB, nA, nH, nW).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nH, nW, nC).fill_(0)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target_sizes[b]): #range(target.shape[1]):
            #if target[b, t].sum() == 0:
            #    continue
            nGT += 1
            # Convert to position relative to box
            gx = target[b, t, 0] / scale
            gy = target[b, t, 1] / scale
            gw = target[b, t, 4] / scale
            gh = target[b, t, 3] / scale
            gr = target[b, t, 2] / scale
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_points = target[b,j,5:13] / scale
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_dists = bbox_dist(gt_points, (gt_h+gt_w)/2.0, anchor_points, anchor_hws)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_dists < ignore_thres, gj, gi] = 0
            # Find the best matching anchor box
            best_n = np.argmin(anch_dists)
            # Get ground truth box
            gt_points[[0,2,4,6]]+=gx
            gt_points[[1,3,5,7]]+=gy
            # Get the best prediction
            pred_points = pred_points[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Rotation
            tr[b, best_n, gj, gi] = gr-anchors[best_n][2]
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            #target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi] = target[b, t,13:]
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            dist = bbox_dist(gt_points, pred_points)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if dist < 0.85 and pred_label == torch.argmax(target[b,t,13:]) and score > 0.0:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


