import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
import timeit
from random import shuffle


def greedy_assignment(scores):
    target_ind=list(range(scores.shape[1]))
    location_ind=min_indexes = np.argmin(scores,axis=0)
    targets = list(range(scores.shape[1]))
    shuffle(targets) #shuffle so there isn't any bias to the greediness
    used = np.zeros(scores.shape[0])
    for target_i in targets:
        if used[location_ind[target_i]]: #has this location been taken already?
            scores[location_ind[target_i],target_i]=np.inf #dont find this as the min
            location_ind[target_i] = np.argmin(scores[:,target_i])
        used[location_ind[target_i]]=1
    return location_ind, target_ind

def alignment_loss(predictions, target, label_sizes, alpha_alignment=1000.0, alpha_backprop=100.0, return_alignment=False, debug=None, points=False, allow_greedy_speedup=True):
    batch_size = predictions.size(0)
    # This should probably be computed using the log_softmax
    confidences = predictions[:,:,0]
    log_one_minus_confidences = torch.log(1.0 - confidences + 1e-10)

    if target is None:
        if return_alignment:
            return -log_one_minus_confidences.sum(), None, None
        return -log_one_minus_confidences.sum()
    
    if points:
        locations = predictions[:,:,1:3]
        target = target[:,:,0:2]
    else:
        locations = predictions[:,:,1:5]
        target = target[:,:,0:4]
    #print('loc {},   tar {}'.format(locations.shape,target.shape))
    #tic=timeit.default_timer()

    log_confidences = torch.log(confidences + 1e-10)

    expanded_locations = locations[:,:,None,:]
    expanded_target = target[:,None,:,:]

    expanded_locations = expanded_locations.expand(locations.size(0), locations.size(1), target.size(1), locations.size(2))
    expanded_target = expanded_target.expand(target.size(0), locations.size(1), target.size(1), target.size(2))

    #Compute All Deltas
    location_deltas = (expanded_locations - expanded_target)

    ious = torch.norm(location_deltas,2,3)**2
    expanded_log_confidences = log_confidences[:,:,None].expand(locations.size(0), locations.size(1), target.size(1))
    expanded_log_one_minus_confidences = log_one_minus_confidences[:,:,None].expand(locations.size(0), locations.size(1), target.size(1))

    C = alpha_alignment/2.0 * ious - expanded_log_confidences + expanded_log_one_minus_confidences

    C = C.data.cpu().numpy()
    X = np.zeros_like(C)
    #print(' pre-batch comp: {}'.format(timeit.default_timer()-tic))
    if return_alignment:
        target_ind_bs=[]
        location_ind_bs=[]
    for b in range(C.shape[0]):
        l = label_sizes[b]
        if l == 0:
            if return_alignment:
                target_ind_bs.append([])
                location_ind_bs.append([])
            continue

        C_i = C[b,:,:l]
        #isnan_ = np.isnan(C_i)
        #if isnan_.any():
        #    print('NaN! {}'.format(C_i.shape))
        #    maxV=C_i[np.logical_not(isnan_)].max()
        #    C_i[isnan_]=maxV
        #tic=timeit.default_timer()
        if debug is None:
            if allow_greedy_speedup and l > 200:
                location_ind, target_ind = greedy_assignment(C_i)
            else:
                target_ind, location_ind = linear_sum_assignment(C_i.T)
        elif debug:
            location_ind, target_ind = greedy_assignment(C_i)
        #print(' batch {} of size {} linear_sum_assign: {}'.format(b,l,timeit.default_timer()-tic))
        X[b][(location_ind, target_ind)] = 1.0
        if return_alignment:
            target_ind_bs.append(target_ind)
            location_ind_bs.append(location_ind)
        #if debug is not None and b==debug:
        #    for i in range(locations.size(1)):
        #        print('loc{}: {}, {},  dist={}'.format(i,locations[b,i,0],locations[b,i,1],ious[b,i,0]))
        #    for i in range(len(target_ind)):
        #        targ_i = target_ind[i]
        #        loc_i = location_ind[i]
        #        print('size locations={}, size target={}'.format(locations.size(1), l))
        #        print('targ_i={}, loc_i={}'.format(targ_i,loc_i))
        #        print('gt location={}, {}'.format(target[b,targ_i,0],target[b,targ_i,1]))
        #        print('aligned C={}, ious={}, expanded_log_confidences={}, expanded_log_one_minus_confidences={}'.format(C_i[loc_i,targ_i],ious[b,loc_i,targ_i],expanded_log_confidences[b,loc_i,targ_i],expanded_log_one_minus_confidences[b,loc_i,targ_i]))
        #        print('aligned pred location={}, {}'.format(locations[b,loc_i,0],locations[b,loc_i,1]))

        #        closest_loc_i = ious[b,:,targ_i].argmin()
        #        print('closest C={}, ious={}, expanded_log_confidences={}, expanded_log_one_minus_confidences={}'.format(C_i[closest_loc_i,targ_i],ious[b,closest_loc_i,targ_i],expanded_log_confidences[b,closest_loc_i,targ_i],expanded_log_one_minus_confidences[b,closest_loc_i,targ_i]))
        #        print('closest pred location={}, {}'.format(locations[b,closest_loc_i,0],locations[b,closest_loc_i,1]))
        #        exit()


    X = torch.from_numpy(X).type(predictions.data.type())
    X2 = 1.0 - torch.sum(X, 2)

    location_loss = (alpha_backprop/2.0 * ious * X).sum()
    confidence_loss =  -(expanded_log_confidences * X).sum() - (log_one_minus_confidences * X2).sum()

    loss = confidence_loss + location_loss

    loss = loss/batch_size

    if return_alignment:
        #for b in range(C.shape[0]):
        #    print('alignemnt')
        #    for i in range(len(target_ind_bs[b])):
        #        print('b={}, i={}, lenlocation={}, lentarget={}, lenlocation[b]={}, lentarget[b]={}'.format(b,i,len(location_ind_bs),len(target_ind_bs),len(location_ind_bs[b]),len(target_ind_bs[b])))
        #        print('location_ind_bs[b][i]={}, target_ind_bs[b][i]={}, targetsize={}, locationsize={}'.format(location_ind_bs[b][i],target_ind_bs[b][i],target.size(1),locations.size(1)))
        #        print(' gt:{},{}\tpred:{},{}'.format(locations[b,location_ind_bs[b][i],0],
        #                                             locations[b,location_ind_bs[b][i],1],
        #                                             target[b,target_ind_bs[b][i],0],
        #                                             target[b,target_ind_bs[b][i],1]))
        return loss, location_ind_bs, target_ind_bs
    return loss

def alignment_loss_points(predictions, target, label_sizes, alpha_alignment=1000.0, alpha_backprop=100.0, return_alignment=False, debug=None):
    return alignment_loss(predictions, target, label_sizes, alpha_alignment,alpha_backprop,return_alignment,debug,points=True)




def box_alignment_loss(predictions, target, target_sizes, num_anchors, ignore_thresh=9.0, return_alignment=False, debug=None, use_point_loss=False, bias_long_side=False):
    batch_size = predictions.size(0)
    #collapse different anchors together
    
    # This should probably be computed using the log_softmax
    confidences = predictions[:,:,0]
    #log_one_minus_confidences = torch.log(1.0 - confidences + 1e-10) 

    if target is None:
        loss = F.binary_cross_entropy_with_logits(confidences,torch.zeros_like(confidences),size_average=False,reduce=True)
        loss /= confidences.size(0)*confidences.size(1)/num_anchors
        if return_alignment:
            return loss, None, None
        return loss
    #0:conf, 1:xc, 2:yx, 3:rot, 4:h, 5:w
    #pred_box = predictions[:,:,1:6].view(batch_size,-1,5)
    cos_rot = torch.cos(predictions[:,:,3])
    sin_rot = torch.sin(predictions[:,:,3])
    p_left_x = predictions[:,:,1]-cos_rot*predictions[:,:,5]
    p_left_y = predictions[:,:,2]-sin_rot*predictions[:,:,5]
    p_right_x = predictions[:,:,1]+cos_rot*predictions[:,:,5]
    p_right_y = predictions[:,:,2]+sin_rot*predictions[:,:,5]
    p_top_x = predictions[:,:,1]+sin_rot*predictions[:,:,4]
    p_top_y = predictions[:,:,2]-cos_rot*predictions[:,:,4]
    p_bot_x = predictions[:,:,1]-sin_rot*predictions[:,:,4]
    p_bot_y = predictions[:,:,2]+cos_rot*predictions[:,:,4]

    #pred_left = torch.stack([p_left_x,p_left_y],dim=2)
    pred_points = torch.stack([p_left_x,p_left_y,p_right_x,p_right_y,p_top_x,p_top_y,p_bot_x,p_bot_y],dim=2)
    pred_heights = predictions[:,:,4]
    pred_widths = predictions[:,:,5]
    pred_classes = predictions[:,:,6:]

    #target_points_left = target[:,:,5:7] #pre-computed points
    #target_points_right = target[:,:,7:9] #pre-computed points
    target_points = target[:,:,5:13] #pre-computed points
    target_box = target[:,:,0:5]
    target_classes = target[:,:,13:]
    target_heights = target[:,:,3]
    target_widths = target[:,:,4]
    #print('loc {},   tar {}'.format(locations.shape,target.shape))
    #tic=timeit.default_timer()

    log_confidences = torch.log(confidences + 1e-10)

    expanded_pred_points = pred_points[:,:,None]
    expanded_pred_heights = pred_heights[:,:,None]
    expanded_pred_widths = pred_widths[:,:,None]

    expanded_target_points = target_points[:,None,:]
    expanded_target_heights = target_heights[:,None,:]
    expanded_target_widths = target_widths[:,None,:]   

    expanded_pred_points = expanded_pred_points.expand(pred_points.size(0), pred_points.size(1), target_points.size(1), pred_points.size(2))
    expanded_pred_heights = expanded_pred_heights.expand(pred_heights.size(0), pred_heights.size(1), target_heights.size(1))
    expanded_pred_widths = expanded_pred_widths.expand(pred_widths.size(0), pred_widths.size(1), target_widths.size(1))
    expanded_target_points = expanded_target_points.expand(target_points.size(0), pred_points.size(1), target_points.size(1), target_points.size(2))
    expanded_target_heights = expanded_target_heights.expand(target_heights.size(0), pred_heights.size(1), target_heights.size(1))
    expanded_target_widths = expanded_target_widths.expand(target_widths.size(0), pred_widths.size(1), target_widths.size(1))

    #Compute All Deltas
    point_deltas = (expanded_pred_points - expanded_target_points)
    if bias_long_side:
        norm_heights = ((expanded_target_heights+expanded_pred_heights)/2)
        norm_widths = ((expanded_target_widths+expanded_pred_widths)/2)
    else:
        norm_heights=norm_widths = (expanded_target_heights+expanded_pred_heights+expanded_target_widths+expanded_pred_widths)/4

    distances = (
            torch.norm(point_deltas[:,:,:,0:2],2,3)/norm_widths +
            torch.norm(point_deltas[:,:,:,2:4],2,3)/norm_widths +
            torch.norm(point_deltas[:,:,:,4:6],2,3)/norm_heights +
            torch.norm(point_deltas[:,:,:,6:8],2,3)/norm_heights 
            )**2

    #we need to produce an error on conf of all pred that don't intersect targets significantly
    above_thresh = (distances>ignore_thresh) #candidates

    if return_alignment:
        bests=[]

    #for b in range(batch_size):
    #    distances[b,target_sizes
    total_targets=0
    for b in range(batch_size): #by batch, has they have different numbers of targets
        if target_sizes[b]>0:
            total_targets+=target_sizes[b]
            minVs,best = distances[b,:,:target_sizes[b]].min(0) #this leaves index of pred for each target
            if return_alignment:
                bests.append(best)
            #box_loss = distances[:,best,range].sum()
            #batchInd = torch.arange(batch_size)[:,None].expand(batch_size,best.size(1)) #so we can use best to index in
            if use_point_loss:
                box_loss = minVs.sum()
            else:
                #import pdb; pdb.set_trace()
                #TODO loss using deactivated values
                #This will require haveing the model keep the anchors channel
                box_loss = F.mse_loss(predictions[b,best,1:6],target_box[b,:target_sizes[b],0:5],size_average=False,reduce=True) #skip conf and classes. this does error between activated  offsets/scaled of achors, not actual
            #import pdb; pdb.set_trace()
            cor_conf_loss = F.binary_cross_entropy_with_logits(confidences[b,best],torch.ones_like(confidences[b,best]),size_average=False,reduce=True) #here's conf
            box_loss += cor_conf_loss
            box_loss += F.binary_cross_entropy_with_logits(pred_classes[b,best],target_classes[b,:target_sizes[b]],size_average=False,reduce=True) #yolov3 doesnt use softmax

            above_thresh[b,best]=0  #0 out the whole vector for preds we selected earlier
            above_thresh[b,:,target_sizes[b]:]=1 #your not too close to targets that don't exist
        elif return_alignment:
            bests.append(None)
    #minVs,best = distances.min(1) #this leaves index of pred for each target
    ##box_loss = distances[:,best,range].sum()
    #batchInd = torch.arange(batch_size)[:,None].expand(batch_size,best.size(1)) #so we can use best to index in
    #if use_point_loss:
    #    box_loss = minVs.sum()
    ##TODO loss using deactivated values
    #else:
    #    box_loss = F.mse_loss(predictions[batchInd,best,1:6],target_box,size_average=False,reduce=True) #skip conf and classes. this does error between activated  offsets/scaled of achors, not actual
    #box_loss += F.binary_cross_entropy(confidences[batchInd,best],1,size_average=False,reduce=True) #here's conf
    #box_loss += F.binary_cross_entropy(pred_classes[batchInd,best],target_classes,size_average=False,reduce=True) #yolov3 doesnt use softmax

    #zero best
    #above_thresh[batchInd,best]=0 #0 out the whole vector for preds we selected earlier
    above_thresh,_ = above_thresh.min(dim=2) #if a pred is over threshold for all targets
    above_thresh=above_thresh.float()
    
    #tell these they should have 0 conf. Everything is zero-ed, so it will produce no error.
    conf_loss = F.binary_cross_entropy_with_logits(confidences*above_thresh,torch.zeros_like(above_thresh),size_average=False,reduce=True)

    #import pdb; pdb.set_trace()
    conf_loss /= (above_thresh.sum()/num_anchors) #normalize
    box_loss /= total_targets


    loss = (conf_loss+box_loss)/batch_size


    if return_alignment:
        return loss, bests
    return loss#, cor_conf_loss, conf_loss, (confidences*above_thresh).sum()/above_thresh.sum(), confidences[0,best]





def iou_alignment_loss(predictions, target, target_sizes, num_anchors, ignore_thresh=0.5, return_alignment=False, debug=None):
    batch_size = predictions.size(0)
    #collapse different anchors together
    
    # This should probably be computed using the log_softmax
    confidences = predictions[:,:,0]
    #log_one_minus_confidences = torch.log(1.0 - confidences + 1e-10) 

    if target is None:
        loss = F.binary_cross_entropy_with_logits(confidences,torch.zeros_like(confidences),size_average=False,reduce=True)
        loss /= confidences.size(0)*confidences.size(1)/num_anchors
        if return_alignment:
            return loss, None, None
        return loss
    #0:conf, 1:xc, 2:yx, 3:rot, 4:h, 5:w
    #pred_box = predictions[:,:,1:6].view(batch_size,-1,5)
    cos_rot = torch.cos(predictions[:,:,3])
    sin_rot = torch.sin(predictions[:,:,3])
    p_left_x = predictions[:,:,1]-cos_rot*predictions[:,:,5]
    p_left_y = predictions[:,:,2]-sin_rot*predictions[:,:,5]
    p_right_x = predictions[:,:,1]+cos_rot*predictions[:,:,5]
    p_right_y = predictions[:,:,2]+sin_rot*predictions[:,:,5]
    p_top_x = predictions[:,:,1]+sin_rot*predictions[:,:,4]
    p_top_y = predictions[:,:,2]-cos_rot*predictions[:,:,4]
    p_bot_x = predictions[:,:,1]-sin_rot*predictions[:,:,4]
    p_bot_y = predictions[:,:,2]+cos_rot*predictions[:,:,4]

    p_left = predictions[:,:,1]-predictions[:,:,5]
    p_right = predictions[:,:,1]+predictions[:,:,5]
    p_top = predictions[:,:,2]-predictions[:,:,4]
    p_bot = predictions[:,:,2]+predictions[:,:,4]

    pred_classes = predictions[:,:,6:]

    #target_points_left = target[:,:,5:7] #pre-computed points
    #target_points_right = target[:,:,7:9] #pre-computed points
    t_left = target[:,:,0]-target[:,:,4]
    t_right = target[:,:,0]+target[:,:,4]
    t_top = target[:,:,1]-target[:,:,3]
    t_bot = target[:,:,1]+target[:,:,3]
    target_classes = target[:,:,13:]
    #print('loc {},   tar {}'.format(locations.shape,target.shape))
    #tic=timeit.default_timer()

    log_confidences = torch.log(confidences + 1e-10)

    pred_x1 = p_left[:,:,None]
    pred_x1 = pred_x1.expand(batch_size, predictions.size(1), target.size(1))
    targ_x1 = t_left[:,None,:]
    targ_x1 = targ_x1.expand(batch_size, predictions.size(1), target.size(1))

    pred_x2 = p_right[:,:,None]
    pred_x2 = pred_x2.expand(batch_size, predictions.size(1), target.size(1))
    targ_x2 = t_right[:,None,:]
    targ_x2 = targ_x2.expand(batch_size, predictions.size(1), target.size(1))

    pred_y1 = p_top[:,:,None]
    pred_y1 = pred_y1.expand(batch_size, predictions.size(1), target.size(1))
    targ_y1 = t_top[:,None,:]
    targ_y1 = targ_y1.expand(batch_size, predictions.size(1), target.size(1))

    pred_y2 = p_bot[:,:,None]
    pred_y2 = pred_y2.expand(batch_size, predictions.size(1), target.size(1))
    targ_y2 = t_bot[:,None,:]
    targ_y2 = targ_y2.expand(batch_size, predictions.size(1), target.size(1))


    middle_right = -1000*torch.ones_like(targ_y2)
    middle_right = torch.where( (targ_x1<pred_x2)*(pred_x2<targ_x2), pred_x2, middle_right)
    middle_right = torch.where( (pred_x1<targ_x2)*(targ_x2<pred_x2), targ_x2, middle_right)

    middle_left = -1000*torch.ones_like(targ_x1)
    middle_left = torch.where( (targ_x1<pred_x1)*(pred_x1<targ_x2), pred_x1, middle_left)
    middle_left = torch.where( (pred_x1<targ_x1)*(targ_x1<pred_x2), targ_x1, middle_left)

    middle_bot = -1000*torch.ones_like(targ_y2)
    middle_bot = torch.where( (targ_y1<pred_y2)*(pred_y2<targ_y2), pred_y2, middle_bot)
    middle_bot = torch.where( (pred_y1<targ_y2)*(targ_y2<pred_y2), targ_y2, middle_bot)

    middle_top = -1000*torch.ones_like(targ_y1)
    middle_top = torch.where( (targ_y1<pred_y1)*(pred_y1<targ_y2), pred_y1, middle_top)
    middle_top = torch.where( (pred_y1<targ_y1)*(targ_y1<pred_y2), targ_y1, middle_top)

    inter_h = middle_bot-middle_top
    inter_w = middle_right-middle_left
    intersections = inter_h*inter_w
    sum_areas = (targ_x2-targ_x1)*(targ_y2-targ_y1) + (pred_x2-pred_x1)*(pred_y2-pred_y1)
    ious = intersections / (sum_areas-intersections)

    #we need to produce an error on conf of all pred that don't intersect targets significantly
    below_thresh = (ious<ignore_thresh) #candidates

    if return_alignment:
        bests=[]

    #for b in range(batch_size):
    #    ious[b,target_sizes
    total_targets=0
    for b in range(batch_size): #by batch, has they have different numbers of targets
        if target_sizes[b]>0:
            total_targets+=target_sizes[b]
            minVs,best = ious[b,:,:target_sizes[b]].max(0) #this leaves index of pred for each target
            if return_alignment:
                bests.append(best)
            #box_loss = ious[:,best,range].sum()
            #batchInd = torch.arange(batch_size)[:,None].expand(batch_size,best.size(1)) #so we can use best to index in
            #import pdb; pdb.set_trace()
            #TODO loss using deactivated values
            #This will require haveing the model keep the anchors channel
            box_loss = F.mse_loss(predictions[b,best,1:6],target[b,:target_sizes[b],0:5],size_average=False,reduce=True) #skip conf and classes. this does error between activated  offsets/scaled of achors, not actual
            #import pdb; pdb.set_trace()
            box_loss += F.binary_cross_entropy_with_logits(confidences[b,best],torch.ones_like(confidences[b,best]),size_average=False,reduce=True) #here's conf
            box_loss += F.binary_cross_entropy_with_logits(pred_classes[b,best],target_classes[b,:target_sizes[b]],size_average=False,reduce=True) #yolov3 doesnt use softmax

            below_thresh[b,best]=0  #0 out the whole vector for preds we selected earlier
            below_thresh[b,:,target_sizes[b]:]=1 #we aren't too close to non-existent targets
        elif return_alignment:
            bests.append(None)
    #minVs,best = ious.min(1) #this leaves index of pred for each target
    ##box_loss = ious[:,best,range].sum()
    #batchInd = torch.arange(batch_size)[:,None].expand(batch_size,best.size(1)) #so we can use best to index in
    #if use_point_loss:
    #    box_loss = minVs.sum()
    ##TODO loss using deactivated values
    #else:
    #    box_loss = F.mse_loss(predictions[batchInd,best,1:6],target_box,size_average=False,reduce=True) #skip conf and classes. this does error between activated  offsets/scaled of achors, not actual
    #box_loss += F.binary_cross_entropy(confidences[batchInd,best],1,size_average=False,reduce=True) #here's conf
    #box_loss += F.binary_cross_entropy(pred_classes[batchInd,best],target_classes,size_average=False,reduce=True) #yolov3 doesnt use softmax

    #zero best
    #below_thresh[batchInd,best]=0 #0 out the whole vector for preds we selected earlier
    below_thresh,_ = below_thresh.min(dim=2) #if a pred is over threshold for all targets
    below_thresh=below_thresh.float()
    
    #tell these they should have 0 conf. Everything is zero-ed, so it will produce no error.
    conf_loss = F.binary_cross_entropy_with_logits(confidences*below_thresh,torch.zeros_like(below_thresh),size_average=False,reduce=True)
    conf_loss /= (below_thresh.sum()/num_anchors) #normalize
    box_loss /= total_targets


    loss = (conf_loss+box_loss)/batch_size

    if return_alignment:
        return loss, bests
    return loss


