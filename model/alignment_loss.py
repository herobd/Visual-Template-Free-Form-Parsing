import torch
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

    normed_difference = torch.norm(location_deltas,2,3)**2
    expanded_log_confidences = log_confidences[:,:,None].expand(locations.size(0), locations.size(1), target.size(1))
    expanded_log_one_minus_confidences = log_one_minus_confidences[:,:,None].expand(locations.size(0), locations.size(1), target.size(1))

    C = alpha_alignment/2.0 * normed_difference - expanded_log_confidences + expanded_log_one_minus_confidences

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
        #        print('loc{}: {}, {},  dist={}'.format(i,locations[b,i,0],locations[b,i,1],normed_difference[b,i,0]))
        #    for i in range(len(target_ind)):
        #        targ_i = target_ind[i]
        #        loc_i = location_ind[i]
        #        print('size locations={}, size target={}'.format(locations.size(1), l))
        #        print('targ_i={}, loc_i={}'.format(targ_i,loc_i))
        #        print('gt location={}, {}'.format(target[b,targ_i,0],target[b,targ_i,1]))
        #        print('aligned C={}, normed_difference={}, expanded_log_confidences={}, expanded_log_one_minus_confidences={}'.format(C_i[loc_i,targ_i],normed_difference[b,loc_i,targ_i],expanded_log_confidences[b,loc_i,targ_i],expanded_log_one_minus_confidences[b,loc_i,targ_i]))
        #        print('aligned pred location={}, {}'.format(locations[b,loc_i,0],locations[b,loc_i,1]))

        #        closest_loc_i = normed_difference[b,:,targ_i].argmin()
        #        print('closest C={}, normed_difference={}, expanded_log_confidences={}, expanded_log_one_minus_confidences={}'.format(C_i[closest_loc_i,targ_i],normed_difference[b,closest_loc_i,targ_i],expanded_log_confidences[b,closest_loc_i,targ_i],expanded_log_one_minus_confidences[b,closest_loc_i,targ_i]))
        #        print('closest pred location={}, {}'.format(locations[b,closest_loc_i,0],locations[b,closest_loc_i,1]))
        #        exit()


    X = torch.from_numpy(X).type(predictions.data.type())
    X2 = 1.0 - torch.sum(X, 2)

    location_loss = (alpha_backprop/2.0 * normed_difference * X).sum()
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
