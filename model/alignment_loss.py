import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def alignment_loss(predictions, target, label_sizes, alpha_alignment=1000.0, alpha_backprop=100.0):
    batch_size = predictions.size(0)
    # This should probably be computed using the log_softmax
    confidences = predictions[:,:,0]
    log_one_minus_confidences = torch.log(1.0 - confidences + 1e-10)

    if target is None:
        return -log_one_minus_confidences.sum()

    locations = predictions[:,:,1:5]
    target = target[:,:,0:4]

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
    for b in xrange(C.shape[0]):
        l = label_sizes[b]
        if l == 0:
            continue

        C_i = C[b,:,:l]
        row_ind, col_ind = linear_sum_assignment(C_i.T)
        X[b][(col_ind, row_ind)] = 1.0

    X = torch.from_numpy(X).type(predictions.data.type())
    X2 = 1.0 - torch.sum(X, 2)

    location_loss = (alpha_backprop/2.0 * normed_difference * X).sum()
    confidence_loss =  -(expanded_log_confidences * X).sum() - (log_one_minus_confidences * X2).sum()

    loss = confidence_loss + location_loss

    loss = loss/batch_size

    return loss
