import torch.nn.functional as F
import utils
#import torch.nn as nn
from model.alignment_loss import alignment_loss
from model.lf_loss import point_loss as lf_point_loss
from model.lf_loss import special_loss as lf_line_loss
from model.lf_loss import xyrs_loss as lf_xyrs_loss
from model.lf_loss import end_pred_loss as lf_end_loss

def my_loss(y_input, y_target):
    return F.nll_loss(y_input, y_target)

def sigmoid_BCE_loss(y_input, y_target):
    return F.binary_cross_entropy_with_logits(y_input, y_target)



def detect_alignment_loss(predictions, target,label_sizes,alpha_alignment, alpha_backprop):
    return alignment_loss(predictions, target, label_sizes, alpha_alignment, alpha_backprop)
def detect_alignment_loss_points(predictions, target,label_sizes,alpha_alignment, alpha_backprop):
    return alignment_loss(predictions, target, label_sizes, alpha_alignment, alpha_backprop,points=True)

#def lf_point_loss(prediction,target):
#    return point_loss(prediction,target)
#def lf_line_loss(prediction,target):
#    return special_loss(prediction,target)
#def lf_xyrs_loss(prediction,target):
#    return xyrs_loss(prediction,target)
#def lf_end_loss(end_pred,path_xyxy,end_point):
    #    return end_pred_loss(end_pred,path_xyxy,end_point)
