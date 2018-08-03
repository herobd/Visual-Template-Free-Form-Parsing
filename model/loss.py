import torch.nn.functional as F
import utils
#import torch.nn as nn
from model import alignment_loss

def my_loss(y_input, y_target):
    return F.nll_loss(y_input, y_target)

def sigmoid_BCE_loss(y_input, y_target):
    return F.binary_cross_entropy_with_logits(y_input, y_target)



def detect_alignment_loss(alpha_alignment, alpha_backprop):
    def loss(predictions, target,label_sizes):
        return alignment_loss(predictions, target, label_sizes, alpha_alignment, alpha_backprop)

