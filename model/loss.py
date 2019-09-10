"""
    Copyright 2019 Brian Davis
    Visual-Template-free-Form-Parsting is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Visual-Template-free-Form-Parsting is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Visual-Template-free-Form-Parsting.  If not, see <https://www.gnu.org/licenses/>.
"""
import torch.nn.functional as F
import utils
#import torch.nn as nn
from model.alignment_loss import alignment_loss, box_alignment_loss, iou_alignment_loss
from model.yolo_loss import YoloLoss, YoloDistLoss, LineLoss

def my_loss(y_input, y_target):
    return F.nll_loss(y_input, y_target)

def sigmoid_BCE_loss(y_input, y_target):
    return F.binary_cross_entropy_with_logits(y_input, y_target)
def MSE(y_input, y_target):
    return F.mse_loss(y_input, y_target.float())



def detect_alignment_loss(predictions, target,label_sizes,alpha_alignment, alpha_backprop):
    return alignment_loss(predictions, target, label_sizes, alpha_alignment, alpha_backprop)
def detect_alignment_loss_points(predictions, target,label_sizes,alpha_alignment, alpha_backprop):
    return alignment_loss(predictions, target, label_sizes, alpha_alignment, alpha_backprop,points=True)

