import torch
import torch.nn as nn
#from torch.autograd import Variable
from torch.nn.modules.module import Module

from utils.fast_inverse import inverse_torch
import numpy as np

def compute_renorm_matrix(img):
    inv_c = np.array([
        [1.0/img.size(2), 0, 0],
        [0, 1.0/img.size(3), 0],
        [0,0,1]
    ], dtype=np.float32)

    inv_b = np.array([
        [2,0,-1],
        [0,2,-1],
        [0,0, 1]
    ], dtype=np.float32)

    inv_c = torch.from_numpy(inv_c).type(img.data.type())
    inv_b = torch.from_numpy(inv_b).type(img.data.type())

    return inv_b.mm(inv_c)

def compute_next_state(delta, state):
    out = torch.zeros(*state.data.shape).type(state.data.type())
    for i in xrange(0,3):
        out[:,i+2] = delta[:,i] + state[:,i+2]
    #r*cos(theta) + x = x'
    out[:,0] = out[:,3] * torch.cos(out[:,2]) + state[:,0]
    #r*sin(theta) + y = y'
    out[:,1] = out[:,3] * torch.sin(out[:,2]) + state[:,1]
    return out

def compute_points(state):
    out = torch.zeros(state.data.shape[0],2,2).type(state.data.type())
    out[:,0,0] = state[:,4] * torch.sin(state[:,2])
    out[:,0,1] = state[:,4] * torch.cos(state[:,2])

    out[:,1] = -out[:,0]

    out[:,:,0] = out[:,:,0] + state[:,0]
    out[:,:,1] = out[:,:,1] + state[:,1]

    return out

import time
def compute_basis(pts):
    #start = time.time()
    A = pts[:,:3,:3]
    b = pts[:,:3,3:4]
    #A_inv = A.clone()
    #for i in xrange(A.data.shape[0]):
    #    A_inv[i,:,:] = torch.inverse(A[i,:,:])

    #A_inv = [t.inverse() for t in torch.functional.unbind(A)]
    #A_inv = torch.functional.stack(A_inv)
    A_inv = inverse_torch(A)


    #print "s", time.time() - start
    x = A_inv.bmm(b)

    B = A.clone()
    for i in xrange(3):
        B[:,:,i] = A[:,:,i] * x[:,i]
    return B

DEFAULT_TARGET = np.array([[
    [-1.0,-1, 1, 1],
    [ 1.0,-1, 1,-1],
    [ 1.0, 1, 1, 1]
]])
BASIS = None
def compute_perspective(pts, target=None):
    global BASIS
    if target is None:
        target = torch.from_numpy(DEFAULT_TARGET).type(pts.data.type())
    if BASIS is None:
        B = compute_basis(target)
        BASIS = inverse_torch(B)

    basis = BASIS.expand(pts.size(0), BASIS.size(1), BASIS.size(2))

    A = compute_basis(pts)
    return A.bmm(basis)

def pt_ori_sca_2_pts(state):
    # Input: b x [x, y, theta, scale]
    out = torch.ones(state.data.shape[0], 3, 2).type(state.data.type())
    out[:,0,0] =  torch.sin(state[:,2]) * state[:,3] + state[:,0]
    out[:,1,0] =  torch.cos(state[:,2]) * state[:,3] + state[:,1]
    out[:,0,1] = -torch.sin(state[:,2]) * state[:,3] + state[:,0]
    out[:,1,1] = -torch.cos(state[:,2]) * state[:,3] + state[:,1]

    return out

def get_init_matrix(input):
    output = torch.zeros((input.size(0), 3, 3)).type(input.data.type())
    output[:,0,0] = 1
    output[:,1,1] = 1
    output[:,2,2] = 1

    x = input[:,0:1]
    y = input[:,1:2]
    angles = input[:,2:3]
    scaler = input[:,3:4]

    cosines = torch.cos(angles)
    sinuses = torch.sin(angles)
    output[:,0,0] =  cosines * scaler
    output[:,1,1] =  cosines * scaler
    output[:,1,0] = -sinuses * scaler
    output[:,0,1] =  sinuses * scaler

    output[:,0,2] = x
    output[:,1,2] = y

    return output

#the input is a delta, I allow either x,y,theta or just theta
def get_step_matrix(input,no_xy,scale_index):
    output = torch.zeros((input.size(0), 3, 3)).type(input.data.type())
    output[:,0,0] = 1
    output[:,1,1] = 1
    output[:,2,2] = 1

    if scale_index is None:
        scale=torch.ones_like(input[:,0])
    else:
        scale=input[:,scale_index]
    if no_xy:
        x = y = 0
        angles = input[:,0:1]
        #if use_scale:
        #    scale = input[:,1:2]
    else:
        x = input[:,0:1]
        y = input[:,1:2]
        angles = input[:,2:3]
        #if use_scale:
        #    scale = input[:,3:4]

    cosines = torch.cos(angles)
    sinuses = torch.sin(angles)
    output[:,0,0] =  cosines*scale
    output[:,1,1] =  cosines*scale
    output[:,1,0] = -sinuses*scale
    output[:,0,1] =  sinuses*scale

    output[:,0,2] = x
    output[:,1,2] = y

    return output

class ScaleRotateMatrixGenerator(Module):
    def __init__(self):
        super(ScaleRotateMatrixGenerator, self).__init__()

    def forward(self, input):
        output = torch.zeros((input.size(0), 3, 2)).type(input.data.type())
        output[:,0,0] = 1
        output[:,1,1] = 1

        angles = input[:,0]
        scaler = input[:,1]

        cosines = torch.cos(angles)
        sinuses = torch.sin(angles)
        output[:,0,0] =  cosines * scaler
        output[:,1,1] =  cosines * scaler
        output[:,1,0] = -sinuses * scaler
        output[:,0,1] =  sinuses * scaler

        return output

#from https://gist.github.com/ncullen93/425ca642955f73452ebc097b3b46c493
def transform_matrix_offset_center(matrix, x, y):
    """Apply offset to a transform matrix so that the image is
    transformed about the center of the image. 
    NOTE: This is a fairly simple operaion, so can easily be
    moved to full torch.
    Arguments
    ---------
    matrix : 3x3 matrix/array
    x : integer
        height dimension of image to be transformed
    y : integer
        width dimension of image to be transformed
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform, fill_mode='nearest', fill_value=0., out_shape=None):
    """Applies an affine transform to a 2D array, or to each channel of a 3D array.
    NOTE: this can and certainly should be moved to full torch operations.
    Arguments
    ---------
    x : np.ndarray
        array to transform. NOTE: array should be ordered CHW
    
    transform : 3x3 affine transform matrix
        matrix to apply
    """
    x = x.astype('float32')
    transform = transform_matrix_offset_center(transform, x.shape[1], x.shape[2])
    final_affine_matrix = transform[:2, :2]
    final_offset = transform[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
            final_offset,output_shape=out_shape, order=0, mode=fill_mode, cval=fill_value) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    return x

def rotate(input, rotation, crop_to):
    #if rotation==0:
    #    return input
    theta = rotation #math.pi / 180 * degree
    rotation_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                                [math.sin(theta), math.cos(theta), 0],
                                [0, 0, 1]])
    x_transformed = torch.from_numpy(apply_transform(input.numpy(), rotation_matrix,
        fill_mode=self.fill_mode, fill_value=self.fill_value, out_shape=crop_to))
    return x_transformed
