import numpy as np
import torch

def adjoint(A):
    """compute inverse without division by det; ...xv3xc3 input, or array of matrices assumed"""
    AI = np.empty_like(A)
    for i in xrange(3):
        AI[...,i,:] = np.cross(A[...,i-2,:], A[...,i-1,:])
    return AI

def inverse_transpose(A):
    """
    efficiently compute the inverse-transpose for stack of 3x3 matrices
    """
    I = adjoint(A)
    det = dot(I, A).mean(axis=-1)
    return I / det[...,None,None]

def inverse(A):
    """inverse of a stack of 3x3 matrices"""
    return np.swapaxes( inverse_transpose(A), -1,-2)
def dot(A, B):
    """dot arrays of vecs; contract over last indices"""
    return np.einsum('...i,...i->...', A, B)

def adjoint_torch(A):
    AI = A.clone()
    for i in xrange(3):
        AI[...,i,:] = torch.cross(A[...,i-2,:], A[...,i-1,:])
    return AI

def inverse_transpose_torch(A):
    I = adjoint_torch(A)
    det = dot_torch(I, A).mean(dim=-1)
    return I / det[:,None,None]

def inverse_torch(A):
    return inverse_transpose_torch(A).transpose(1, 2)

def dot_torch(A, B):
    A_view = A.view(-1,1,3)
    B_view = B.contiguous().view(-1,3,1)
    out = torch.bmm(A_view, B_view)
    out_view = out.view(A.size()[:-1])
    return out_view


if __name__ == "__main__":
    A = np.random.rand(2,3,3)
    I = inverse(A)

    A_torch = torch.from_numpy(A)

    I_torch = inverse_torch(A_torch)
    print(I)
    print(I_torch)
