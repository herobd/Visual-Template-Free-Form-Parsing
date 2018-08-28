import torch
from torch.autograd import Function
#from torch.autograd import Variable
from torch.nn.modules.module import Module
import numpy as np



class AffineGridGenFunction(Function):
    def __init__(self, height, width):
        super(AffineGridGenFunction, self).__init__()
        self.height, self.width = height, width
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        #print(self.grid)

    def forward(self, input1):
        self.input1 = input1
        output = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid

        if input1.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            output = output.cuda()

        for i in range(input1.size(0)):
                output = torch.bmm(self.batchgrid.view(-1, self.height*self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 2)

        return output

    def backward(self, grad_output):

        grad_input1 = torch.zeros(self.input1.size())

        if grad_output.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            grad_input1 = grad_input1.cuda()
        grad_input1 = torch.baddbmm(grad_input1, torch.transpose(grad_output.view(-1, self.height*self.width, 2), 1,2), self.batchgrid.view(-1, self.height*self.width, 3))

        return grad_input1

class AffineGridGen(Module):
    def __init__(self, height, width):
        super(AffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.f = AffineGridGenFunction(self.height, self.width)

    def forward(self, input):
        return self.f(input)


class PerspectiveGridGenFunction(Function):
    def __init__(self, height, width):
        super(PerspectiveGridGenFunction, self).__init__()
        self.height, self.width = height, width
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.linspace(-1, 1, self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.linspace(-1, 1, self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.input1 = input1
        output = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid

        if input1.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            output = output.cuda()

        for i in range(input1.size(0)):
                output = torch.bmm(self.batchgrid.view(-1, self.height*self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 3)

        return output

    def backward(self, grad_output):

        grad_input1 = torch.zeros(self.input1.size())

        if grad_output.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            grad_input1 = grad_input1.cuda()
        grad_input1 = torch.baddbmm(grad_input1, torch.transpose(grad_output.view(-1, self.height*self.width, 3), 1,2), self.batchgrid.view(-1, self.height*self.width, 3))

        return grad_input1

class PerspectiveGridGen(Module):
    def __init__(self, height, width):
        super(PerspectiveGridGen, self).__init__()
        self.height, self.width = height, width
        self.f = PerspectiveGridGenFunction(self.height, self.width)

    def forward(self, input):
        return self.f(input)

class GridGen(Module):
    def __init__(self, height, width):
        super(GridGen, self).__init__()
        self.height, self.width = height, width
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)

        grid_space_h = (np.arange(self.height) + 0.5) / float(self.height)
        grid_space_w = (np.arange(self.width) + 0.5) / float(self.width)

        grid_space_h = 2 * grid_space_h - 1
        grid_space_w = 2 * grid_space_w - 1

        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(grid_space_h, 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(grid_space_w, 0), repeats = self.height, axis = 0), 0)
        # self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.linspace(-1, 1, self.height), 0), repeats = self.width, axis = 0).T, 0)
        # self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.linspace(-1, 1, self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32)).cuda()

    def forward(self, input):
        out = torch.matmul(input[:,None,None,:,:], self.grid[None,:,:,:,None])
        out = out.squeeze(-1)
        return out
