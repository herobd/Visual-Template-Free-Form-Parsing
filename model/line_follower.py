import torch
from base import BaseModel
import torch.nn as nn
from model.gridgen import AffineGridGen, PerspectiveGridGen, GridGen
from model.lf_loss import getMinimumDists
import numpy as np
from utils import transformation_utils
#from lf_cnn import makeCnn
#from fast_patch_view import get_patches

def convRelu(i, batchNormalization=False, leakyRelu=False):
    nc = 3
    ks = [3, 3, 3, 3, 3, 3, 2]
    ps = [1, 1, 1, 1, 1, 1, 1]
    ss = [1, 1, 1, 1, 1, 1, 1]
    nm = [64, 128, 256, 256, 512, 512, 512]

    cnn = nn.Sequential()

    nIn = nc if i == 0 else nm[i - 1]
    nOut = nm[i]
    cnn.add_module('conv{0}'.format(i),
                   nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
    if batchNormalization:
        cnn.add_module('batchnorm{0}'.format(i), nn.InstanceNorm2d(nOut))
        # cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
    if leakyRelu:
        cnn.add_module('relu{0}'.format(i),
                       nn.LeakyReLU(0.2, inplace=True))
    else:
        cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
    return cnn

def makeCnn():

    cnn = nn.Sequential()
    cnn.add_module('convRelu{0}'.format(0), convRelu(0))
    cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(1), convRelu(1))
    cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(2), convRelu(2, True))
    cnn.add_module('convRelu{0}'.format(3), convRelu(3))
    cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(4), convRelu(4, True))
    cnn.add_module('convRelu{0}'.format(5), convRelu(5))
    cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(6), convRelu(6, True))
    cnn.add_module('pooling{0}'.format(4), nn.MaxPool2d(2, 2))

    return cnn

class LineFollower(BaseModel):
    def __init__(self, config, dtype=torch.cuda.FloatTensor):
        super(LineFollower, self).__init__(config)

        cnn = makeCnn()
        if "angle_only" in config and config["angle_only"]:
            self.no_xy=True
            num_pos=1
        else:
            self.no_xy=False
            num_pos=3

        self.pred_end = "pred_end" in config and config['pred_end']

        self.pred_scale = 'pred_scale' in config and config['pred_scale']
        
        position_linear = nn.Linear(512,num_pos + int(self.pred_end))
        position_linear.weight.data.zero_()
        position_linear.bias.data[0:num_pos] = 0 #dont shift or rotate, no scale is zero as well
        if self.pred_scale:
            self.scale_linear = nn.Linear(512,1)
            self.scale_linear.weight.data.zero_()
            self.scale_linear.bias.data[0] = 1 #scale is zero as well

        if 'noise_scale' in config:
            self.noise_scale = config['noise_scale']
        else:
            self.noise_scale = 1

        
        if 'output_grid_size' in config:
            self.output_grid_size = output_grid_size['output_grid_size']
        else:
            self.output_grid_size=32

        self.dtype = dtype
        self.cnn = cnn
        self.position_linear = position_linear

    def forward(self, image, positions, steps=None, all_positions=[], all_xy_positions=[], reset_interval=-1, randomize=False, negate_lw=False, skip_grid=False, allow_end_early=False):

        ##ttt=[]
        ##ttt2=[]

        batch_size = image.size(0)
        renorm_matrix = transformation_utils.compute_renorm_matrix(image)
        expanded_renorm_matrix = renorm_matrix.expand(batch_size,3,3)

        t = ((np.arange(self.output_grid_size) + 0.5) / float(self.output_grid_size))[:,None].astype(np.float32)
        t = np.repeat(t,axis=1, repeats=self.output_grid_size)
        t = torch.from_numpy(t).cuda()
        s = t.t()

        t = t[:,:,None]
        s = s[:,:,None]

        interpolations = torch.cat([
            (1-t)*s,
            (1-t)*(1-s),
            t*s,
            t*(1-s),
        ], dim=-1)

        view_window = torch.cuda.FloatTensor([
            [2,0,2],
            [0,2,0],
            [0,0,1]
        ]).expand(batch_size,3,3)

        step_bias = torch.cuda.FloatTensor([
            [1,0,-2],
            [0,1,0],
            [0,0,1]
        ]).expand(batch_size,3,3)

        invert = torch.cuda.FloatTensor([
            [-1,0,0],
            [0,-1,0],
            [0,0,1]
        ]).expand(batch_size,3,3)

        a_pt = torch.Tensor(
            [
                [0, 1,1],
                [0,-1,1]
            ]
        ).cuda()
        a_pt = a_pt.transpose(1,0)
        a_pt = a_pt.expand(batch_size, a_pt.size(0), a_pt.size(1))

        if negate_lw:
            view_window = invert.bmm(view_window)

        grid_gen = GridGen(32,32)

        view_window_imgs = []
        next_windows = []
        reset_windows = True
        for i in range(steps):

            if i%reset_interval != 0 or reset_interval==-1:
                p_0 = positions[-1]

                if i == 0 and len(p_0.size()) == 3 and p_0.size()[1] == 3 and p_0.size()[2] == 3:
                    current_window = p_0
                    reset_windows = False
                    next_windows.append(p_0)

            else:
                #p_0 = all_positions[i].type(self.dtype)
                if len(next_windows)>0:
                    w_0 = next_windows[-1]
                    cur_xy_pos = w_0.bmm(a_pt)
                    d_t, p_t, d_t, p_b = getMinimumDists(cur_xy_pos[0,:2,0],cur_xy_pos[0,:2,1],all_xy_positions, return_points=True) #all_positions[i].type(self.dtype)
                    d = p_t-p_b
                    scale = d.norm()/2
                    mx = (p_t[0]+p_b[0])/2.0
                    my = (p_t[1]+p_b[1])/2.0
                    theta = -torch.atan2(d[0],-d[1])
                    #print('d={}, scale={}, mx={}, my={}, theta={}'.format(d.size(),scale.size(),mx.size(),my.size(),theta.size()))
                    #print('w_0={}, cur_xy_pos={}, d={}, scale={}, mx={}, my={}, theta={}'.format(w_0.requires_grad,cur_xy_pos.requires_grad,d.requires_grad,scale.requires_grad,mx.requires_grad,my.requires_grad,theta.requires_grad))
                    #p_0 = torch.cat([mx,my,theta,scale,torch.ones_like(scale, requires_grad=True)])[None,...] #add batch dim
                    p_0 = torch.tensor([mx,my,theta,scale,1.0], requires_grad=True).cuda()[None,...] #add batch dim
                    #TODO may not requer grad
                else:
                    p_0 = all_positions[i].type(self.dtype) #this only occus an index 0 (?)
                reset_windows = True
                if randomize:
                    add_noise = p_0.clone()
                    add_noise.data.zero_()
                    mul_moise = p_0.clone()
                    mul_moise.data.fill_(1.0)

                    add_noise[:,0].data.uniform_(-2*self.noise_scale, 2*self.noise_scale)
                    add_noise[:,1].data.uniform_(-2*self.noise_scale, 2*self.noise_scale)
                    add_noise[:,2].data.uniform_(-.1*self.noise_scale, .1*self.noise_scale)

                    p_0 = p_0 * mul_moise + add_noise

            if reset_windows:
                reset_windows = False

                current_window = transformation_utils.get_init_matrix(p_0)

                if len(next_windows) == 0:
                    next_windows.append(current_window)
            else:
                current_window = next_windows[-1].detach()

            crop_window = current_window.bmm(view_window)
            #I need the x,y cords from here

            resampled = get_patches(image, crop_window, grid_gen, allow_end_early)

            if resampled is None and i > 0:
                #get patches checks to see if stopping early is allowed
                break

            if resampled is None and i == 0:
                #Odd case where it start completely off of the edge
                #This happens rarely, but maybe should be more eligantly handled
                #in the future
                resampled = torch.zeros(crop_window.size(0), 3, 32, 32).type_as(image.data)


            # Process Window CNN
            cnn_out = self.cnn(resampled)
            cnn_out = torch.squeeze(cnn_out, dim=2)
            cnn_out = torch.squeeze(cnn_out, dim=2)
            delta = self.position_linear(cnn_out)
            if self.pred_scale:
                scale_out = self.scale_linear(cnn_out)
                ##
                ##ttt.append(scale_out.item())
                scale_out = torch.clamp(scale_out,-0.1,0.1)
                ##scale_out = torch.clamp(scale_out,0.8,1.3)
                ##ttt2.append(scale_out.item())
                ##delta_scale=scale_out
                ##
                twos = 2*torch.ones_like(scale_out)
                delta_scale = torch.pow(twos, scale_out)
            else:
                delta_scale = None

            #if self.pred_scale:
            #    if self.no_xy:
            #        index=1
            #    else:
            #        index=3
            #    twos = 2*torch.ones_like(delta[:,index])
            #    #delta[:,index]=torch.pow(twos,torch.clamp(delta[:,index],-2,2)) #we clamp to prevent really weird things, having 2^x makes the scaling linear with respect the the nets linear output
            #    delta[:,index]=torch.pow(twos,delta[:,index]).clone() #having 2^x makes the scaling linear with respect the the nets linear output



            next_window = transformation_utils.get_step_matrix(delta,self.no_xy,delta_scale)
            next_window = next_window.bmm(step_bias)
            if negate_lw:
                next_window = invert.bmm(next_window).bmm(invert)

            next_windows.append(current_window.bmm(next_window))

            #if self.pred_end:



        grid_line = []
        mask_line = []
        line_done = []
        xy_positions = []


        for i in range(0, len(next_windows)-1):

            w_0 = next_windows[i]
            w_1 = next_windows[i+1]

            pts_0 = w_0.bmm(a_pt)
            pts_1 = w_1.bmm(a_pt)
            xy_positions.append(pts_0) #[[xU,xL],[yU,yL],[1,1]]

            if skip_grid:
                continue

            pts = torch.cat([pts_0, pts_1], dim=2)

            grid_pts = expanded_renorm_matrix.bmm(pts)

            grid = interpolations[None,:,:,None,:] * grid_pts[:,None,None,:,:]
            grid = grid.sum(dim=-1)[...,:2]

            grid_line.append(grid)
        if len(next_windows)==1:
            w_0 = next_windows[0]
            pts_0 = w_0.bmm(a_pt)
            b_pt = torch.Tensor(
                [
                    [-1,0,1],
                    [ 1,0,1]
                ]
            ).cuda()
            b_pt = a_pt.transpose(1,0)
            b_pt = b_pt.expand(batch_size, b_pt.size(0), b_pt.size(1))
            pts_1 = w_0.bmm(b_pt)
            xy_positions.append(pts_0)
            if not skip_grid:
                pts = torch.cat([pts_0, pts_1], dim=2)

                grid_pts = expanded_renorm_matrix.bmm(pts)

                grid = interpolations[None,:,:,None,:] * grid_pts[:,None,None,:,:]
                grid = grid.sum(dim=-1)[...,:2]

                grid_line.append(grid)
            

        xy_positions.append(pts_1)

        #print('pre-clamp {}, post-clamp {}'.format(['{:0.3f}'.format(v) for v in ttt],['{:0.3f}'.format(v) for v in ttt2]))

        if skip_grid:
            #grid_line = None
            return xy_positions
        else:
            grid_line = torch.cat(grid_line, dim=1)

        return grid_line, view_window_imgs, next_windows, xy_positions



def get_patches(image, crop_window, grid_gen, allow_end_early=False, end_points=None):


        pts = torch.FloatTensor([
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [ 1.0, 1.0,  1.0, 1.0]
        ]).type_as(image.data)[None,...]

        bounds = crop_window.matmul(pts)

        min_bounds, _ = bounds.min(dim=-1)
        max_bounds, _ = bounds.max(dim=-1)
        d_bounds = max_bounds - min_bounds
        floored_idx_offsets = torch.floor(min_bounds[:,:2].data).long()
        max_d_bounds = d_bounds.max(dim=0)[0].max(dim=0)[0]
        crop_size = torch.ceil(max_d_bounds).long()
        if image.is_cuda:
            crop_size = crop_size.cuda()
        w = crop_size.data.item() #[0]
        if w==0:
            w=1

        memory_space = torch.zeros(d_bounds.size(0), 3, w, w).type_as(image.data)
        translations = []
        N = transformation_utils.compute_renorm_matrix(memory_space)
        all_skipped = True

        for b_i in range(memory_space.size(0)):

            o = floored_idx_offsets[b_i]

            t = torch.cuda.FloatTensor([
                [1,0,-o[0]],
                [0,1,-o[1]],
                [0,0,    1]
            ]).expand(3,3)
            translations.append(N.mm(t)[None,...])

            skip_slice = False

            s_x = (o[0], o[0]+w)
            s_y = (o[1], o[1]+w)
            t_x = (0, w)
            t_y = (0, w)
            if o[0] < 0:
                s_x = (0, w+o[0])
                t_x = (-o[0], w)

            if o[1] < 0:
                s_y = (0, w+o[1])
                t_y = (-o[1], w)

            if o[0]+w >= image.size(2):
                s_x = (s_x[0], image.size(2))
                t_x = (t_x[0], t_x[0]+image.size(2) - s_x[0])

            if o[1]+w >= image.size(3):
                s_y = (s_y[1], image.size(3))
                t_y = (t_y[1], t_y[1]+image.size(3) - s_y[1])

            if s_x[0] >= s_x[1]:
                skip_slice = True

            if t_x[0] >= t_x[1]:
                skip_slice = True

            if s_y[0] >= s_y[1]:
                skip_slice = True

            if t_y[0] >= t_y[1]:
                skip_slice = True

            if not skip_slice:
                all_skipped = False
                i_s  = image[b_i:b_i+1, :, s_x[0]:s_x[1], s_y[0]:s_y[1]]
                #print(i_s.size())
                #print(memory_space.size())
                memory_space[b_i:b_i+1, :, t_x[0]:t_x[1], t_y[0]:t_y[1]] = i_s

                #if end_points is not None:
                    #get all points in bounds
                    #transform points

        if all_skipped and allow_end_early:
            return None

        translations = torch.cat(translations, 0)
        grid = grid_gen(translations.bmm(crop_window))
        grid = grid[:,:,:,0:2] / grid[:,:,:,2:3]

        resampled = torch.nn.functional.grid_sample(memory_space.transpose(2,3), grid, mode='bilinear')

        #if end_points is None:
        return resampled
        #else:

