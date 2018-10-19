import torch

def compute_distance(s0, e0, p0, return_point=False):

    l = e0 - s0
    v = p0 - s0
    length = l.norm()

    t = torch.dot(l/length, v/length)
    t = t.clamp(0,1)
    d = l * t
    distance = (d - v).norm()
    
    if return_point:
        return distance, s0+d
    else:
        return distance

def getMinimumDists(p0,p1,xy_positions, return_points=False):
    min_d0 = None
    min_d1 = None
    for j in range(len(xy_positions)-1):
        #print(xy_positions[j].size())
        s0 = xy_positions[j][0,:2,0]
        e0 = xy_positions[j+1][0,:2,0]
        if return_points:
            d0,point0 = compute_distance(s0,e0,p0,True)
            if min_d0 is None:
                min_d0 = d0
                min_p0 = point0
            else:
                min_locs = d0<min_d0
                min_p0 = torch.where(min_locs,point0,min_p0)
                min_d0 = torch.where(min_locs,d0,min_d0)
        else:
            d0 = compute_distance(s0,e0,p0)

            if min_d0 is None:
                min_d0 = d0
            else:
                min_d0 = torch.min(min_d0, d0)

        s1 = xy_positions[j][0,:2,1]
        e1 = xy_positions[j+1][0,:2,1]
        if return_points:
            d1,point1 = compute_distance(s1,e1,p1,True)
            if min_d1 is None:
                min_d1 = d1
                min_p1 = point1
            else:
                min_locs = d1<min_d1
                min_p1 = torch.where(min_locs,point1,min_p1)
                min_d1 = torch.where(min_locs,d1,min_d1)
        else:
            d1 = compute_distance(s1,e1,p1)

            if min_d1 is None:
                min_d1 = d1
            else:
                min_d1 = torch.min(min_d1, d1)
    if return_points:
        return min_d0, min_p0, min_d1, min_p1
    else:
        return min_d0,min_d1

def getMinimumDists_rs(p,xyrs_positions, return_points=False):
    min_d = None
    b=0
    for j in range(len(xyrs_positions)-1):
        #print(xy_positions[j].size())
        s = xyrs_positions[j][b,:2]
        e = xyrs_positions[j+1][b,:2]
        if return_points:
            d,point = compute_distance(s,e,p,True)
            if min_d is None:
                min_d = d
                min_p = point
                min_j=j
            else:
                if d<min_d:
                    min_p = point #torch.where(min_locs,point,min_p)
                    min_d = d #torch.where(min_locs,d,min_d)
                    min_j=j
        else:
            d = compute_distance(s,e,p)

            if min_d is None:
                min_d = d
                min_j=j
            else:
                min_d = torch.min(min_d, d)
                if (d<min_d):
                    min_j=j

    rot = xyrs_positions[min_j][b,2]
    scale = xyrs_positions[min_j][b,3]
    if return_points:
        return min_d, rot,scale, min_p
    else:
        return min_d, rot,scale

#special loss only works with batch size 1
#this computes the loss from the predicted point to the neartest point-on-a-line defined by the GT points
def special_loss(xy_output, xy_positions):
    assert(xy_output[0].size(0)==1)
    b=0
    loss = 0
    for i in range(len(xy_output)):
        p0 = xy_output[i][b,:2,0]
        p1 = xy_output[i][b,:2,1]

        min_d0,min_d1 = getMinimumDists(p0,p1,xy_positions)
        #if (min_d0>14 or min_d1>14) and i!=len(xy_output)-1:
        #    print('min_d0:{}, min_d1:{}'.format(min_d0,min_d1))
        #    import pdb; pdb.set_trace()

        loss += min_d0
        loss += min_d1

    return loss

def xyrs_loss(xyrs_output, xyrs_positions):
    #assert(xyrs_output[0].size(0)==1)
    loss = 0
    for i in range(len(xyrs_output)):
        p = xyrs_output[i][0:2]

        min_d, rot,scale = getMinimumDists_rs(p,xyrs_positions)
        #if (min_d0>14 or min_d1>14) and i!=len(xy_output)-1:
        #    print('min_d0:{}, min_d1:{}'.format(min_d0,min_d1))
        #    import pdb; pdb.set_trace()
        scale_dif = scale-xyrs_output[i][3]
        rot_dif = rot-xyrs_output[i][2]

        loss += torch.pow(min_d,2) + torch.pow(scale_dif,2) + 3*torch.pow(rot_dif,2)

    return loss

def point_loss(xy_output, xy_positions):
    loss_fn = torch.nn.MSELoss()
    loss = 0
    for i, l in enumerate(xy_positions):
        loss += loss_fn(xy_output[i][:,:2,:2], l)
    return loss

def get_side(a, b):
    x = x_product(a, b)
    if x < 0:
        return 'left'
    elif x > 0:
        return 'right'
    else:
        return None

def v_sub(a, b):
    return (a[0]-b[0], a[1]-b[1])

def x_product(a, b):
    return a[0]*b[1]-a[1]*b[0]


def checkInsidePoly(point,vertices):
    #point=(x,y)
    previous_side = None
    n_vertices = len(vertices)
    for n in range(n_vertices):
        a, b = vertices[n], vertices[(n+1)%n_vertices]
        affine_segment = a-b #v_sub(b, a)
        affine_point = point-a #v_sub(point, a)
        current_side = get_side(affine_segment, affine_point)
        if current_side is None:
            return False #outside or over an edge
        elif previous_side is None: #first segment
            previous_side = current_side
        elif previous_side != current_side:
            return False
    return True

#This assumes batch size of 1
def end_pred_loss(end_pred,path_xyxy,end_point):
    assert(end_point.size(0)==1)
    assert(len(end_pred)==len(path_xyxy)-1)
    b=0
    passed_end=False
    for i in range(len(end_pred)):
        pred = end_pred[i][b]
        tl = path_xyxy[i][b,0:2,0]
        bl = path_xyxy[i][b,0:2,1]
        tr = path_xyxy[i+1][b,0:2,0]
        br = path_xyxy[i+1][b,0:2,1]
        if passed_end or checkInsidePoly(end_point[b],[tl,tr,br,bl]):
            passed_end=True
            truth=1
        else:
            truth=0
        return F.binary_cross_entropy(pred,truth)
