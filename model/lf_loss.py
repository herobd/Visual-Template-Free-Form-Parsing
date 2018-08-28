import torch

def compute_distance(s0, e0, p0):

    l = e0 - s0
    v = p0 - s0
    length = l.norm()

    t = torch.dot(l/length, v/length)
    t = t.clamp(0,1)
    d = l * t
    distance = (d - v).norm()

    return distance

def special_loss(xy_output, xy_positions):
    loss = 0
    for i in range(len(xy_output)-1):
        p0 = xy_output[i][0,:2,0]
        p1 = xy_output[i][0,:2,1]
        min_d0 = None
        min_d1 = None
        for j in range(len(xy_positions)-1):

            s0 = xy_positions[j][0,:2,0]
            e0 = xy_positions[j+1][0,:2,0]
            d0 = compute_distance(s0,e0,p0)

            if min_d0 is None:
                min_d0 = d0
            else:
                min_d0 = torch.min(min_d0, d0)

            s1 = xy_positions[j][0,:2,1]
            e1 = xy_positions[j+1][0,:2,1]
            d1 = compute_distance(s1,e1,p1)

            if min_d1 is None:
                min_d1 = d1
            else:
                min_d1 = torch.min(min_d1, d1)

        loss += min_d0
        loss += min_d1

    return loss

def point_loss(xy_output, xy_positions):
    loss_fn = torch.nn.MSELoss()
    loss = 0
    for i, l in enumerate(xy_positions):
        loss += loss_fn(xy_output[i][:,:2,:2], l)
    return loss
