import numpy as np
import cv2
import timeit
import warnings
import random

def perform_crop(img, gt, crop):
    #csX,csY = crop['crop_size']
    cropped_gt_img = img[crop['dim0'][0]:crop['dim0'][1], crop['dim1'][0]:crop['dim1'][1]]
    scaled_gt_img = cropped_gt_img #cv2.resize(cropped_gt_img, (csY, csX), interpolation = cv2.INTER_CUBIC)
    if len(scaled_gt_img.shape)==2:
        scaled_gt_img = scaled_gt_img[...,None]
    scaled_gt = None
    if gt is not None:
        cropped_gt = gt[crop['dim0'][0]:crop['dim0'][1], crop['dim1'][0]:crop['dim1'][1]]
        scaled_gt = cropped_gt #cv2.resize(cropped_gt, (cs, cs), interpolation = cv2.INTER_CUBIC)
        if len(scaled_gt.shape)==2:
            scaled_gt = scaled_gt[...,None]
    return scaled_gt_img, scaled_gt


def generate_random_crop(img, pixel_gt, line_gts, point_gts, params, bb_gt=None, bb_ids=None, query_bb=None,cropPoint=None):
    
    contains_label = np.random.random() < params['prob_label'] if 'prob_label' in params else None
    cs = params['crop_size']
    if type(cs)==int:
        csX=cs
        csY=cs
    else:
        csX=cs[1]
        csY=cs[0]
    cs=None

    cnt = 0
    while True: #we loop random crops to try and get an instance
        if cropPoint is None:
            if query_bb is None:
                dim0 = np.random.randint(0,img.shape[0]-csY)
                dim1 = np.random.randint(0,img.shape[1]-csX)
            else:
                #force the random crop to fully contain the query, if it can
                # otherwise contain part of it
                minY=int(max(0,query_bb[9]-csY,query_bb[11]-csY,query_bb[13]-csY,query_bb[15]-csY))
                maxY=int(min(img.shape[0]-csY,query_bb[9]+1,query_bb[11]+1,query_bb[13]+1,query_bb[15]+1))
                if minY>=maxY:
                    minY= random.choice([query_bb[11]-csY,query_bb[13]-csY,query_bb[15]-csY])
                    maxY= random.choice([query_bb[9]+1,query_bb[11]+1,query_bb[13]+1,query_bb[15]+1])
                    if minY>=maxY:
                        dim0 = random.choice([minY,maxY])
                    else:
                        dim0 = np.random.randint(minY,maxY)
                    dim0 = int(min(img.shape[0]-csY,max(0,dim0)))
                    #minY=int(max(0,min(query_bb[9],query_bb[11],query_bb[13],query_bb[15])))
                    #maxY=int(min(img.shape[0]-csY,1+max(query_bb[9]-csY,query_bb[11]-csY,query_bb[13]-csY,query_bb[15]-csY)))
                else:
                    dim0 = np.random.randint(minY,maxY)
                minX=int(max(0,query_bb[8]-csX,query_bb[10]-csX,query_bb[12]-csX,query_bb[14]-csX))
                maxX=int(min(img.shape[1]-csX,query_bb[8]+1,query_bb[10]+1,query_bb[12]+1,query_bb[14]+1))
                if minX>=maxX:
                    minX= random.choice([query_bb[8]-csY,query_bb[10]-csX,query_bb[12]-csX,query_bb[14]-csX])
                    maxX= random.choice([query_bb[8]+1,query_bb[10]+1,query_bb[12]+1,query_bb[14]+1])
                    if minX>=maxX:
                        dim1 = random.choice([minX,maxX])
                    else:
                        dim1 = np.random.randint(minX,maxX)
                    dim1 = int(min(img.shape[1]-csX,max(0,dim1)))
                    #minX=int(max(0,min(query_bb[8],query_bb[10],query_bb[12],query_bb[14])))
                    #maxX=int(min(img.shape[1]-csX,1+max(query_bb[8]-csX,query_bb[10]-csX,query_bb[12]-csX,query_bb[14]-csY)))
                else:
                    dim1 = np.random.randint(minX,maxX)
        else:
            dim0=cropPoint[1]
            dim1=cropPoint[0]

        crop = {
            "dim0": [dim0, dim0+csY],
            "dim1": [dim1, dim1+csX],
            #"crop_size": (csX,csY)
        }
        hit=False
    
        if line_gts is not None:
            line_gt_match={}
            for name, gt in line_gts.items():
                ##tic=timeit.default_timer()
                line_gt_match[name] = np.zeros_like(gt)
                line_gt_match[name][...,0][gt[...,0] < dim1] = 1
                line_gt_match[name][...,0][gt[...,0] > dim1+csX] = 1

                line_gt_match[name][...,1][gt[...,1] < dim0] = 1
                line_gt_match[name][...,1][gt[...,1] > dim0+csY] = 1

                line_gt_match[name][...,2][gt[...,2] < dim1] = 1
                line_gt_match[name][...,2][gt[...,2] > dim1+csX] = 1

                line_gt_match[name][...,3][gt[...,3] < dim0] = 1
                line_gt_match[name][...,3][gt[...,3] > dim0+csY] = 1

                line_gt_match[name] = 1-line_gt_match[name]
                line_gt_match[name] = np.logical_and.reduce((line_gt_match[name][...,0], line_gt_match[name][...,1], line_gt_match[name][...,2], line_gt_match[name][...,3]))
                if line_gt_match[name].sum() > 0:
                    hit=True
        else:
            line_gt_match=None



        got_all=True
        if bb_gt is not None:
            bb_gt_match=np.zeros_like(bb_gt)

            bb_gt_match[...,8][bb_gt[...,8] < dim1] = 1
            bb_gt_match[...,0][bb_gt[...,8] >= dim1+csX] = 1

            bb_gt_match[...,9][bb_gt[...,9] < dim0] = 1
            bb_gt_match[...,1][bb_gt[...,9] >= dim0+csY] = 1

            bb_gt_match[...,10][bb_gt[...,10] < dim1] = 1
            bb_gt_match[...,2][bb_gt[...,10] >= dim1+csX] = 1

            bb_gt_match[...,11][bb_gt[...,11] < dim0] = 1
            bb_gt_match[...,3][bb_gt[...,11] >= dim0+csY] = 1

            bb_gt_match[...,12][bb_gt[...,12] < dim1] = 1
            bb_gt_match[...,12][bb_gt[...,12] >= dim1+csX] = 1

            bb_gt_match[...,13][bb_gt[...,13] < dim0] = 1
            bb_gt_match[...,13][bb_gt[...,13] >= dim0+csY] = 1

            bb_gt_match[...,14][bb_gt[...,14] < dim1] = 1
            bb_gt_match[...,14][bb_gt[...,14] >= dim1+csX] = 1

            bb_gt_match[...,15][bb_gt[...,15] < dim0] = 1
            bb_gt_match[...,15][bb_gt[...,15] >= dim0+csY] = 1


            bb_gt_match = 1-bb_gt_match
            left_inside_l = bb_gt_match[...,8]
            left_inside_r = bb_gt_match[...,0]
            left_inside_t = bb_gt_match[...,9]
            left_inside_b = bb_gt_match[...,1]
            has_left= np.logical_and.reduce([left_inside_l,left_inside_r,left_inside_t,left_inside_b])
            right_inside_l = bb_gt_match[...,10]
            right_inside_r = bb_gt_match[...,2]
            right_inside_t = bb_gt_match[...,11]
            right_inside_b = bb_gt_match[...,3]
            has_right= np.logical_and.reduce([right_inside_l,right_inside_r,right_inside_t,right_inside_b]) 
            has_top= np.logical_and(bb_gt_match[...,12], bb_gt_match[...,13])  
            has_bot= np.logical_and(bb_gt_match[...,14], bb_gt_match[...,15])

            #bb_gt_cornerCount = has_left+has_right+has_top+has_bot
            #bb_gt_part = bb_gt_cornerCount==2 #if you have two corners in, your a partial
            #bb_gt_candidate = np.logical_or( np.logical_and(np.logical_or(has_top,has_bot),np.logical_or(has_left,has_right)),
            bb_gt_candidate = np.logical_or( np.logical_or(has_left,has_right),
                                             np.logical_and(has_top,has_bot))
            got_all = bb_gt_candidate.all()
            
            

            if bb_gt_candidate.sum() > 0:
                hit=True
        else:
            got_all = True
            bb_gt_match= None

        point_gt_match={}
        if point_gts is not None:
            for name, gt in point_gts.items():
                if gt is not None:
                    ##tic=timeit.default_timer()
                    point_gt_match[name] = np.zeros_like(gt)
                    point_gt_match[name][...,0][gt[...,0] < dim1] = 1
                    point_gt_match[name][...,0][gt[...,0] > dim1+csX] = 1

                    point_gt_match[name][...,1][gt[...,1] < dim0] = 1
                    point_gt_match[name][...,1][gt[...,1] > dim0+csY] = 1

                    point_gt_match[name] = 1-point_gt_match[name]
                    point_gt_match[name] = np.logical_and(point_gt_match[name][...,0], point_gt_match[name][...,1])
                    if point_gt_match[name].sum() > 0:
                        hit=True
                    ##print('match: {}'.format(timeit.default_timer()-##tic))
        else:
            point_gt_match=None
        
        if (
            (cropPoint is not None)
            or
            (
            query_bb is not None and (
                got_all or
                cnt>50 ) 
            )
            or 
            (
            query_bb is None and (
                cnt > 100 or 
                    (contains_label is None or
                    (hit and contains_label) or
                    (not hit and not contains_label) ) )
            )
           ):
                cropped_gt_img, cropped_pixel_gt = perform_crop(img,pixel_gt, crop)
                if line_gts is not None:
                    for name in line_gts:
                        line_gt_match[name] = np.where(line_gt_match[name]!=0)
                if bb_gt is not None:
                    with warnings.catch_warnings(): 
                        warnings.simplefilter("ignore")#we do some div by zero stuff that's caught within
                        #We need to clip bbs that go outsire crop
                        #this is a bit of a mess...
                        #we do the clipping for all BBs, but those inside just dont get clipped
                        bb_gt = bb_gt[np.where(bb_gt_candidate)]
                        left_inside_l = left_inside_l[np.where(bb_gt_candidate)] 
                        left_inside_r = left_inside_r[np.where(bb_gt_candidate)] 
                        left_inside_t = left_inside_t[np.where(bb_gt_candidate)] 
                        left_inside_b = left_inside_b[np.where(bb_gt_candidate)] 
                        right_inside_l = right_inside_l[np.where(bb_gt_candidate)]
                        right_inside_r = right_inside_r[np.where(bb_gt_candidate)]
                        right_inside_t = right_inside_t[np.where(bb_gt_candidate)]
                        right_inside_b = right_inside_b[np.where(bb_gt_candidate)]
                        #we're going to edit bb_gt to make boxes partially in crop to be fully in crop 
                        #bring in left side
                        #needs_left = np.logical_and(bb_gt_candidate,1-has_left)[:,:,None]#, [1,1,2]) # things that are candidates where the left point is out-of-bounds
                        v_r = bb_gt[...,10:12]-bb_gt[...,8:10] #vector to opposite point
                        #what do we need to bring in?
                        dist1_l = (1-left_inside_l)*(dim1-bb_gt[...,8])/v_r[...,0] #distance along vector till intersecting left clipped boundary
                        dist1_r = (1-left_inside_r)*(dim1+csX-bb_gt[...,8])/v_r[...,0] # " right boundary
                        dist0_t = (1-left_inside_t)*(dim0-bb_gt[...,9])/v_r[...,1] # " top boudary
                        dist0_b = (1-left_inside_b)*(dim0+csY-bb_gt[...,9])/v_r[...,1] # " bottom boundarya
                        np.nan_to_num(dist1_l,False)
                        np.nan_to_num(dist1_r,False)
                        np.nan_to_num(dist0_t,False)
                        np.nan_to_num(dist0_b,False)
                        # #Take the closest boundary intersection and get the vector that corresponds
                        # #mv_left = v_r*(np.maximum(np.minimum.reduce([dist1_l,dist1_r,dist0_t,dist0_b]),0)[:,:,None])
                        #Take the largest of the boundaries we need (others are zeroed out)
                        mv_left = v_r*(np.maximum.reduce([dist1_l,dist1_r,dist0_t,dist0_b])[...,None])
                        #Now add that vector to the two corner points to bring them in
                        #bb_gt[...,0:2] = np.where( needs_left , bb_gt[...,0:2]+mv_left, bb_gt[...,0:2])
                        #bb_gt[...,6:8] = np.where( needs_left , bb_gt[...,6:8]+mv_left, bb_gt[...,6:8])
                        bb_gt[...,0:2] += mv_left
                        bb_gt[...,6:8] += mv_left

                        #bring in right side
                        #same process as left side
                        #needs_right = np.logical_and(bb_gt_candidate,1-has_right)[:,:,None]#, [1,1,2])
                        v_l = -bb_gt[...,10:12]+bb_gt[...,8:10]
                        dist1_l = (1-right_inside_l)*(dim1-bb_gt[...,10])/v_l[...,0]
                        dist1_r = (1-right_inside_r)*(dim1+csX-bb_gt[...,10])/v_l[...,0]
                        dist0_t = (1-right_inside_t)*(dim0-bb_gt[...,11])/v_l[...,1]
                        dist0_b = (1-right_inside_b)*(dim0+csY-bb_gt[...,11])/v_l[...,1]
                        np.nan_to_num(dist1_l,False)
                        np.nan_to_num(dist1_r,False)
                        np.nan_to_num(dist0_t,False)
                        np.nan_to_num(dist0_b,False)
                        #mv_right = v_l*(np.maximum(np.minimum.reduce([dist1_l,dist1_r,dist0_t,dist0_b]),0)[:,:,None])
                        mv_right = v_l*(np.maximum.reduce([dist1_l,dist1_r,dist0_t,dist0_b])[...,None])
                        #bb_gt[...,2:4] = np.where( needs_right, bb_gt[...,2:4]+mv_right, bb_gt[...,2:4])
                        #bb_gt[...,4:6] = np.where( needs_right, bb_gt[...,4:6]+mv_right, bb_gt[...,4:6])
                        bb_gt[...,2:4] += mv_right
                        bb_gt[...,4:6] += mv_right
                        #bb_gt = bb_gt[np.where(bb_gt_candidate)]

                        if bb_ids is not None:
                            bb_ids = [id for ind,id in enumerate(bb_ids) if  bb_gt_candidate[0,ind]]
                if point_gts is not None:
                    for name in point_gt_match:
                        point_gt_match[name] = np.where(point_gt_match[name]!=0)
                return crop, cropped_gt_img, cropped_pixel_gt, line_gt_match, point_gt_match, bb_gt, bb_ids, (dim1,dim0)

        cnt += 1

class CropTransform(object):
    def __init__(self, crop_params):
        crop_size = crop_params['crop_size']
        self.random_crop_params = crop_params
        if 'pad' in crop_params:
            pad_by = crop_params['pad']
        else:
            pad_by = crop_size//2
        self.pad_params = ((pad_by,pad_by),(pad_by,pad_by),(0,0))

    def __call__(self, sample):
        org_img = sample['img']
        line_gts = sample['line_gt']
        point_gts = sample['point_gt']
        pixel_gt = sample['pixel_gt']

        #pad out to allow random samples to take space off of the page
        ##tic=timeit.default_timer()
        #org_img = np.pad(org_img, self.pad_params, 'mean')
        org_img = np.pad(org_img, self.pad_params, 'constant')
        if pixel_gt is not None:
            pixel_gt = np.pad(pixel_gt, self.pad_params, 'constant')
        ##print('pad: {}'.format(timeit.default_timer()-##tic))
        
        ##tic=timeit.default_timer()
        j=0
        #pad the points accordingly
        for name, gt in line_gts.items():
            #if np.isnan(gt).any():
            #    print('gt has nan, {}'.format(name))
            gt[:,:,0] = gt[:,:,0] + self.pad_params[0][0]
            gt[:,:,1] = gt[:,:,1] + self.pad_params[1][0]

            gt[:,:,2] = gt[:,:,2] + self.pad_params[0][0]
            gt[:,:,3] = gt[:,:,3] + self.pad_params[1][0]
        for name, gt in point_gts.items():
            gt[:,:,0] = gt[:,:,0] + self.pad_params[0][0]
            gt[:,:,1] = gt[:,:,1] + self.pad_params[1][0]

        crop_params, org_img, pixel_gt, line_gt_match, point_gt_match, _, _, cropPoint = generate_random_crop(org_img, pixel_gt, line_gts, point_gts, self.random_crop_params)
        #print(crop_params)
        #print(gt_match)
        
        ##tic=timeit.default_timer()
        new_line_gts={}
        for name, gt in line_gts.items():
            gt = gt[line_gt_match[name]][None,...] #add batch dim (?)
            gt[...,0] = gt[...,0] - crop_params['dim1'][0]
            gt[...,1] = gt[...,1] - crop_params['dim0'][0]

            gt[...,2] = gt[...,2] - crop_params['dim1'][0]
            gt[...,3] = gt[...,3] - crop_params['dim0'][0]
            new_line_gts[name]=gt
        new_point_gts={}
        for name, gt in point_gts.items():
            gt = gt[point_gt_match[name]][None,...] #add batch dim (?)
            gt[...,0] = gt[...,0] - crop_params['dim1'][0]
            gt[...,1] = gt[...,1] - crop_params['dim0'][0]
            new_point_gts[name]=gt
        ##print('pad-minus: {}'.format(timeit.default_timer()-##tic))

            #if 'start' in name:
            #    for j in range(min(10,gt.size(1))):
            #        ##print('a {},{}   {},{}'.format(gt[:,j,0],gt[:,j,1],gt[:,j,2],gt[:,j,3]))

        return {
            "img": org_img,
            "line_gt": new_line_gts,
            "point_gt": new_point_gts,
            "pixel_gt": pixel_gt
        }
class CropBoxTransform(object):
    def __init__(self, crop_params,rotate):
        self.crop_size = crop_params['crop_size']
        if type(self.crop_size) is int:
            self.crop_size = (self.crop_size,self.crop_size)
        self.random_crop_params = crop_params
        self.rotate=rotate
        if 'pad' in crop_params:
            pad_by = crop_params['pad']
        else:
            pad_by = min(self.crop_size)//2
        self.pad_params = ((pad_by,pad_by),(pad_by,pad_by),(0,0))
        #self.all_bbs=all_bbs
        if rotate:
            if 'rot_degree_std_dev' in crop_params:
                self.degree_std_dev = crop_params['rot_degree_std_dev']
            else:
                self.degree_std_dev = 1

    def __call__(self, sample,cropPoint=None):
        org_img = sample['img']
        bb_gt = sample['bb_gt']
        bb_ids = sample['bb_ids'] if 'bb_ids' in sample else None
        line_gts = sample['line_gt'] if 'line_gt' in sample else None
        point_gts = sample['point_gt'] if 'point_gt' in sample else None
        pixel_gt = sample['pixel_gt'] if 'pixel_gt' in sample else None
        query_bb = sample['query_bb'] if 'query_bb' in sample else None

        #rotation
        if self.rotate:
            amount = np.random.normal(0,self.degree_std_dev)
            M = cv2.getRotationMatrix2D((org_img.shape[1]/2,org_img.shape[0]/2),amount,1)
            #rotate image
            org_img = cv2.warpAffine(org_img,M,(org_img.shape[1],org_img.shape[0]))
            if len(org_img.shape)==2:
                org_img = org_img[:,:,None]
            if pixel_gt is not None:
                pixel_gt = cv2.warpAffine(pixel_gt,M,(pixel_gt.shape[1],pixel_gt.shape[0]))
                if len(pixel_gt.shape)==2:
                    pixel_gt = pixel_gt[:,:,None]
            #rotate points
            if bb_gt is not None:
                points = np.reshape(bb_gt[0,:,0:16],(-1,2)) #reshape all box points to vector of x,y pairs
                points = np.append(points,np.ones((points.shape[0],1)),axis=1) #append 1 to make homogeneous (x,y,1)
                points = M.dot(points.T).T #multiply rot matrix
                bb_gt[0,:,0:16] = np.reshape(points,(-1,16)) #reshape back to single vector for each bb

            if line_gts is not None:
                for name,gt in line_gts.items():
                    if gt is not None:
                        points = np.reshape(gt[0,:,0:4],(-1,2)) #reshape all line points to vector of x,y pairs
                        points = np.append(points,np.ones((points.shape[0],1)),axis=1) #append 1 to make homogeneous (x,y,1)
                        points = M.dot(points.T).T #multiply rot matrix
                        gt[0,:,0:4] = np.reshape(points,(-1,4)) #reshape back to single vector for each line
    
            if point_gts is not None:
                for name,gt in point_gts.items():
                    if gt is not None:
                        points = gt[0,:,0:2]
                        points = np.append(points,np.ones((points.shape[0],1)),axis=1) #append 1 to make homogeneous (x,y,1)
                        points = M.dot(points.T).T #multiply rot matrix
                        gt[0,:,0:2] = points

            if query_bb is not None:
                points = np.reshape(query_bb[0:16],(8,2)) #reshape all box points to vector of x,y pairs
                points = np.append(points,np.ones((points.shape[0],1)),axis=1) #append 1 to make homogeneous (x,y,1)
                points = M.dot(points.T).T #multiply rot matrix
                query_bb[0:16] = np.reshape(points,16) #reshape back to single vector

        #page_boundaries =
        pad_params = self.pad_params
        if org_img.shape[0]+pad_params[0][0]+pad_params[0][1] < self.crop_size[0]+1:
            diff = self.crop_size[0]+1-(org_img.shape[0]+pad_params[0][0]+pad_params[0][1])
            pad_byT = diff//2
            pad_byB = diff//2 + diff%2
            pad_params = ((pad_byT,pad_byB),)+pad_params[1:]
        if org_img.shape[1]+pad_params[1][0]+pad_params[1][1] < self.crop_size[1]+1:
            diff = self.crop_size[1]+1-(org_img.shape[1]+pad_params[1][0]+pad_params[1][1])
            pad_byL = diff//2
            pad_byR = diff//2 + diff%2
            pad_params = (pad_params[0],(pad_byL,pad_byR),pad_params[2])
        #print(pad_params)


        #pad out to allow random samples to take space off of the page
        ##tic=timeit.default_timer()
        #org_img = np.pad(org_img, self.pad_params, 'mean')
        if org_img.shape[2]==3:
            org_img = np.pad(org_img, pad_params, 'constant', constant_values=0) #zero, since that what Conv2d pads with
        else:
            org_img = np.pad(org_img, pad_params, 'constant', constant_values=0)
        if pixel_gt is not None:
            pixel_gt = np.pad(pixel_gt, pad_params, 'constant')
        ##print('pad: {}'.format(timeit.default_timer()-##tic))
        
        ##tic=timeit.default_timer()
        #corner points
        bb_gt[:,:,0] = bb_gt[:,:,0] + pad_params[1][0]
        bb_gt[:,:,1] = bb_gt[:,:,1] + pad_params[0][0]
        bb_gt[:,:,2] = bb_gt[:,:,2] + pad_params[1][0]
        bb_gt[:,:,3] = bb_gt[:,:,3] + pad_params[0][0]
        bb_gt[:,:,4] = bb_gt[:,:,4] + pad_params[1][0]
        bb_gt[:,:,5] = bb_gt[:,:,5] + pad_params[0][0]
        bb_gt[:,:,6 ] = bb_gt[:,:,6 ] + pad_params[1][0]
        bb_gt[:,:,7 ] = bb_gt[:,:,7 ] + pad_params[0][0]

        #cross/edge points
        bb_gt[:,:,8 ] = bb_gt[:,:,8 ] + pad_params[1][0]
        bb_gt[:,:,9 ] = bb_gt[:,:,9 ] + pad_params[0][0]
        bb_gt[:,:,10] = bb_gt[:,:,10] + pad_params[1][0]
        bb_gt[:,:,11] = bb_gt[:,:,11] + pad_params[0][0]
        bb_gt[:,:,12] = bb_gt[:,:,12] + pad_params[1][0]
        bb_gt[:,:,13] = bb_gt[:,:,13] + pad_params[0][0]
        bb_gt[:,:,14] = bb_gt[:,:,14] + pad_params[1][0]
        bb_gt[:,:,15] = bb_gt[:,:,15] + pad_params[0][0]

        if query_bb is not None:
            query_bb[8 ] = query_bb[8 ] + pad_params[1][0]
            query_bb[9 ] = query_bb[9 ] + pad_params[0][0]
            query_bb[10] = query_bb[10] + pad_params[1][0]
            query_bb[11] = query_bb[11] + pad_params[0][0]
            query_bb[12] = query_bb[12] + pad_params[1][0]
            query_bb[13] = query_bb[13] + pad_params[0][0]
            query_bb[14] = query_bb[14] + pad_params[1][0]
            query_bb[15] = query_bb[15] + pad_params[0][0]


        if point_gts is not None:
            for name, gt in point_gts.items():
                if gt is not None:
                    gt[:,:,0] = gt[:,:,0] + pad_params[1][0]
                    gt[:,:,1] = gt[:,:,1] + pad_params[0][0]
        if line_gts is not None:
            for name, gt in line_gts.items():
                if gt is not None:
                    gt[:,:,0] = gt[:,:,0] + pad_params[1][0]
                    gt[:,:,1] = gt[:,:,1] + pad_params[0][0]
                    gt[:,:,2] = gt[:,:,2] + pad_params[1][0]
                    gt[:,:,3] = gt[:,:,3] + pad_params[0][0]


        crop_params, org_img, pixel_gt, line_gt_match, point_gt_match, new_bb_gt, new_bb_ids, cropPoint = generate_random_crop(org_img, pixel_gt, line_gts, point_gts, self.random_crop_params, bb_gt=bb_gt, bb_ids=bb_ids, query_bb=query_bb, cropPoint=cropPoint)
        #print(crop_params)
        #print(gt_match)
        
        ##tic=timeit.default_timer()
        #new_bb_gt=bb_gt[bb_gt_match][None,...] #this is done in generate_random_crop() as it modified some bbs
        new_bb_gt=new_bb_gt[None,...] #this re-adds the batch dim
        new_bb_gt[...,0] = new_bb_gt[...,0] - crop_params['dim1'][0]
        new_bb_gt[...,1] = new_bb_gt[...,1] - crop_params['dim0'][0]
        new_bb_gt[...,2] = new_bb_gt[...,2] - crop_params['dim1'][0]
        new_bb_gt[...,3] = new_bb_gt[...,3] - crop_params['dim0'][0]
        new_bb_gt[...,4] = new_bb_gt[...,4] - crop_params['dim1'][0]
        new_bb_gt[...,5] = new_bb_gt[...,5] - crop_params['dim0'][0]
        new_bb_gt[...,6 ] = new_bb_gt[...,6 ] - crop_params['dim1'][0]
        new_bb_gt[...,7 ] = new_bb_gt[...,7 ] - crop_params['dim0'][0]
        #the cross/edge points are invalid now
        new_point_gts={}
        if point_gts is not None:
            for name, gt in point_gts.items():
                if gt is not None:
                    gt = gt[point_gt_match[name]][None,...] #add batch dim (?)
                    gt[...,0] = gt[...,0] - crop_params['dim1'][0]
                    gt[...,1] = gt[...,1] - crop_params['dim0'][0]
                    new_point_gts[name]=gt
        new_line_gts={}
        if line_gts is not None:
            for name, gt in line_gts.items():
                if gt is not None:
                    gt = gt[line_gt_match[name]][None,...] #add batch dim (?)
                    gt[...,0] = gt[...,0] - crop_params['dim1'][0]
                    gt[...,1] = gt[...,1] - crop_params['dim0'][0]
                    gt[...,2] = gt[...,2] - crop_params['dim1'][0]
                    gt[...,3] = gt[...,3] - crop_params['dim0'][0]
                    new_line_gts[name]=gt
        ##print('pad-minus: {}'.format(timeit.default_timer()-##tic))

            #if 'start' in name:
            #    for j in range(min(10,gt.size(1))):
            #        ##print('a {},{}   {},{}'.format(gt[:,j,0],gt[:,j,1],gt[:,j,2],gt[:,j,3]))

        return ({
            "img": org_img,
            "bb_gt": new_bb_gt,
            "bb_ids": new_bb_ids,
            "line_gt": new_line_gts,
            "point_gt": new_point_gts,
            "pixel_gt": pixel_gt
        }, cropPoint)
