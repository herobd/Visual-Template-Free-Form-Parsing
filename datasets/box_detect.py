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
import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math
from utils.crop_transform import CropBoxTransform
from utils import augmentation
from collections import defaultdict, OrderedDict
from utils.forms_annotations import fixAnnotations, convertBBs, getBBWithPoints, getStartEndGT
import timeit

import utils.img_f as cv2


def collate(batch):

    ##tic=timeit.default_timer()
    batch_size = len(batch)
    imageNames=[]
    scales=[]
    imgs = []
    pixel_gt=[]
    max_h=0
    max_w=0
    line_label_sizes = defaultdict(list)
    point_label_sizes = defaultdict(list)
    largest_line_label = {}
    largest_point_label = {}
    bb_sizes=[]
    bb_dim=None
    line_dim=None
    if len(batch)==1:
        pairs = batch[0]['pairs']
    else:
        pairs = None
    
    for b in batch:
        if b is None:
            continue
        imageNames.append(b['imgName'])
        scales.append(b['scale'])
        imgs.append(b["img"])
        pixel_gt.append(b['pixel_gt'])
        max_h = max(max_h,b["img"].size(2))
        max_w = max(max_w,b["img"].size(3))
        gt = b['bb_gt']
        if gt is None:
            bb_sizes.append(0)
        else:
            bb_sizes.append(gt.size(1)) 
            bb_dim=gt.size(2)
        for name,gt in b['line_gt'].items():
            if gt is None:
                line_label_sizes[name].append(0)
            else:
                line_label_sizes[name].append(gt.size(1)) 
                line_dim = gt.size(2)
        for name,gt in b['point_gt'].items():
            if gt is None:
                point_label_sizes[name].append(0)
            else:
                point_label_sizes[name].append(gt.size(1)) 
    if len(imgs) == 0:
        return None

    largest_bb_count = max(bb_sizes)
    for name in b['point_gt']:
        largest_point_label[name] = max(point_label_sizes[name])
    for name in b['line_gt']:
        largest_line_label[name] = max(line_label_sizes[name])

    ##print(' col channels: {}'.format(len(imgs[0].size())))
    batch_size = len(imgs)

    resized_imgs = []
    resized_pixel_gt = []
    index=0
    for img in imgs:
        if img.size(2)<max_h or img.size(3)<max_w:
            resized = torch.zeros([1,img.size(1),max_h,max_w]).type(img.type())
            diff_h = max_h-img.size(2)
            pos_r = 0#np.random.randint(0,diff_h+1)
            diff_w = max_w-img.size(3)
            pos_c = 0#np.random.randint(0,diff_w+1)
            #if len(img.size())==3:
                #    resized[:,pos_r:pos_r+img.size(1), pos_c:pos_c+img.size(2)]=img
            #else:
                #    resized[pos_r:pos_r+img.size(1), pos_c:pos_c+img.size(2)]=img
            resized[:,:,pos_r:pos_r+img.size(2), pos_c:pos_c+img.size(3)]=img
            resized_imgs.append(resized)

            if pixel_gt[index] is not None:
                resized_gt = torch.zeros([1,pixel_gt[index].size(1),max_h,max_w]).type(pixel_gt[index].type())
                resized_gt[:,:,pos_r:pos_r+img.size(2), pos_c:pos_c+img.size(3)]=pixel_gt[index]
                resized_pixel_gt.append(resized_gt)
        else:
            resized_imgs.append(img)
            if pixel_gt[index] is not None:
                resized_pixel_gt.append(pixel_gt[index])
        index+=1

            

    if largest_bb_count != 0:
        bbs = torch.zeros(batch_size, largest_bb_count, bb_dim)
        numNeighbors = torch.zeros(batch_size, largest_bb_count)
        for i, b in enumerate(batch):
            gt = b['bb_gt']
            if bb_sizes[i] == 0:
                continue
            bbs[i, :bb_sizes[i]] = gt
            numNeighbors[i, :bb_sizes[i]] = b['num_neighbors']
    else:
        bbs=None
        numNeighbors=None

    line_labels = {}
    for name,count in largest_line_label.items():
        if count != 0:
            line_labels[name] = torch.zeros(batch_size, count, line_dim)
        else:
            line_labels[name]=None
    for i, b in enumerate(batch):
        for name,gt in b['line_gt'].items():
            if line_label_sizes[name][i] == 0:
                continue
            #print(line_label_sizes[name][i])
            #print(gt.shape)
            line_labels[name][i, :line_label_sizes[name][i]] = gt
    point_labels = {}
    for name,count in largest_point_label.items():
        if count != 0:
            point_labels[name] = torch.zeros(batch_size, count, 2)
        else:
            point_labels[name]=None
    for i, b in enumerate(batch):
        for name,gt in b['point_gt'].items():
            if point_label_sizes[name][i] == 0:
                continue
            #print(point_label_sizes[name][i])
            #print(gt.shape)
            point_labels[name][i, :point_label_sizes[name][i]] = gt

    imgs = torch.cat(resized_imgs)
    if len(resized_pixel_gt)==1:
        pixel_gt = resized_pixel_gt[0]
    elif len(resized_pixel_gt)>1:
        pixel_gt = torch.cat(resized_pixel_gt)
    else:
        pixel_gt = None


    ##print('collate: '+str(timeit.default_timer()-tic))
    return {
        'img': imgs,
        'bb_gt': bbs,
        'num_neighbors':numNeighbors,
        "bb_sizes": bb_sizes,
        'line_gt': line_labels,
        "line_label_sizes": line_label_sizes,
        'point_gt': point_labels,
        "point_label_sizes": point_label_sizes,
        'pixel_gt': pixel_gt,
        "imgName": imageNames,
        "scale": scales,
        'pairs': pairs #this is only used to save a new json
    }


class BoxDetectDataset(torch.utils.data.Dataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        #if 'augmentation_params' in config['data_loader']:
        #    self.augmentation_params=config['augmentation_params']
        #else:
        #    self.augmentation_params=None
        self.rotate = config['rotation'] if 'rotation' in config else True
        #patchSize=config['patch_size']
        if 'crop_params' in config and config['crop_params']:
            self.transform = CropBoxTransform(config['crop_params'],self.rotate)
        else:
            self.transform = None
        self.rescale_range = config['rescale_range']
        if type(self.rescale_range) is float or type(self.rescale_range) is int:
            self.rescale_range = [self.rescale_range,self.rescale_range]
        if self.rescale_range[0]==450:
            self.rescale_range[0]=0.2
        elif self.rescale_range[0]>1.0:
            self.rescale_range[0]=0.27
        if self.rescale_range[1]==800:
            self.rescale_range[1]=0.33
        elif self.rescale_range[1]>1.0:
            self.rescale_range[1]=0.27
        if 'cache_resized_images' in config:
            self.cache_resized = config['cache_resized_images']
            if self.cache_resized:
                self.cache_path = os.path.join(dirPath,'cache_'+str(self.rescale_range[1]))
                if not os.path.exists(self.cache_path):
                    os.mkdir(self.cache_path)
        else:
            self.cache_resized = False
        self.pixel_count_thresh = config['pixel_count_thresh'] if 'pixel_count_thresh' in config else 10000000
        self.max_dim_thresh = config['max_dim_thresh'] if 'max_dim_thresh' in config else 2700
        self.color = config['color'] if 'color' in config else True

        if 'random_image_aug' in config and config['random_image_aug'] is not None:
            self.useRandomAugProb = config['random_image_aug'] if type(config['random_image_aug']) is float else 0.05
            self.randomImageTypes = config['random_image_types'] if 'random_image_types' in config else ['blank','uniform','gaussian']
        else:
            self.useRandomAugProb = None

        self.coordConv = config['coord_conv'] if 'coord_conv' in config else False




    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        return self.getitem(index)
    def getitem(self,index,scaleP=None,cropPoint=None):
        if self.useRandomAugProb is not None and np.random.rand()<self.useRandomAugProb and scaleP is None and cropPoint is None:
            return self.getRandomImage()
        ##ticFull=timeit.default_timer()
        imagePath = self.images[index]['imagePath']
        imageName = self.images[index]['imageName']
        annotationPath = self.images[index]['annotationPath']
        #print(annotationPath)
        rescaled = self.images[index]['rescaled']
        with open(annotationPath) as annFile:
            annotations = json.loads(annFile.read())

        ##tic=timeit.default_timer()
        np_img = cv2.imread(imagePath, 1 if self.color else 0)#/255.0
        if np_img is None or np_img.shape[0]==0:
            print("ERROR, could not open "+imagePath)
            return self.__getitem__((index+1)%self.__len__())

        if scaleP is None:
            s = np.random.uniform(self.rescale_range[0], self.rescale_range[1])
        else:
            s = scaleP
        partial_rescale = s/rescaled
        if self.transform is None: #we're doing the whole image
            #this is a check to be sure we don't send too big images through
            pixel_count = partial_rescale*partial_rescale*np_img.shape[0]*np_img.shape[1]
            if pixel_count > self.pixel_count_thresh:
                partial_rescale = math.sqrt(partial_rescale*partial_rescale*self.pixel_count_thresh/pixel_count)
                print('{} exceed thresh: {}: {}, new {}: {}'.format(imageName,s,pixel_count,rescaled*partial_rescale,partial_rescale*partial_rescale*np_img.shape[0]*np_img.shape[1]))
                s = rescaled*partial_rescale


            max_dim = partial_rescale*max(np_img.shape[0],np_img.shape[1])
            if max_dim > self.max_dim_thresh:
                partial_rescale = partial_rescale*(self.max_dim_thresh/max_dim)
                print('{} exceed thresh: {}: {}, new {}: {}'.format(imageName,s,max_dim,rescaled*partial_rescale,partial_rescale*max(np_img.shape[0],np_img.shape[1])))
                s = rescaled*partial_rescale

        
        
        ##tic=timeit.default_timer()
        #np_img = cv2.resize(np_img,(target_dim1, target_dim0), interpolation = cv2.INTER_CUBIC)
        np_img = cv2.resize(np_img,(0,0),
                fx=partial_rescale,
                fy=partial_rescale,
                interpolation = cv2.INTER_CUBIC)
        if not self.color:
            np_img=np_img[...,None] #add 'color' channel
        ##print('resize: {}  [{}, {}]'.format(timeit.default_timer()-tic,np_img.shape[0],np_img.shape[1]))
        

        bbs,line_gts,point_gts,pixel_gt,numClasses,numNeighbors,pairs = self.parseAnn(np_img,annotations,s,imagePath)

        if self.coordConv: #add absolute position information
            xs = 255*np.arange(np_img.shape[1])/(np_img.shape[1]) 
            xs = np.repeat(xs[None,:,None],np_img.shape[0], axis=0)
            ys = 255*np.arange(np_img.shape[0])/(np_img.shape[0]) 
            ys = np.repeat(ys[:,None,None],np_img.shape[1], axis=1)
            np_img = np.concatenate((np_img,xs.astype(np_img.dtype),ys.astype(np_img.dtype)), axis=2)

        ##ticTr=timeit.default_timer()
        if self.transform is not None:
            pairs = None
            out, cropPoint = self.transform({
                "img": np_img,
                "bb_gt": bbs,
                "bb_auxs": numNeighbors,
                "line_gt": line_gts,
                "point_gt": point_gts,
                "pixel_gt": pixel_gt,
                
            }, cropPoint)
            np_img = out['img']
            bbs = out['bb_gt']
            numNeighbors = out['bb_auxs']
            #if 'table_points' in out['point_gt']:
            #    table_points = out['point_gt']['table_points']
            #else:
            #    table_points=None
            point_gts = out['point_gt']
            pixel_gt = out['pixel_gt']
            #start_of_line = out['line_gt']['start_of_line']
            #end_of_line = out['line_gt']['end_of_line']
            line_gts = out['line_gt']

            ##tic=timeit.default_timer()
            if self.color:
                np_img[:,:,:3] = augmentation.apply_random_color_rotation(np_img[:,:,:3])
                np_img[:,:,:3] = augmentation.apply_tensmeyer_brightness(np_img[:,:,:3])
            else:
                np_img[:,:,0:1] = augmentation.apply_tensmeyer_brightness(np_img[:,:,0:1])
            ##print('augmentation: {}'.format(timeit.default_timer()-tic))
        ##print('transfrm: {}  [{}, {}]'.format(timeit.default_timer()-ticTr,org_img.shape[0],org_img.shape[1]))

        #if len(np_img.shape)==2:
        #    img=np_img[None,None,:,:] #add "color" channel and batch
        #else:
        img = np_img.transpose([2,0,1])[None,...] #from [row,col,color] to [batch,color,row,col]
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = 1.0 - img / 128.0 #ideally the median value would be 0
        #img = 1.0 - img / 255.0 #this way ink is on, page is off
        if pixel_gt is not None:
            pixel_gt = pixel_gt.transpose([2,0,1])[None,...]
            pixel_gt = torch.from_numpy(pixel_gt)

        #start_of_line = None if start_of_line is None or start_of_line.shape[1] == 0 else torch.from_numpy(start_of_line)
        #end_of_line = None if end_of_line is None or end_of_line.shape[1] == 0 else torch.from_numpy(end_of_line)
        for name in line_gts:
            line_gts[name] = None if line_gts[name] is None or line_gts[name].shape[1] == 0 else torch.from_numpy(line_gts[name])
        
        #import pdb; pdb.set_trace()
        #bbs = None if bbs.shape[1] == 0 else torch.from_numpy(bbs)
        bbs = convertBBs(bbs,self.rotate,numClasses)
        if len(numNeighbors)>0:
            numNeighbors = torch.tensor(numNeighbors)[None,:] #add batch dim
        else:
            numNeighbors=None
            #start_of_line = convertLines(start_of_line,numClasses)
        #end_of_line = convertLines(end_of_line,numClasses)
        for name in point_gts:
            #if table_points is not None:
            #table_points = None if table_points.shape[1] == 0 else torch.from_numpy(table_points)
            if point_gts[name] is not None:
                point_gts[name] = None if point_gts[name].shape[1] == 0 else torch.from_numpy(point_gts[name])

        ##print('__getitem__: '+str(timeit.default_timer()-ticFull))
        if self.only_types is None:
            return {
                "img": img,
                "bb_gt": bbs,
                "num_neighbors": numNeighbors,
                "line_gt": line_gts,
                "point_gt": point_gts,
                "pixel_gt": pixel_gt,
                "imgName": imageName,
                "scale": s,
                "cropPoint": cropPoint,
                "pairs": pairs
                }
        else:
            if 'boxes' not in self.only_types or not self.only_types['boxes']:
                bbs=None
            line_gt={}
            if 'line' in self.only_types:
                for ent in self.only_types['line']:
                    if type(ent)==list:
                        toComb=[]
                        for inst in ent[1:]:
                            einst = line_gts[inst]
                            if einst is not None:
                                toComb.append(einst)
                        if len(toComb)>0:
                            comb = torch.cat(toComb,dim=1)
                            line_gt[ent[0]]=comb
                        else:
                            line_gt[ent[0]]=None
                    else:
                        line_gt[ent]=line_gts[ent]
            point_gt={}
            if 'point' in self.only_types:
                for ent in self.only_types['point']:
                    if type(ent)==list:
                        toComb=[]
                        for inst in ent[1:]:
                            einst = point_gts[inst]
                            if einst is not None:
                                toComb.append(einst)
                        if len(toComb)>0:
                            comb = torch.cat(toComb,dim=1)
                            point_gt[ent[0]]=comb
                        else:
                            line_gt[ent[0]]=None
                    else:
                        point_gt[ent]=point_gts[ent]
            pixel_gtR=None
            #for ent in self.only_types['pixel']:
            #    if type(ent)==list:
            #        comb = ent[1]
            #        for inst in ent[2:]:
            #            comb = (comb + inst)==2 #:eq(2) #pixel-wise AND
            #        pixel_gt[ent[0]]=comb
            #    else:
            #        pixel_gt[ent]=eval(ent)
            if 'pixel' in self.only_types:# and self.only_types['pixel'][0]=='table_pixels':
                pixel_gtR=pixel_gt

            return {
                "img": img,
                "bb_gt": bbs,
                "num_neighbors": numNeighbors,
                "line_gt": line_gt,
                "point_gt": point_gt,
                "pixel_gt": pixel_gtR,
                "imgName": imageName,
                "scale": s,
                "cropPoint": cropPoint,
                "pairs": pairs,
                }


    #this is a funny kind of augmentation where random images are show 
    #simply to improve generalization. Also may help decrease false positives?
    def getRandomImage(self):
        assert(self.transform is not None)
        w = self.transform.crop_size[1]
        h= self.transform.crop_size[0]
        if self.color:
            shape = (3,h,w)
        else:
            shape = (1,h,w)
        typ = np.random.choice(self.randomImageTypes)
        center=np.random.uniform(-1,1)
        if typ=='blank': #blank
            image = torch.FloatTensor(*shape).fill_(center)
        elif typ=='uniform': #uniform random
            maxRange = 1-abs(center)
            second = np.random.uniform(0,maxRange)
            image = torch.FloatTensor(*shape).uniform_(center-maxRange,center+maxRange)
        elif typ=='gaussian': #guassian
            maxRange = 1-abs(center)
            second = np.random.uniform(0,maxRange)
            image = torch.FloatTensor(*shape).normal_(center,maxRange)
        image = image[None,:,:]#add batch channel
    
        return {
            "img": image,
            "bb_gt": None,
            "num_neighbors": None,
            "line_gt": {},
            "point_gt": {},
            "pixel_gt": None,
            "imgName": 'rand_'+typ,
            "scale": 1.0,
            "cropPoint": (0,0),
            "pairs": None,
            }

    def cluster(self,k,sample_count,outPath):
        def makePointsAndRects(h,w,r=None):
            if r is None:
                return np.array([-w/2.0,0,w/2.0,0,0,-h/2.0,0,h/2.0, 0,0, 0, h,w])
            else:
                lx= -math.cos(r)*w
                ly= -math.sin(r)*w
                rx= math.cos(r)*w
                ry= math.sin(r)*w
                tx= math.sin(r)*h
                ty= -math.cos(r)*h
                bx= -math.sin(r)*h
                by= math.cos(r)*h
                return np.array([lx,ly,rx,ry,tx,ty,bx,by, 0,0, r, h,w])
        meanH=62.42
        stdH=87.31
        meanW=393.03
        stdW=533.53
        ratios=[4.0,7.18,11.0,15.0,19.0,27.0]
        pointsAndRects=[]
        for inst in self.images:
            annotationPath = inst['annotationPath']
            #rescaled = inst['rescaled']
            with open(annotationPath) as annFile:
                annotations = json.loads(annFile.read())
            for i in range(1):#sample_count):
                if i==0:
                    s = (self.rescale_range[0]+self.rescale_range[1])/2
                else:
                    s = np.random.uniform(self.rescale_range[0], self.rescale_range[1])
                #partial_rescale = s/rescaled
                #bbs = getBBWithPoints(annotations['byId'].values(),s)
                bbs,line_gts,point_gts,pixel_gt,numClasses = self.parseAnn(np.array([[0,0],[0,0]]),annotations,s,'')
                #field_bbs = self.getBBGT(annotations['fieldBBs'],s,fields=True)
                #bbs = np.concatenate([text_bbs,field_bbs],axis=1)
                assert(not np.isnan(bbs).any())
                bbs = convertBBs(bbs,self.rotate,numClasses).numpy()[0]
                assert(not np.isnan(bbs).any())
                cos_rot = np.cos(bbs[:,2])
                sin_rot = np.sin(bbs[:,2])
                p_left_x = -cos_rot*bbs[:,4]
                p_left_y = -sin_rot*bbs[:,4]
                p_right_x = cos_rot*bbs[:,4]
                p_right_y = sin_rot*bbs[:,4]
                p_top_x = sin_rot*bbs[:,3]
                p_top_y = -cos_rot*bbs[:,3]
                p_bot_x = -sin_rot*bbs[:,3]
                p_bot_y = cos_rot*bbs[:,3]
                points = np.stack([p_left_x,p_left_y,p_right_x,p_right_y,p_top_x,p_top_y,p_bot_x,p_bot_y],axis=1)
                pointsAndRects.append(np.concatenate([points,bbs[:,:5]],axis=1))
        pointsAndRects = np.concatenate(pointsAndRects,axis=0)
        #all_points = pointsAndRects[:,0:8]
        #all_heights = pointsAndRects[:,11]
        #all_widths = pointsAndRects[:,12]
        
        bestDistsFromMean=None
        for attempt in range(20 if k>0 else 1):
            if k>0:
                randomIndexes = np.random.randint(0,pointsAndRects.shape[0],(k))
                means=pointsAndRects[randomIndexes]
            else:
                #minH=5
                #minW=5
                means=[]

                ##smaller than mean
                #for step in range(5):
                #    height = minH + (meanH-minH)*(step/5.0)
                #    width = minW + (meanW-minW)*(step/5.0)
                #    for ratio in ratios:
                #        means.append(makePointsAndRects(height,ratio*height))
                #        means.append(makePointsAndRects(width/ratio,width))
                #for stddev in range(0,5):
                #    for step in range(5-stddev):
                #        height = meanH + stddev*stdH + stdH*(step/(5.0-stddev))
                #        width = meanW + stddev*stdW + stdW*(step/(5.0-stddev))
                #        for ratio in ratios:
                #            means.append(makePointsAndRects(height,ratio*height))
                #            means.append(makePointsAndRects(width/ratio,width))
                rots = [0,math.pi/2,math.pi,1.5*math.pi]
                if self.rotate:
                    for height in np.linspace(15,200,num=4):
                        for width in np.linspace(30,1200,num=4):
                            for rot in rots:
                                means.append(makePointsAndRects(height,width,rot))
                        #long boxes
                    for width in np.linspace(1600,4000,num=3):
                        #for height in np.linspace(30,100,num=3):
                        #    for rot in rots:
                        #        means.append(makePointsAndRects(height,width,rot))
                        for rot in rots:
                            means.append(makePointsAndRects(50,width,rot))
                else:
                    #rotated boxes
                    #for height in np.linspace(13,300,num=4):
                    for height in np.linspace(13,300,num=3):
                        means.append(makePointsAndRects(height,20))
                    #general boxes
                    #for height in np.linspace(15,200,num=4):
                        #for width in np.linspace(30,1200,num=4):
                    for height in np.linspace(15,200,num=2):
                        for width in np.linspace(30,1200,num=3):
                            means.append(makePointsAndRects(height,width))
                    #long boxes
                    for width in np.linspace(1600,4000,num=3):
                        #for height in np.linspace(30,100,num=3):
                        #    means.append(makePointsAndRects(height,width))
                        means.append(makePointsAndRects(50,width))

                k=len(means)
                print('K: {}'.format(k))
                means = np.stack(means,axis=0)
            #pointsAndRects [0:p_left_x, 1:p_left_y,2:p_right_x,3:p_right_y,4:p_top_x,5:p_top_y,6:p_bot_x,7:p_bot_y, 8:xc, 9:yc, 10:rot, 11:h, 12:w
            cluster_centers=means
            distsFromMean=None
            prevDistsFromMean=None
            for iteration in range(100000): #intended to break out
                print('attempt:{}, bestDistsFromMean:{}, iteration:{}, bestDistsFromMean:{}'.format(attempt,bestDistsFromMean,iteration,prevDistsFromMean), end='\r')
                #means_points = means[:,0:8]
                #means_heights = means[:,11]
                #means_widths = means[:,12]
                # = groups = assignGroups(means,pointsAndRects)
                expanded_all_points = pointsAndRects[:,None,0:8]
                expanded_all_heights = pointsAndRects[:,None,11]
                expanded_all_widths = pointsAndRects[:,None,12]

                expanded_means_points = means[None,:,0:8]
                expanded_means_heights = means[None,:,11]
                expanded_means_widths = means[None,:,12]

                #expanded_all_points = expanded_all_points.expand(all_points.shape[0], all_points.shape[1], means_points.shape[1], all_points.shape[2])
                expanded_all_points = np.tile(expanded_all_points,(1,means.shape[0],1))
                expanded_all_heights = np.tile(expanded_all_heights,(1,means.shape[0]))
                expanded_all_widths = np.tile(expanded_all_widths,(1,means.shape[0]))
                #expanded_means_points = expanded_means_points.expand(means_points.shape[0], all_points.shape[0], means_points.shape[0], means_points.shape[2])
                expanded_means_points = np.tile(expanded_means_points,(pointsAndRects.shape[0],1,1))
                expanded_means_heights = np.tile(expanded_means_heights,(pointsAndRects.shape[0],1))
                expanded_means_widths = np.tile(expanded_means_widths,(pointsAndRects.shape[0],1))

                point_deltas = (expanded_all_points - expanded_means_points)
                #avg_heights = ((expanded_means_heights+expanded_all_heights)/2)
                #avg_widths = ((expanded_means_widths+expanded_all_widths)/2)
                avg_heights=avg_widths = (expanded_means_heights+expanded_all_heights+expanded_means_widths+expanded_all_widths)/4
                #print point_deltas
                assert(not np.isnan(point_deltas).any())

                normed_difference = (
                    np.linalg.norm(point_deltas[:,:,0:2],2,2)/avg_widths +
                    np.linalg.norm(point_deltas[:,:,2:4],2,2)/avg_widths +
                    np.linalg.norm(point_deltas[:,:,4:6],2,2)/avg_heights +
                    np.linalg.norm(point_deltas[:,:,6:8],2,2)/avg_heights
                    )**2
                #print normed_difference
                #import pdb; pdb.set_trace()
                assert(not np.isnan(normed_difference).any())

                groups = normed_difference.argmin(1) #this should list the mean (index) for each element of all
                distsFromMean = normed_difference.min(1).mean()
                if math.isnan(distsFromMean):
                    import pdb; pdb.set_trace()
                if prevDistsFromMean is not None and distsFromMean >= prevDistsFromMean:
                    break
                prevDistsFromMean = distsFromMean

                #means = computeMeans(groups,pointsAndRects)
                #means = np.zeros(k,13)
                for ki in range(k):
                    selected = (groups==ki)[:,None]
                    numSel = float(selected.sum())
                    if (numSel==0):
                        break
                    means[ki,:] = (pointsAndRects*np.tile(selected,(1,13))).sum(0)/numSel
            if bestDistsFromMean is None or distsFromMean<bestDistsFromMean:
                bestDistsFromMean = distsFromMean
                cluster_centers=means
        #cluster_centers=means
        dH=600
        dW=3000
        draw = np.zeros([dH,dW,3],dtype=np.float)
        toWrite = []
        final_k=k
        for ki in range(k):
            pop = (groups==ki).sum().item()
            if pop>2:
                color = np.random.uniform(0.2,1,3).tolist()
                #d=math.sqrt(mean[ki,11]**2 + mean[ki,12]**2)
                #theta = math.atan2(mean[ki,11],mean[ki,12]) + mean[ki,10]
                h=cluster_centers[ki,11]
                w=cluster_centers[ki,12]
                rot=cluster_centers[ki,10]
                toWrite.append({'height':h.item(),'width':w.item(),'rot':rot.item(),'popularity':pop})
                tr = ( int(math.cos(rot)*w-math.sin(rot)*h)+dW//2,   int(math.sin(rot)*w+math.cos(rot)*h)+dH//2 )
                tl = ( int(math.cos(rot)*-w-math.sin(rot)*h)+dW//2,  int(math.sin(rot)*-w+math.cos(rot)*h)+dH//2 )
                br = ( int(math.cos(rot)*w-math.sin(rot)*-h)+dW//2,  int(math.sin(rot)*w+math.cos(rot)*-h)+dH//2 )
                bl = ( int(math.cos(rot)*-w-math.sin(rot)*-h)+dW//2, int(math.sin(rot)*-w+math.cos(rot)*-h)+dH//2 )
                
                cv2.line(draw,tl,tr,color)
                cv2.line(draw,tr,br,color)
                cv2.line(draw,br,bl,color)
                cv2.line(draw,bl,tl,color,2)
            else:
                final_k-=1
        
        #print(toWrite)
        with open(outPath.format(final_k),'w') as out:
            out.write(json.dumps(toWrite))
            print('saved '+outPath.format(final_k))
        cv2.imshow('clusters',draw)
        cv2.waitKey()


