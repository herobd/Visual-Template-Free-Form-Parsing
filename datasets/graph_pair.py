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
    assert(len(batch)==1)
    return batch[0]


class GraphPairDataset(torch.utils.data.Dataset):
    """
    Class for reading dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        #if 'augmentation_params' in config['data_loader']:
        #    self.augmentation_params=config['augmentation_params']
        #else:
        #    self.augmentation_params=None
        self.color = config['color'] if 'color' in config else True
        self.rotate = config['rotation'] if 'rotation' in config else False
        #patchSize=config['patch_size']
        if 'crop_params' in config and config['crop_params'] is not None:
            self.transform = CropBoxTransform(config['crop_params'],self.rotate)
        else:
            self.transform = None
        self.rescale_range = config['rescale_range']
        if type(self.rescale_range) is float:
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






    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        return self.getitem(index)
    def getitem(self,index,scaleP=None,cropPoint=None):
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
                fy=partial_rescale)
                #interpolation = cv2.INTER_CUBIC)
        if not self.color:
            np_img=np_img[...,None] #add 'color' channel
        ##print('resize: {}  [{}, {}]'.format(timeit.default_timer()-tic,np_img.shape[0],np_img.shape[1]))
        
        ##tic=timeit.default_timer()

        bbs,ids,numClasses,trans = self.parseAnn(annotations,s)

        #start_of_line, end_of_line = getStartEndGT(annotations['byId'].values(),s)
        #Try:
        #    table_points, table_pixels = self.getTables(
        #            fieldBBs,
        #            s, 
        #            np_img.shape[0], 
        #            np_img.shape[1],
        #            annotations['samePairs'])
        #Except Exception as inst:
        #    if imageName not in self.errors:
        #        table_points=None
        #        table_pixels=None
        #        print(inst)
        #        print('Table error on: '+imagePath)
        #        self.errors.append(imageName)


        #pixel_gt = table_pixels

        ##ticTr=timeit.default_timer()
        if self.transform is not None:
            out, cropPoint = self.transform({
                "img": np_img,
                "bb_gt": bbs,
                'bb_auxs':ids,
                #"line_gt": {
                #    "start_of_line": start_of_line,
                #    "end_of_line": end_of_line
                #    },
                #"point_gt": {
                #        "table_points": table_points
                #        },
                #"pixel_gt": pixel_gt,
                
            }, cropPoint)
            np_img = out['img']
            bbs = out['bb_gt']
            ids= out['bb_auxs']


            ##tic=timeit.default_timer()
            if np_img.shape[2]==3:
                np_img = augmentation.apply_random_color_rotation(np_img)
                np_img = augmentation.apply_tensmeyer_brightness(np_img)
            else:
                np_img = augmentation.apply_tensmeyer_brightness(np_img)
            ##print('augmentation: {}'.format(timeit.default_timer()-tic))
        ##print('transfrm: {}  [{}, {}]'.format(timeit.default_timer()-ticTr,org_img.shape[0],org_img.shape[1]))
        pairs=set()
        #import pdb;pdb.set_trace()
        numNeighbors=[0]*len(ids)
        for index1,id in enumerate(ids): #updated
            responseBBIdList = self.getResponseBBIdList(id,annotations)
            for bbId in responseBBIdList:
                try:
                    index2 = ids.index(bbId)
                    #adjMatrix[min(index1,index2),max(index1,index2)]=1
                    pairs.add((min(index1,index2),max(index1,index2)))
                    numNeighbors[index1]+=1
                except ValueError:
                    pass
        #ones = torch.ones(len(pairs))
        #if len(pairs)>0:
        #    pairs = torch.LongTensor(list(pairs)).t()
        #else:
        #    pairs = torch.LongTensor(pairs)
        #adjMatrix = torch.sparse.FloatTensor(pairs,ones,(len(ids),len(ids))) # This is an upper diagonal matrix as pairings are bi-directional

        #if len(np_img.shape)==2:
        #    img=np_img[None,None,:,:] #add "color" channel and batch
        #else:
        img = np_img.transpose([2,0,1])[None,...] #from [row,col,color] to [batch,color,row,col]
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = 1.0 - img / 128.0 #ideally the median value would be 0
        #if pixel_gt is not None:
        #    pixel_gt = pixel_gt.transpose([2,0,1])[None,...]
        #    pixel_gt = torch.from_numpy(pixel_gt)

        #start_of_line = None if start_of_line is None or start_of_line.shape[1] == 0 else torch.from_numpy(start_of_line)
        #end_of_line = None if end_of_line is None or end_of_line.shape[1] == 0 else torch.from_numpy(end_of_line)
        
        bbs = convertBBs(bbs,self.rotate,numClasses)
        if len(numNeighbors)>0:
            numNeighbors = torch.tensor(numNeighbors)[None,:] #add batch dim
        else:
            numNeighbors=None
        #if table_points is not None:
        #    table_points = None if table_points.shape[1] == 0 else torch.from_numpy(table_points)

        return {
                "img": img,
                "bb_gt": bbs,
                "num_neighbors": numNeighbors,
                "adj": pairs,#adjMatrix,
                "imgName": imageName,
                "scale": s,
                "cropPoint": cropPoint,
                "transcription": [trans[id] for id in ids if id in trans]
                }


