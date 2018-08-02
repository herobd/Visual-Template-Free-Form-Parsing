import torch
import torch.utils.data
import numpy as np
import json
#from skimage import io
#from skimage import draw
#import skimage.transform as sktransform
import os
import math
from utils.util import get_image_size
import cv2

def collate(batch):

    batch_size = len(batch)
    imgs = []
    label_sizes = defaultdict(list)
    largest_label = {}
    for b in batch:
        if b is None:
            continue
        imgs.append(b["img"])
        for name,gt in b['sol_eol_gt'].items():
            if gt is None:
                label_sizes[name].append(0)
            else:
                label_sizes[name].append(gt.size(1)) 
    for name in b['sol_eol_gt']:
        largest_label[name] = max(label_sizes[name])

    if len(imgs) == 0:
        return None
    batch_size = len(imgs)


    labels = {}
    for name,gt in b['sol_eol_gt'].items():
        if largest_label[name] != 0:
            labels[name] = torch.zeros(batch_size, largest_label[name], 4)
            for i, b in enumerate(batch):
                if label_sizes[name][i] == 0:
                    continue
                labels[name][i, :label_sizes[name][i]] = gt

    imgs = torch.cat(imgs)

    return {
        'sol_eol_gt': labels,
        'img': imgs,
        "label_sizes": label_sizes
    }

def getStartEndGT(bbs):
    start_gt = np.zeros((1,len(bbs), 4), dtype=np.float32)
    end_gt = np.zeros((1,len(bbs), 4), dtype=np.float32)
    for bb in annotations['textBBs']:
        tlX = bb['poly_points'][0][0]
        tlY = bb['poly_points'][0][1]
        trX = bb['poly_points'][1][0]
        trY = bb['poly_points'][1][1]
        brX = bb['poly_points'][2][0]
        brY = bb['poly_points'][2][1]
        blX = bb['poly_points'][3][0]
        blY = bb['poly_points'][3][1]

        lX = (tlX+blX)/2.0
        lY = (tlY+blY)/2.0
        rX = (trX+brX)/2.0
        rY = (trY+brY)/2.0
        d=math.sqrt((lX-rX)**2 + (lY-rY)**2)

        hl = ((tlX-lX)*-(rY-lY) + (tlY-lY)*(rX-lX))/d #projection of half-left edge onto transpose horz run
        hr = ((brX-rX)*-(lY-rY) + (brY-rY)*(lX-rX))/d #projection of half-right edge onto transpose horz run
        h = (hl+hr)/2.0

        tX = lX + h*-(rY-lY)/d
        tY = lY + h*(rX-lX)/d
        bX = lX - h*-(rY-lY)/d
        bY = lY - h*(rX-lX)/d
        start_gt[:,j,0] = tX*s
        start_gt[:,j,1] = tY*s
        start_gt[:,j,2] = bX*s
        start_gt[:,j,3] = bY*s

        etX += rX-lX
        etY += rY-lY
        ebX += rX-lX
        ebY += rY-lY
        end_gt[:,j,0] = etX*s
        end_gt[:,j,1] = etY*s
        end_gt[:,j,2] = ebX*s
        end_gt[:,j,3] = ebY*s
    return start_gt, end_gt

class Forms(torch.utils.data.Dataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, transform=None, images=None):
        #if 'augmentation_params' in config['data_loader']:
        #    self.augmentation_params=config['data_loader']['augmentation_params']
        #else:
        #    self.augmentation_params=None
        self.cropToPage=config['data_loader']['crop_to_page']
        #patchSize=config['data_loader']['patch_size']
        if 'crop_params' in config['data_loader']:
            self.transform = CropTransform[config['data_loader']['crop_params']
        else:
            self.transform = None

        self.rescale_range = config['data_loader']['rescale_range']
        if images is not None:
            self.images=images
        else:
            centerJitterFactor=config['data_loader']['center_jitter']
            sizeJitterFactor=config['data_loader']['size_jitter']
            self.cropResize = self.__cropResizeF(patchSize,centerJitterFactor,sizeJitterFactor)
            with open(os.path.join(dirPath,'train_valid_test_split.json')) as f:
                #if split=='valid' or split=='validation':
                #    trainTest='train'
                #else:
                #    trainTest=split
                groupsToUse = json.loads(f.read())[split]
            self.images=[]
            for groupName, imageNames_noExt in groupsToUse.items():
                for imageName in imageNames_noExt:
                    path = os.path.join(dirPath,'groups',imageName)
                    if os.path.exists(path+'.json'):
                        images.append({'id':imageName, 'imagePath':path+'.jpg', 'annotationPath':path+'.json'})
                        # with open(path+'.json') as f:
                        #    annotations = json.loads(f.read())
                        #    imH = annotations['height']
                        #    imW = annotations['width']
                        #    #startCount=len(self.instances)
                        #    for bb in annotations['textBBs']:

        



    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        id = self.images[index]['id']
        imagePath = self.images[index]['imagePath']
        annotationPath = self.images[index]['annotationPath']
        with open(annotationPath) as annFile:
            annotations = json.loads(annFile.read())

        org_img = cv2.imread(imagePath)#/255.0
        if self.cropToPage:
            pageCorners = annotations['page_corners']
            xl = max(0,int(min(pageCorners['tl'],pageCorners['bl'])))
            xr = min(org_img.shape[1]-1,int(max(pageCorners['tr'],pageCorners['br'])))
            yt = max(0,int(min(pageCorners['tl'],pageCorners['tr'])))
            yb = min(org_img.shape[0]-1,int(max(pageCorners['bl'],pageCorners['br'])))
            org_img = org_img[yt:yb+1:xl:xr+1,:]
        target_dim1 = int(np.random.uniform(self.rescale_range[0], self.rescale_range[1]))
        s = target_dim1 / float(org_img.shape[1])
        target_dim0 = int(org_img.shape[0]/float(org_img.shape[1]) * target_dim1)
        org_img = cv2.resize(org_img,(target_dim1, target_dim0), interpolation = cv2.INTER_CUBIC)
        
        text_start_gt, text_end_gt = getStartEndGT(annotations['textBBs'])
        field_start_gt, field_end_gt = getStartEndGT(annotations['fieldBBs'])

        if self.transform is not None:
            out = self.transform({
                "img": org_img,
                "sol_eol_gt": {
                        "text_start_gt": text_start_gt,
                        "text_end_gt": text_end_gt,
                        "field_start_gt": field_start_gt,
                        "field_end_gt": field_end_gt
                        }
            })
            org_img = out['img']
            text_start_gt = out['text_start_gt']
            text_end_gt = out['text_end_gt']
            field_start_gt = out['field_start_gt']
            field_end_gt = out['field_end_gt']

            org_img = augmentation.apply_random_color_rotation(org_img)
            org_img = augmentation.apply_tensmeyer_brightness(org_img)


        img = org_img.transpose([2,1,0])[None,...]
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = 1.0 - img / 128.0 #ideally the median value would be 0
        #img = 1.0 - img / 255.0 #this way ink is on, page is off
        
        text_start_gt = None if text_start_gt.shape[1] == 0 else torch.from_numpy(text_start_gt)
        text_end_gt = None if text_end_gt.shape[1] == 0 else torch.from_numpy(text_end_gt)
        field_start_gt = None if field_start_gt.shape[1] == 0 else torch.from_numpy(field_start_gt)
        field_end_gt = None if field_end_gt.shape[1] == 0 else torch.from_numpy(field_end_gt)

        return {
            "img": img,
            "sol_eol_gt": {
                    "text_start_gt": text_start_gt,
                    "text_end_gt": text_end_gt,
                    "field_start_gt": field_start_gt,
                    "field_end_gt": field_end_gt
                    }
        }




