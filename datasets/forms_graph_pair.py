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

import cv2

SKIP=['174']#['193','194','197','200']
ONE_DONE=[]


def polyIntersect(poly1, poly2):
    prevPoint = poly1[-1]
    for point in poly1:
        perpVec = np.array([ -(point[1]-prevPoint[1]), point[0]-prevPoint[0] ])
        perpVec = perpVec/np.linalg.norm(perpVec)
        
        maxPoly1=np.dot(perpVec,poly1[0])
        minPoly1=maxPoly1
        for p in poly1:
            p_onLine = np.dot(perpVec,p)
            maxPoly1 = max(maxPoly1,p_onLine)
            minPoly1 = min(minPoly1,p_onLine)
        maxPoly2=np.dot(perpVec,poly2[0])
        minPoly2=maxPoly2
        for p in poly2:
            p_onLine = np.dot(perpVec,p)
            maxPoly2 = max(maxPoly2,p_onLine)
            minPoly2 = min(minPoly2,p_onLine)

        if (maxPoly1<minPoly2 or minPoly1>maxPoly2):
            return False
        prevPoint = point
    return True

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def lineIntersection(lineA, lineB, threshA_low=10, threshA_high=10, threshB_low=10, threshB_high=10, both=False):
    a1=lineA[0]
    a2=lineA[1]
    b1=lineB[0]
    b2=lineB[1]
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    point = (num / denom.astype(float))*db + b1
    #check if it is on atleast one line segment
    vecA = da/np.linalg.norm(da)
    p_A = np.dot(point,vecA)
    a1_A = np.dot(a1,vecA)
    a2_A = np.dot(a2,vecA)

    vecB = db/np.linalg.norm(db)
    p_B = np.dot(point,vecB)
    b1_B = np.dot(b1,vecB)
    b2_B = np.dot(b2,vecB)
    
    ###rint('A:{},  B:{}, int p:{}'.format(lineA,lineB,point))
    ###rint('{:.0f}>{:.0f} and {:.0f}<{:.0f}  and/or  {:.0f}>{:.0f} and {:.0f}<{:.0f} = {} {} {}'.format((p_A+threshA_low),(min(a1_A,a2_A)),(p_A-threshA_high),(max(a1_A,a2_A)),(p_B+threshB_low),(min(b1_B,b2_B)),(p_B-threshB_high),(max(b1_B,b2_B)),(p_A+threshA_low>min(a1_A,a2_A) and p_A-threshA_high<max(a1_A,a2_A)),'and' if both else 'or',(p_B+threshB_low>min(b1_B,b2_B) and p_B-threshB_high<max(b1_B,b2_B))))
    if both:
        if ( (p_A+threshA_low>min(a1_A,a2_A) and p_A-threshA_high<max(a1_A,a2_A)) and
             (p_B+threshB_low>min(b1_B,b2_B) and p_B-threshB_high<max(b1_B,b2_B)) ):
            return point
    else:
        if ( (p_A+threshA_low>min(a1_A,a2_A) and p_A-threshA_high<max(a1_A,a2_A)) or
             (p_B+threshB_low>min(b1_B,b2_B) and p_B-threshB_high<max(b1_B,b2_B)) ):
            return point
    return None

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
    else:
        bbs=None
    for i, b in enumerate(batch):
        gt = b['bb_gt']
        if bb_sizes[i] == 0:
            continue
        bbs[i, :bb_sizes[i]] = gt

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
        "bb_sizes": bb_sizes,
        'line_gt': line_labels,
        "line_label_sizes": line_label_sizes,
        'point_gt': point_labels,
        "point_label_sizes": point_label_sizes,
        'pixel_gt': pixel_gt,
        "imgName": imageNames,
        "scale": scales
    }


class FormsBoxDetect(torch.utils.data.Dataset):
    """
    Class for reading forms dataset and creating starting and ending gt
    """


    def __init__(self, dirPath=None, split=None, config=None, images=None):
        #if 'augmentation_params' in config['data_loader']:
        #    self.augmentation_params=config['augmentation_params']
        #else:
        #    self.augmentation_params=None
        self.cropToPage=config['crop_to_page']
        self.rotate = config['rotation'] if 'rotation' in config else True
        #patchSize=config['patch_size']
        if 'crop_params' in config:
            self.transform = CropBoxTransform(config['crop_params'],self.rotate)
        else:
            self.transform = None
        self.rescale_range = config['rescale_range']
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
        if 'only_types' in config:
            self.only_types = config['only_types']
        else:
            self.only_types=None
        #print( self.only_types)
        if 'swap_circle' in config:
            self.swapCircle = config['swap_circle']
        else:
            self.swapCircle = False
        self.color = config['color'] if 'color' in config else True

        self.simple_dataset = config['simple_dataset'] if 'simple_dataset' in config else False

        if images is not None:
            self.images=images
        else:
            if self.simple_dataset:
                splitFile = 'simple_train_valid_test_split.json'
            else:
                splitFile = 'train_valid_test_split.json'
            with open(os.path.join(dirPath,splitFile)) as f:
                #if split=='valid' or split=='validation':
                #    trainTest='train'
                #else:
                #    trainTest=split
                groupsToUse = json.loads(f.read())[split]
            self.images=[]
            groupNames = list(groupsToUse.keys())
            groupNames.sort()
            for groupName in groupNames:
                imageNames=groupsToUse[groupName]
                #print('{} {}'.format(groupName, imageNames))
                #oneonly=False
                if groupName in SKIP:
                    print('Skipped group {}'.format(groupName))
                    continue
                #    if groupName in ONE_DONE:
                #        oneonly=True
                #        with open(os.path.join(dirPath,'groups',groupName,'template'+groupName+'.json')) as f:
                #            T_annotations = json.loads(f.read())
                #    else:
                for imageName in imageNames:
                    #if oneonly and T_annotations['imageFilename']!=imageName:
                    #    #print('skipped {} {}'.format(imageName,groupName))
                    #    continue
                    #elif oneonly:
                    #    print('only {} from {}'.format(imageName,groupName))
                    org_path = os.path.join(dirPath,'groups',groupName,imageName)
                    if self.cache_resized:
                        path = os.path.join(self.cache_path,imageName)
                    else:
                        path = org_path
                    jsonPath = org_path[:org_path.rfind('.')]+'.json'
                    #print(jsonPath)
                    if os.path.exists(jsonPath):
                        rescale=1.0
                        if self.cache_resized:
                            rescale = self.rescale_range[1]
                            if not os.path.exists(path):
                                org_img = cv2.imread(org_path)
                                if org_img is None:
                                    print('WARNING, could not read {}'.format(org_img))
                                    continue
                                #target_dim1 = self.rescale_range[1]
                                #target_dim0 = int(org_img.shape[0]/float(org_img.shape[1]) * target_dim1)
                                #resized = cv2.resize(org_img,(target_dim1, target_dim0), interpolation = cv2.INTER_CUBIC)
                                resized = cv2.resize(org_img,(0,0),
                                        fx=self.rescale_range[1], 
                                        fy=self.rescale_range[1], 
                                        interpolation = cv2.INTER_CUBIC)
                                cv2.imwrite(path,resized)
                                #rescale = target_dim1/float(org_img.shape[1])
                        #elif self.cache_resized:
                            #print(jsonPath)
                            #with open(jsonPath) as f:
                            #    annotations = json.loads(f.read())
                            #imW = annotations['width']

                            #target_dim1 = self.rescale_range[1]
                            #rescale = target_dim1/float(imW)
                        #print('addint {}'.format(imageName))
                        self.images.append({'id':imageName, 'imagePath':path, 'annotationPath':jsonPath, 'rescaled':rescale, 'imageName':imageName[:imageName.rfind('.')]})
                    #else:
                    #    print('couldnt find {}'.format(jsonPath))
                            
                        # with open(path+'.json') as f:
                        #    annotations = json.loads(f.read())
                        #    imH = annotations['height']
                        #    imW = annotations['width']
                        #    #startCount=len(self.instances)
                        #    for bb in annotations['textBBs']:
        
        self.no_blanks = config['no_blanks'] if 'no_blanks' in config else False
        self.use_paired_class = config['use_paired_class'] if 'use_paired_class' in config else False
        if 'no_print_fields' in config:
            self.no_print_fields = config['no_print_fields']
        else:
            self.no_print_fields = False
        self.no_graphics =  config['no_graphics'] if 'no_graphics' in config else False
        self.only_opposite_pairs = config['only_opposite_pairs'] if 'only_opposite_pairs' in config else False
        self.errors=[]



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
        fieldBBs = annotations['fieldBBs']
        fixAnnotations(self,annotations)

        ##tic=timeit.default_timer()
        np_img = cv2.imread(imagePath, 1 if self.color else 0)#/255.0
        if np_img is None or np_img.shape[0]==0:
            print("ERROR, could not open "+imagePath)
            return self.__getitem__((index+1)%self.__len__())
        ##print('imread: {}  [{}, {}]'.format(timeit.default_timer()-tic,np_img.shape[0],np_img.shape[1]))
        ##print('       channels : {}'.format(len(np_img.shape)))
        if self.cropToPage:
            print('Not implemented')
            assert(False)
            pageCorners = annotations['page_corners']
            xl = max(0,int(rescaled*min(pageCorners['tl'],pageCorners['bl'])))
            xr = min(np_img.shape[1]-1,int(rescaled*max(pageCorners['tr'],pageCorners['br'])))
            yt = max(0,int(rescaled*min(pageCorners['tl'],pageCorners['tr'])))
            yb = min(np_img.shape[0]-1,int(rescaled*max(pageCorners['bl'],pageCorners['br'])))
            np_img = np_img[yt:yb+1,xl:xr+1,:]
        #target_dim1 = int(np.random.uniform(self.rescale_range[0], self.rescale_range[1]))
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
        
        ##tic=timeit.default_timer()


        bbs = getBBWithPoints(annotations['byId'].values(),s,useBlankClass=(not self.no_blanks),usePairedClass=self.use_paired_class)
        numClasses = bbs.shape[2]-16
        adjMatrix = torch.sparse.IntTensor((len(annotations['byId']),len(annotations['byId']))) # This is an upper diagonal matrix as pairings are bi-directional
        index1=0
        for id in ids:
            bb = annotations['byId'][id]
            if not self.onlyFormStuff or ('paired' in bb and bb['paired']):
                responseBBList = self.__getResponseBBList(id,annotations)
                for bb in resposeBBList:
                    index2 = ids.indexOf(bb['id'])
                    adjMatrix[min(index1,index2),max(index1,index2)]=1
            index1+=1
        #field_bbs = getBBWithPoints(annotations['fieldBBs'],s)
        #bbs = np.concatenate([text_bbs,field_bbs],axis=1) #has batch dim
        start_of_line, end_of_line = getStartEndGT(annotations['byId'].values(),s)
        try:
            table_points, table_pixels = self.getTables(
                    fieldBBs,
                    s, 
                    np_img.shape[0], 
                    np_img.shape[1],
                    annotations['samePairs'])
        except Exception as inst:
            if imageName not in self.errors:
                table_points=None
                table_pixels=None
                print(inst)
                print('Table error on: '+imagePath)
                self.errors.append(imageName)

        ##print('getStartEndGt: '+str(timeit.default_timer()-tic))

        pixel_gt = table_pixels

        ##ticTr=timeit.default_timer()
        if self.transform is not None:
            out, cropPoint = self.transform({
                "img": np_img,
                "bb_gt": bbs,
                "line_gt": {
                    "start_of_line": start_of_line,
                    "end_of_line": end_of_line
                    },
                "point_gt": {
                        "table_points": table_points
                        },
                "pixel_gt": pixel_gt,
                
            }, cropPoint)
            np_img = out['img']
            bbs = out['bb_gt']
            if 'table_points' in out['point_gt']:
                table_points = out['point_gt']['table_points']
            else:
                table_points=None
            pixel_gt = out['pixel_gt']
            start_of_line = out['line_gt']['start_of_line']
            end_of_line = out['line_gt']['end_of_line']

            ##tic=timeit.default_timer()
            if np_img.shape[2]==3:
                np_img = augmentation.apply_random_color_rotation(np_img)
                np_img = augmentation.apply_tensmeyer_brightness(np_img)
            else:
                np_img = augmentation.apply_tensmeyer_brightness(np_img)
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

        start_of_line = None if start_of_line is None or start_of_line.shape[1] == 0 else torch.from_numpy(start_of_line)
        end_of_line = None if end_of_line is None or end_of_line.shape[1] == 0 else torch.from_numpy(end_of_line)
        
        #import pdb; pdb.set_trace()
        #bbs = None if bbs.shape[1] == 0 else torch.from_numpy(bbs)
        bbs = convertBBs(bbs,self.rotate,numClasses)
        #start_of_line = convertLines(start_of_line,numClasses)
        #end_of_line = convertLines(end_of_line,numClasses)

        if table_points is not None:
            table_points = None if table_points.shape[1] == 0 else torch.from_numpy(table_points)

        ##print('__getitem__: '+str(timeit.default_timer()-ticFull))
        if self.only_types is None:
            return {
                "img": img,
                "bb_gt": bbs,
                "line_gt": {
                    "start_of_line": start_of_line,
                    "end_of_line": end_of_line
                    },
                "point_gt": {
                        "table_points":table_points
                        },
                "pixel_gt": pixel_gt,
                "imgName": imageName,
                "scale": s,
                "cropPoint": cropPoint
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
                            einst = eval(inst)
                            if einst is not None:
                                toComb.append(einst)
                        if len(toComb)>0:
                            comb = torch.cat(toComb,dim=1)
                            line_gt[ent[0]]=comb
                        else:
                            line_gt[ent[0]]=None
                    else:
                        line_gt[ent]=eval(ent)
            point_gt={}
            if 'point' in self.only_types:
                for ent in self.only_types['point']:
                    if type(ent)==list:
                        toComb=[]
                        for inst in ent[1:]:
                            einst = eval(inst)
                            if einst is not None:
                                toComb.append(einst)
                        if len(toComb)>0:
                            comb = torch.cat(toComb,dim=1)
                            point_gt[ent[0]]=comb
                        else:
                            line_gt[ent[0]]=None
                    else:
                        point_gt[ent]=eval(ent)
            pixel_gtR=None
            #for ent in self.only_types['pixel']:
            #    if type(ent)==list:
            #        comb = ent[1]
            #        for inst in ent[2:]:
            #            comb = (comb + inst)==2 #:eq(2) #pixel-wise AND
            #        pixel_gt[ent[0]]=comb
            #    else:
            #        pixel_gt[ent]=eval(ent)
            if 'pixel' in self.only_types and self.only_types['pixel'][0]=='table_pixels':
                pixel_gtR=pixel_gt

            return {
                "img": img,
                "bb_gt": bbs,
                "line_gt": line_gt,
                "point_gt": point_gt,
                "pixel_gt": pixel_gtR,
                "imgName": imageName,
                "scale": s,
                "cropPoint": cropPoint
                }



            #def getBBGT(self,bbs,s):

            #    useBBs=bbs
            #    #for bb in bbs:
            #    #    if fields and self.isSkipField(bb):
            #    #        continue
            #    #    else:
            #    #        useBBs.append(bb)
            #    bbs = np.empty((1,len(useBBs), 8+8+2), dtype=np.float32) #2x4 corners, 2x4 cross-points, 2 classes
            #    j=0
            #    for bb in useBBs:
            #        tlX = bb['poly_points'][0][0]
            #        tlY = bb['poly_points'][0][1]
            #        trX = bb['poly_points'][1][0]
            #        trY = bb['poly_points'][1][1]
            #        brX = bb['poly_points'][2][0]
            #        brY = bb['poly_points'][2][1]
            #        blX = bb['poly_points'][3][0]
            #        blY = bb['poly_points'][3][1]

            #        bbs[:,j,0]=tlX*s
            #        bbs[:,j,1]=tlY*s
            #        bbs[:,j,2]=trX*s
            #        bbs[:,j,3]=trY*s
            #        bbs[:,j,4]=brX*s
            #        bbs[:,j,5]=brY*s
            #        bbs[:,j,6]=blX*s
            #        bbs[:,j,7]=blY*s
            #        #we add these for conveince to crop BBs within window
            #        bbs[:,j,8]=s*(tlX+blX)/2
            #        bbs[:,j,9]=s*(tlY+blY)/2
            #        bbs[:,j,10]=s*(trX+brX)/2
            #        bbs[:,j,11]=s*(trY+brY)/2
            #        bbs[:,j,12]=s*(tlX+trX)/2
            #        bbs[:,j,13]=s*(tlY+trY)/2
            #        bbs[:,j,14]=s*(brX+blX)/2
            #        bbs[:,j,15]=s*(brY+blY)/2
            #        bbs[:,j,16]=1 if not fields else 0
            #        bbs[:,j,17]=1 if fields else 0
            #        j+=1
            #    return bbs


    def getTables(self,bbs,s, sH, sW,selfPairs):
        #find which rows and cols are part of the same tables
        groups={}
        #groupsT={}
        groupId=0
        lookup=defaultdict(lambda: None)
        #print('num bbs {}'.format(len(bbs)))
        intersect=defaultdict(list)
        for bb in bbs:
            if bb['type'] == 'fieldCol': # and bb['type'] != 'fieldRow':
                myGroup = lookup[bb['id']]
            

                for bb2 in bbs:
                    if bb2['type'] == 'fieldRow':
                        if polyIntersect(bb['poly_points'], bb2['poly_points']):
                            intersect[bb['id']].append(bb2['id'])
                            
                            otherGroup = lookup[bb2['id']]
                            #if otherGroup is None:
                            #    print('intersection! {}:{},  {}:{}'.format(bb['id'],bb['poly_points'],bb2['id'],bb2['poly_points']))
                            if otherGroup is not None:
                                if myGroup is None:
                                    groups[otherGroup].append(bb)
                                    #groupsT[otherGroup].append(bb['id'])
                                    lookup[bb['id']]=otherGroup
                                    myGroup=otherGroup
                                elif myGroup!=otherGroup:
                                    #merge groups
                                    for ele in groups[otherGroup]:
                                        lookup[ele['id']]=myGroup
                                    groups[myGroup] +=  groups[otherGroup]
                                    #groupsT[myGroup] +=  groupsT[otherGroup]
                                    del groups[otherGroup]
                                    #del groupsT[otherGroup]
                            elif myGroup is None:
                                myGroup = str(groupId)
                                groupId+=1
                                groups[myGroup] = [bb,bb2]
                                #groupsT[myGroup] = [bb['id'],bb2['id']]
                                lookup[bb['id']] = myGroup
                                lookup[bb2['id']] = myGroup
                            else:
                                groups[myGroup].append(bb2)
                                #groupsT[myGroup].append(bb2['id'])
                                lookup[bb2['id']] = myGroup
                        #print(bb['id']+'  '+bb2['id'])
                        #print('{}  {}'.format(myGroup,otherGroup))
                        #print(groupsT)
                        #input("Press Enter to continue...")
        #now we check if a group needs split into two tables (horizontally)
        final_groups=[]
        for gid, group in groups.items():
            rowsOfCol=defaultdict(list)
            colOfX={}
            for bbCol in group:
                if bbCol['type'] == 'fieldCol':
                    colOfX[bbCol['poly_points'][0][0]] = bbCol['id']
                    for bbRow in group:
                        if bbRow['type'] == 'fieldRow' and bbRow['id'] in intersect[bbCol['id']]:
                            rowsOfCol[bbCol['id']].append(bbRow['id'])



            if len(rowsOfCol)>4:
                xs=list(colOfX.keys())
                xs.sort()

                numColAnchors = math.ceil(len(rowsOfCol)*0.25)
                leftAnchors=[colOfX[x] for x in xs[0:numColAnchors]]
                rightAnchors=[colOfX[x] for x in xs[-numColAnchors:]]
                leftRows = []
                for col in leftAnchors:
                    leftRows+=rowsOfCol[col]
                rightRows = []
                for col in rightAnchors:
                    rightRows+=rowsOfCol[col]
                leftRows=set(leftRows)
                rightRows=set(rightRows)

                def addPaired(s,pairs):
                    for pair in pairs:
                        if pair[0] in s:
                            s.add(pair[1])
                        elif pair[1] in s:
                            s.add(pair[0])
                addPaired(leftRows,selfPairs)
                addPaired(rightRows,selfPairs)
                
                intersected = leftRows.intersection(rightRows)
                if len(intersected)==0:
                    #split
                    loseRowCounts=defaultdict(lambda:0)
                    leftGroup=[]#list(leftRows)
                    rightGroup=[]#list(rightRows)
                    for bb in group:
                        if bb['type'] == 'fieldCol':
                            count=0
                            addLose=[]
                            for rowId in rowsOfCol[bb['id']]:
                                if rowId in leftRows:
                                    count-=1
                                elif rowId in rightRows:
                                    count+=1
                                else:
                                    addLose.append(rowId)
                            if count<0:
                                leftGroup.append(bb)
                                for id in addLose:
                                    loseRowCounts[id]-=1
                            elif count>0:
                                rightGroup.append(bb)
                                for id in addLose:
                                    loseRowCounts[id]+=1
                            else:
                                #error, this col is ambigous
                                raise Exception("ambig col",bb)
                        if bb['type'] == 'fieldRow':
                            if bb['id'] in leftRows:
                                leftGroup.append(bb)
                            elif bb['id'] in rightRows:
                                rightGroup.append(bb)
                            else:
                                loseRowCounts[bb['id']]+=0
                    for bb in group:
                        if bb['id'] in loseRowCounts:
                            if loseRowCounts[bb['id']]<0:
                                leftGroup.append(bb)
                            elif loseRowCounts[bb['id']]>0:
                                rightGroup.append(bb)
                            else:
                                raise Exception("ambig row",bb)
                    final_groups+=[leftGroup,rightGroup]
                else:
                    final_groups.append(group)
            else:
                final_groups.append(group)
                #elif len(intersect)<len(leftRows):
                #    #error, we have a row spannign left to right, but some dont

        #print(groups)
        #parse table bbs for intersection points
        intersectionPoints = []
        pixelMap = np.zeros((sH,sW,1),dtype=np.float32)
        #print('num tables {}'.format(len(groups)))
        for tableBBs in final_groups:
            #print('tableBBs {}'.format(len(tableBBs)))
            table = {}
            for bb in tableBBs:
                bb['poly_points'] = np.array( bb['poly_points'])
                table[bb['id']]=bb
            rows=[]
            cols=[]

            #I assume that rows and columns may be represented by multiple BBs
            combIds={}
            idToComb={}
            curComb=0
            #first we figure out what these are (they are paired)
            for pair in selfPairs:
                if pair[0] in table and pair[1] in table and table[pair[0]]['type']==table[pair[1]]['type']:
                    if pair[0] in idToComb:
                        pair0Comb=idToComb[pair[0]]
                        if pair[1] in idToComb:
                            pair1Comb=idToComb[pair[1]]
                            if pair0Comb!=pair1Comb:
                                #merge
                                #print('merge {}:{} and {}:{}'.format(pair0Comb,combIds[pair0Comb],pair1Comb,combIds[pair0Comb]))
                                combIds[pair0Comb] += combIds[pair1Comb]
                                for id in combIds[pair1Comb]:
                                    idToComb[id]=pair0Comb
                                del combIds[pair1Comb]
                        else:
                            combIds[pair0Comb].append(pair[1])
                            idToComb[pair[1]]=pair0Comb
                    elif pair[1] in idToComb:
                        pair1Comb=idToComb[pair[1]]
                        combIds[pair1Comb].append(pair[0])
                        idToComb[pair[0]]=pair1Comb
                    else:
                        combIds[curComb]=[pair[0],pair[1]]
                        idToComb[pair[0]]=curComb
                        idToComb[pair[1]]=curComb
                        curComb+=1
                    #print(combIds)
                    #print(idToComb)
            #sort them in order (left->right or top->botton) and add them to our lists
            for _,ids in combIds.items():
                    typ = table[ids[0]]['type']
                    toApp=[]
                    for id in ids:
                        toApp.append(table[id]['poly_points'])
                        cv2.fillConvexPoly(pixelMap[:,:,0],(table[id]['poly_points']*s).astype(int),1)
                        del table[id]

                    if typ=='fieldRow':
                        toApp.sort(key=lambda a: a[0,0])#sort horz by top-left point
                        rows.append(toApp)
                    else:
                        toApp.sort(key=lambda a: a[0,1])#sort vert by top-left point
                        cols.append(toApp)

            #add the single BB rows and columns
            for id,bb in table.items():
                #npBB = np.array(bb['poly_points'])
                if bb['type']=='fieldRow':
                    rows.append([bb['poly_points']])
                else:
                    cols.append([bb['poly_points']])
                #print(npBB*s)
                cv2.fillConvexPoly(pixelMap[:,:,0],(bb['poly_points']*s).astype(int),1)

            rows.sort(key=lambda a: a[0][0,1])#sort vertically by top-left point
            cols.sort(key=lambda a: a[0][0,0])#sort horizontally by top-left point
            #print (len(rows))
            #print (len(cols))
            #for each row seperator line (top and bottom lines of BBs) find intersecting column sep lines
            #we must iterate over all the components of each row
            #the very top boundary (and bottom) must be handeled specially since they don't have two lines
            nextInd=0
            j=0
            for lineComponent in rows[0]:
                ###rint('row start, comp:{}'.format(j))
                height = getHeightFromBB(lineComponent)
                if j==0:
                    distFromPrev=getWidthFromBB(lineComponent)/2#float('inf')
                else:
                    distFromPrev=np.linalg.norm(lineComponent[0]-rows[0][j-1][1])
                if j==len(rows[0])-1:
                    distToNext=getWidthFromBB(lineComponent)/2#float('inf')
                else:
                    distToNext=np.linalg.norm(lineComponent[1]-rows[0][j+1][0])
                somePoints,nextInd,before = getIntersectsCols(lineComponent[0:2],
                        cols,
                        nextInd,
                        threshLine_low=height/2,#float('inf'),
                        threshLine_high=height/2,
                        threshLeft=distFromPrev,
                        threshRight=distToNext)
                if before:
                    intersectionPoints[-1] = (intersectionPoints[-1]+somePoints[0]*s)/2
                    for p in somePoints[1:]:
                        intersectionPoints.append(s*p)
                else:
                    for p in somePoints:
                        intersectionPoints.append(s*p)
                j+=1
            for i in range(len(rows)-1):
                avgHHeight_ip1=0
                for lineComponent in rows[i+1]:
                    height = getHeightFromBB(lineComponent)
                    avgHHeight_ip1+=height/2
                avgHHeight_ip1/=len(rows[i+1])
                nextInd=0
                pointsU=[] #points from the bottom line of the BB above the seperator
                avgHHeight_i=0
                j=0
                #get the upper points (bottom of above BB)
                for lineComponent in rows[i]:
                    ###rint('row U {}, comp:{}'.format(i,j))
                    height = getHeightFromBB(lineComponent)
                    avgHHeight_i+=height/2
                    if j==0:
                        distFromPrev=getWidthFromBB(lineComponent)/2#float('inf')
                    else:
                        distFromPrev=np.linalg.norm(lineComponent[3]-rows[i][j-1][2])
                    if j==len(rows[i])-1:
                        distToNext=getWidthFromBB(lineComponent)/2#float('inf')
                    else:
                        distToNext=np.linalg.norm(lineComponent[2]-rows[i][j+1][3])
                    somePoints,nextInd,before = getIntersectsCols(lineComponent[2:4],
                            cols,
                            nextInd,
                            threshLine_low=height/2,
                            threshLine_high=avgHHeight_ip1,
                            threshLeft=distFromPrev,
                            threshRight=distToNext)
                    if before:
                        if len(pointsU)>0:
                            pointsU = pointsU[:-1] + [(pointsU[-1]+somePoints[0])/2] + somePoints[1:]
                        else:
                            pointsU = somePoints[1:]
                    else:
                        pointsU+=somePoints
                    j+=1
                avgHHeight_i/=len(rows[i])
                pointsL=[] #points from the top line of the BB below the seperator
                nextInd=0
                j=0
                #get lower points (top of below BB)
                for lineComponent in rows[i+1]:
                    ###rint('row L {}, comp:{}'.format(i,j))
                    height = getHeightFromBB(lineComponent)
                    if j==0:
                        distFromPrev=getWidthFromBB(lineComponent)/2#float('inf')
                    else:
                        distFromPrev=np.linalg.norm(lineComponent[0]-rows[i+1][j-1][1])
                    if j==len(rows[i+1])-1:
                        distToNext=getWidthFromBB(lineComponent)/2#float('inf')
                    else:
                        distToNext=np.linalg.norm(lineComponent[1]-rows[i+1][j+1][0])
                    somePoints,nextInd,before = getIntersectsCols(lineComponent[0:2],
                            cols,
                            nextInd,
                            threshLine_low=avgHHeight_i,
                            threshLine_high=height/2,
                            threshLeft=distFromPrev,
                            threshRight=distToNext)
                    if before:
                        if len(pointsL)>0:
                            pointsL = pointsL[:-1] + [(pointsL[-1]+somePoints[0])/2] + somePoints[1:]
                        else:
                            pointsL = somePoints[1:]
                    else:
                        pointsL+=somePoints
                    j+=1
                #print(i)
                #print(pointsU)
                #print(pointsL)
                if len(pointsU) != len(pointsL):
                    raise Exception(i,pointsU,pointsL)
                #average the upper and lower points (and scale them)
                for pi in range(len(pointsU)):
                    intersectionPoints.append(s*(pointsU[pi]+pointsL[pi])/2.0)
                    
            #special handeling of bottom boundary
            nextInd=0
            j=0
            for lineComponent in rows[-1]:
                ###rint('row end, comp:{}'.format(j))
                if j==0:
                    distFromPrev=getWidthFromBB(lineComponent)/2#float('inf')
                else:
                    distFromPrev=np.linalg.norm(lineComponent[3]-rows[-1][j-1][2])
                if j==len(rows[-1])-1:
                    distToNext=getWidthFromBB(lineComponent)/2#float('inf')
                else:
                    distToNext=np.linalg.norm(lineComponent[2]-rows[-1][j+1][3])
                somePoints,nextInd,before = getIntersectsCols(lineComponent[2:4],
                        cols,
                        nextInd,
                        threshLine_low=height/2,
                        threshLine_high=height/2,#float('inf'),
                        threshLeft=distFromPrev,
                        threshRight=distToNext)
                j+=1
                if before:
                    intersectionPoints[-1] = (intersectionPoints[-1]+somePoints[0]*s)/2
                    for p in somePoints[1:]:
                        intersectionPoints.append(s*p)
                else:
                    for p in somePoints:
                        intersectionPoints.append(s*p)

                    #rowLines=[ [rows[0][0], rows[0][1]] ]
                    #for i in range(len(rows)-1):
                    #    rowLines.append( [(rows[i][3]+rows[i+1][0])/2, (rows[i][2]+rows[i+1][1])/2] )
                    #rowLines.append( [rows[-1][3], rows[-1][2]] )

                    #colLines=[ [cols[0][0], cols[0][3]] ]
                    #for i in range(len(cols)-1):
                    #    colLines.append( [(cols[i][1]+cols[i+1][0])/2, (cols[i][2]+cols[i+1][3])/2] )
                    #colLines.append( [cols[-1][1], cols[-1][2]] )

                    #for rowLine in rowLines:
                    #    for colLine in colLines:
                    #        p = lineIntersection(rowLine,colLine)
                    #        if p is not None:
                    #            #print('{} {} = {}'.format(rowLine,colLine,p))
                    #            intersectionPoints.append((p[0]*s,p[1]*s))

        intersectionPointsM = np.empty((1,len(intersectionPoints), 2), dtype=np.float32)
        j=0
        for x, y in intersectionPoints:
            intersectionPointsM[0,j,0]=x
            intersectionPointsM[0,j,1]=y
            j+=1
            
        return intersectionPointsM, pixelMap


    def __getResponseBBList(self,queryId,annotations):
        responseBBList=[]
        for pair in annotations['pairs']: #done already +annotations['samePairs']:
            if queryId in pair:
                if pair[0]==queryId:
                    otherId=pair[1]
                else:
                    otherId=pair[0]
                if otherId in annotations['byId']: #catch for gt error
                    responseBBList.append(annotations['byId'][otherId])
                #if not self.isSkipField(annotations['byId'][otherId]):
                #    poly = np.array(annotations['byId'][otherId]['poly_points']) #self.__getResponseBB(otherId,annotations)  
                #    responseBBList.append(poly)
        return responseBBList


def getWidthFromBB(bb):
    return (np.linalg.norm(bb[0]-bb[1]) + np.linalg.norm(bb[3]-bb[2]))/2
def getHeightFromBB(bb):
    return (np.linalg.norm(bb[0]-bb[3]) + np.linalg.norm(bb[1]-bb[2]))/2

#This iterates over each part of the columns, seeing if their seperator lines intersect the argument line
##We assume that an intersection must occur (we don't have row components that don't intersect any columns)
def getIntersectsCols(line,cols,startInd,threshLine_low=10,threshLine_high=10,threshLeft=float('inf'),threshRight=float('inf'),failed=0):
    if startInd>0:
        startInd-=1
        tryBefore=True
    else:
        tryBefore=False
    intersectionThresh=20
    intersectionBoth=True
    if failed==1:
        intersectionThresh=40
    elif failed==2:
        intersectionBoth=False
    elif failed>2:
        return [], 0, tryBefore

    #left-most boundary
    p=None
    if startInd==0:
        j=0
        for lineComponent in cols[0]:
            ###rint('first, j:{}, failed:{}'.format(j,failed))
            width = getWidthFromBB(lineComponent)
            p = lineIntersection(line,[lineComponent[0],lineComponent[3]], 
                    threshA_low=threshLeft, #float("inf"), 
                    threshA_high=width/2, 
                    threshB_low=threshLine_low if j==len(cols[0])-1 or threshLine_low!=float('inf') else 10,
                    threshB_high=threshLine_high if j==len(cols[0])-1 or threshLine_high!=float('inf') else 10,
                    both=intersectionBoth)
            j+=1
            if p is not None:
                break
        if p is None:
            if tryBefore:
                tryBefore=False
                startInd=1
                iPoints=[]
            elif failed==2:
                return [], 0, tryBefore
            else:
                return getIntersectsCols(line,cols,startInd,threshLine_low,threshLine_high,threshLeft,threshRight,failed+1)
        else:
            iPoints=[p]
            startInd=1
            #tryBefore=False
    else:
        iPoints=[]

    done=False
    i = startInd-1 #in case the for-loop doesn't run at all
    for i in range(startInd-1,len(cols)-1):
        #if i==(startInd):
            #tryBefore=False
        avgWidth_ip1=0
        for lineComponent in cols[i+1]:
            width = getWidthFromBB(lineComponent)
            avgWidth_ip1+=width
        avgWidth_ip1/=len(cols[i+1])
        avgHWidth_ip1=avgWidth_ip1/2
        pL=pR=None
        avgWidth_i=0
        j=0
        for lineComponent in cols[i]:
            ###rint('L i:{}, j:{}, failed:{}'.format(i,j,failed))
            width = getWidthFromBB(lineComponent)
            avgWidth_i+=width
            pL = lineIntersection(line,lineComponent[1:3], 
                    threshA_low=width/2, 
                    threshA_high=avgHWidth_ip1, 
                    threshB_low=threshLine_low if j==len(cols[0])-1 or threshLine_low!=float('inf') else 10,
                    threshB_high=threshLine_high if j==len(cols[0])-1 or threshLine_high!=float('inf') else 10,
                    both=intersectionBoth)
            j+=1
            if pL is not None:
                break
        avgWidth_i/=len(cols[i])
        avgHWidth_i=avgWidth_i/2
        j=0
        for lineComponent in cols[i+1]:
            ###rint('R i:{}, j:{}, failed:{}'.format(i,j,failed))
            pR = lineIntersection(line,[lineComponent[0],lineComponent[3]], 
                    threshA_low=avgHWidth_i, 
                    threshA_high=width/2, 
                    threshB_low=threshLine_low if j==len(cols[0])-1 or threshLine_low!=float('inf') else 10,
                    threshB_high=threshLine_high if j==len(cols[0])-1 or threshLine_high!=float('inf') else 10,
                    both=intersectionBoth)
            j+=1
            if pR is not None:
                break
        #print('pL {}'.format(pL))
        #print('pR {}'.format(pR))
        #print('failed {}, i={}, line={}'.format(failed,i,line))
        #assert((pL is None) == (pR is None))
        if (pL is None) and (pR is None):
            if tryBefore and i==startInd-1:
                tryBefore=False
                continue
            else:
                done=True
                break
        elif pL is None:
            iPoints.append(pR)
        elif pR is None:
            iPoints.append(pL)
        else:
            iPoints.append((pL+pR)/2.0)
    if not done:
        #right-most boundary
        j=0
        for lineComponent in cols[-1]:
            ###rint('last, j:{}, failed:{}'.format(j,failed))
            width = getWidthFromBB(lineComponent)
            p = lineIntersection(line,lineComponent[1:3], 
                    threshA_low=width/2, 
                    threshA_high=threshRight, #float('inf'), 
                    threshB_low=threshLine_low if j==len(cols[0])-1 or threshLine_low!=float('inf') else 10,
                    threshB_high=threshLine_high if j==len(cols[0])-1 or threshLine_high!=float('inf') else 10,
                    both=intersectionBoth)
            j+=1
            if p is not None:
                iPoints.append(p)
                i = len(cols)+1
                break
            else:
                i=len(cols)
    else:
        i+=1
    if len(iPoints)>0 or failed==2:
        return iPoints,i,tryBefore
    else:
        return getIntersectsCols(line,cols,startInd,threshLine_low,threshLine_high,threshLeft,threshRight,failed+1)


