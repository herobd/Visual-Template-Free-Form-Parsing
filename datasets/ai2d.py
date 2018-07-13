import torch
import json
from skimage import io

class AI2D(torch.utils.data.Dataset):
    """
    Class for reading AI2D dataset and creating query/result masks from bounding polygons
    """
    def __init__(self, dirPath=None, split=None, config=None, images=None, queryPolys=None, responsePolyLists=None):
        if 'augmentation_params' in config['data_loader']:
            self.augmentation_params=config['data_loader']['augmentation_params']
        else:
            self.augmentation_params=None
        if images is not None:
            self.imagePaths=images
            self.queryPolys=queryPolys
            self.responsePolyLists=responsePolyLists
        else:
            with open(os.path.join(dirPath,'categories.json') as f:
                imageToCategories = json.loads(f.read())
            with open(os.path.join(dirPath,'traintestplit_categories.json') as f:
                #if split=='valid' or split=='validation':
                #    trainTest='train'
                #else:
                #    trainTest=split
                categoriesToUse = json.loads(f.read())[split]
            self.imagePaths=[]
            self.queryPolys=[]
            self.responsePolyLists=[]
            for image, category in imageToCategories.items():
                if category in categoriesToUse:
                    with open(os.path.join(dirPath,'annotations',image+'.json')) as f:
                        annotations = json.loads(f.read())
                    for blobId, blob in annotations['blob'].items():
                        self.imagePaths.append(os.path.join(dirPath,'images',image))
                        self.queryPolys.append(blob['polygon'])
                        self.responsePolyLists.append(getResponsePolyList(blobId,annotations))

                                    
                    annotationPaths.append(os.path.join(dirPath,'annotations',image+'.json'))


    def getResponsePolyList(queryId,annotaions):
        responsePolyList=[]
        for relId in annotations['relationships']:
            if queryId in relId:
                pos = relId.find(queryId)
                #only the objects listed immediatley before or after this one are important
                if pos>0 and relId[pos-1]=='+':
                    nextPlus = relId.rfind('+',0,pos)
                    neighborId = relId[nextPlus+1:pos-1]
                    responsePolyList.append(getResponsePoly(neighborId,annotations))
                if pos+len(queryId)+1<len(relId) and relId[pos+len(queryId)+1]=='+':
                    nextPlus = relId.find('+',pos+len(queryId)+2)
                    if nextPlus==-1:
                        neighborId=relId[pos+len(queryId)+1:]
                    else:
                        neighborId=relId[pos+len(queryId)+1:nextPlus]
                    responsePolyList.append(getResponsePoly(neighborId,annotations))
        return responsePolyList

    def getResponsePoly(neighborId,annotations):
        if neighborId[0]=='T':
            rect=annotations['text'][neighborId]['rectangle']
            poly = [ rect[0], [rect[1][0],rect[0][1]], rect[1], [rect[0][1],rect[1][0]] ]
            return poly
        elif neighborId[0]=='A':
            return annotations['arrows'][neighborId]['polygon']
        elif neighborId[0]=='B':
            return annotations['blobs'][neighborId]['polygon']

    def __len__(self):
        return len(imagePaths)

    def __getitem__(self,index):
        image = io.imread(self.imagePaths[index])
        queryMask = np.zeros([image.shape[0],image.shape[1]])
        rr, cc = polygon(self.queryPolys[index][:, 0], self.queryPolys[index][:, 1], queryMask.shape)
        queryMask[rr,cc]=1
        responseMask = np.zeros([image.shape[0],image.shape[1]])
        for poly in self.responsePolyLists[index]:
            rr, cc = polygon(poly[:, 0], poly[:, 1], responseMask.shape)
            responseMask[rr,cc]=1

        imageWithQuery = np.stack(np.dsplit(image,3)+queryMask,-1)

        sample = (imageWithQuery, responseMask)
        if self.augmentation_params is not None:
            sample = self.augment(sample)
        return sample

    def splitValidation(self, config):
        validation_split = config['validation']['validation_split']
        split = int(len(self) * validation_split)
        perm = np.random.permutation(len(self))
        images = [self.images[x] for x in perm]
        queryPolys = [self.queryPolys[x] for x in perm]
        responsePolyLists = [self.responsePolyLists[x] for x in perm]

        self.images=images[split:]
        self.queryPolys=queryPolys[split:]
        self.responsePolyLists=responsePolyLists[split:]

        return AI2D(config=config, images=images[:split], queryPolys=queryPolys[:split], responsePolyLists=responsePolyLists[:split])
