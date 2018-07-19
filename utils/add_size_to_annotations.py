import sys
import json
import os
from skimage import io


dirPath = sys.argv[1]

with open(os.path.join(dirPath,'categories.json')) as f:
    imageToCategories = json.loads(f.read())

for imageName in imageToCategories:
    image = io.imread(os.path.join(dirPath,'images',imageName))
    with open(os.path.join(dirPath,'annotations',imageName+'.json')) as f:
        annotations = json.loads(f.read())
    annotations['imageConsts']['height']=image.shape[0]
    annotations['imageConsts']['width']=image.shape[1]
    with open(os.path.join(dirPath,'annotationsMod',imageName+'.json'),'w') as f:
        f.write(json.dumps(annotations, sort_keys=True, indent=4, separators=(',', ': ')))
