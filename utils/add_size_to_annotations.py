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
