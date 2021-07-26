import argparse
import math
import torch
import cv2
import numpy as np
from tqdm import tqdm
from skimage import color, io
from model import *

DETECTOR_TRAINED_MODEL = "saved/detector/checkpoint-iteration150000.pth"
PAIRING_TRAINED_MODEL = "saved/pairing/checkpoint-iteration125000.pth"
SCALE_IMAGE_DEFAULT = 1 # percent of original size
INCLUDE_THRESHOLD_DEFAULT = 0.9 # threshold for using the bounding box (0 to 1)


def getCorners(xyrhw):
    xc=xyrhw[0]
    yc=xyrhw[1]
    rot=xyrhw[2]
    h=xyrhw[3]
    w=xyrhw[4]
    h = min(30000,h)
    w = min(30000,w)
    tr = ( int(w*math.cos(rot)-h*math.sin(rot) + xc),  int(w*math.sin(rot)+h*math.cos(rot) + yc) )
    tl = ( int(-w*math.cos(rot)-h*math.sin(rot) + xc), int(-w*math.sin(rot)+h*math.cos(rot) + yc) )
    br = ( int(w*math.cos(rot)+h*math.sin(rot) + xc),  int(w*math.sin(rot)-h*math.cos(rot) + yc) )
    bl = ( int(-w*math.cos(rot)+h*math.sin(rot) + xc), int(-w*math.sin(rot)-h*math.cos(rot) + yc) )
    return tl,tr,br,bl
    
def plotRect(img,color,xyrhw,lineW=1):
    tl,tr,br,bl = getCorners(xyrhw)

    cv2.line(img,tl,tr,color,lineW)
    cv2.line(img,tr,br,color,lineW)
    cv2.line(img,br,bl,color,lineW)
    cv2.line(img,bl,tl,color,lineW)

def detect_boxes(imagePath, scale_image=SCALE_IMAGE_DEFAULT, include_threshold=INCLUDE_THRESHOLD_DEFAULT, output_image=None):
    # fetch the model
    checkpoint = torch.load(DETECTOR_TRAINED_MODEL, map_location=lambda storage, location: storage)
    print(f"Using {checkpoint['arch']}")
    model = eval(checkpoint['arch'])(checkpoint['config']['model'])
    model.load_state_dict(checkpoint['state_dict'])

    # load the image
    print(f"Loading image: {imagePath}")
    np_img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    
    print(f"Transforming image: {imagePath}")
    width = int(np_img.shape[1] * scale_image)
    height = int(np_img.shape[0] * scale_image)
    new_size = (width, height)
    np_img = cv2.resize(np_img,new_size)
    img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    img = img[None,None,:,:]
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    img = 1.0 - img / 128.0

    # run the image through the model
    print(f"Run image through model: {imagePath}")
    result = model(img)

    # produce the output
    boundingboxes = result[0].tolist()
    output = []

    print(f"Process bounding boxes: {imagePath}")
    for i in tqdm(boundingboxes[0]):
        if i[0] < include_threshold:
            continue
        print(i)
        tl,tr,br,bl = getCorners(i[1:])
        scale=1
        bb = {
            'poly_points': [ [float(tl[0]/scale),float(tl[1]/scale)], 
                            [float(tr[0]/scale),float(tr[1]/scale)], 
                            [float(br[0]/scale),float(br[1]/scale)], 
                            [float(bl[0]/scale),float(bl[1]/scale)] ],
            'type':'detectorPrediction',
            'textPred': float(i[7]),
            'fieldPred': float(i[8])
        }
        colour = (255,0,0)  # red
        if bb['textPred'] > bb['fieldPred']:
            colour = (0,0,255)  # blue
        output.append(bb)
        if output_image:
            plotRect(np_img, colour, i[1:6])

    if output_image:
        print(f"Saving output: {output_image}")
        io.imsave(output_image, np_img)
    return output

    
def detect_pairs(imagePath, scale_image=SCALE_IMAGE_DEFAULT, include_threshold=INCLUDE_THRESHOLD_DEFAULT, output_image=None):
    # fetch the model
    checkpoint = torch.load(PAIRING_TRAINED_MODEL, map_location=lambda storage, location: storage)
    print(f"Using {checkpoint['arch']}")
    model = eval(checkpoint['arch'])(checkpoint['config']['model'])
    model.load_state_dict(checkpoint['state_dict'])

    # load the image
    print(f"Loading image: {imagePath}")
    np_img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    
    print(f"Transforming image: {imagePath}")
    width = int(np_img.shape[1] * scale_image)
    height = int(np_img.shape[0] * scale_image)
    new_size = (width, height)
    np_img = cv2.resize(np_img,new_size)
    img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    img = img[None,None,:,:]
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    img = 1.0 - img / 128.0

    # run the image through the model
    print(f"Run image through model: {imagePath}")
    result = model(img)
    # print(result)

    # produce the output
    # boundingboxes = result[0].tolist()
    # output = []

    # print(f"Process bounding boxes: {imagePath}")
    # for i in tqdm(boundingboxes[0]):
    #     if i[0] < include_threshold:
    #         continue
    #     print(i)
    #     tl,tr,br,bl = getCorners(i[1:])
    #     scale=1
    #     bb = {
    #         'poly_points': [ [float(tl[0]/scale),float(tl[1]/scale)], 
    #                         [float(tr[0]/scale),float(tr[1]/scale)], 
    #                         [float(br[0]/scale),float(br[1]/scale)], 
    #                         [float(bl[0]/scale),float(bl[1]/scale)] ],
    #         'type':'detectorPrediction',
    #         'textPred': float(i[7]),
    #         'fieldPred': float(i[8])
    #     }
    #     colour = (255,0,0)  # red
    #     if bb['textPred'] > bb['fieldPred']:
    #         colour = (0,0,255)  # blue
    #     output.append(bb)
    #     if output_image:
    #         plotRect(np_img, colour, i[1:6])

    # if output_image:
    #     print(f"Saving output: {output_image}")
    #     io.imsave(output_image, np_img)
    # return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve bounding boxes for a form image')
    parser.add_argument('image', type=str, help='Path to the image to convert')
    parser.add_argument('output-image', type=str, help="A path to save a version of the original image with form boxes overlaid")
    parser.add_argument('--scale-image', type=float, default=SCALE_IMAGE_DEFAULT,
                        help='Scale the image by this proportion (between 0 and 1)')
    parser.add_argument('--detect-threshold', type=float, default=INCLUDE_THRESHOLD_DEFAULT,
                        help='Include boxes where the confidence is above this threshold (between 0 and 1)')
    args = parser.parse_args()

    result = detect_boxes(
        args.image,
        scale_image=args.scale_image,
        include_threshold=args.detect_threshold,
        output_image=args.output_image,
    )
