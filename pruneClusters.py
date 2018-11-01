import json
import numpy as np
import math
import sys
import cv2

if len(sys.argv)<2:
    print('usage: '+sys.argv[0]+' in.json k out.json')
    exit()

def makePointsAndRects(h,w):
    return np.array([-w/2.0,0,w/2.0,0,0,-h/2.0,0,h/2.0, 0,0, 0, h,w])

with open(sys.argv[1]) as file:
    anchors = json.loads(file.read())
goalK = int(sys.argv[2])
outPath = sys.argv[3]

#remove very unpopular anchors
toRemove=[]
for i in range(len(anchors)):
    if anchors[i]['popularity']<5:
        toRemove.append(i)
toRemove.sort(reverse=True)
#print(toRemove)
for idx in toRemove:
    del anchors[idx]

points = np.zeros([len(anchors),13])
for i in range(len(anchors)):
    points[i]=makePointsAndRects(anchors[i]['height'],anchors[i]['width'])
expanded_points1_points = points[:,None,0:8]
expanded_points1_heights = points[:,None,11]
expanded_points1_widths = points[:,None,12]

expanded_points2_points = points[None,:,0:8]
expanded_points2_heights = points[None,:,11]
expanded_points2_widths = points[None,:,12]

#expanded_all_points = expanded_all_points.expand(all_points.shape[0], all_points.shape[1], means_points.shape[1], all_points.shape[2])
expanded_points1_points = np.tile(expanded_points1_points,(1,points.shape[0],1))
expanded_points1_heights = np.tile(expanded_points1_heights,(1,points.shape[0]))
expanded_points1_widths = np.tile(expanded_points1_widths,(1,points.shape[0]))
#expanded_means_points = expanded_means_points.expand(means_points.shape[0], all_points.shape[0], means_points.shape[0], means_points.shape[2])
expanded_points2_points = np.tile(expanded_points2_points,(points.shape[0],1,1))
expanded_points2_heights = np.tile(expanded_points2_heights,(points.shape[0],1))
expanded_points2_widths = np.tile(expanded_points2_widths,(points.shape[0],1))

point_deltas = (expanded_points1_points - expanded_points2_points)
#avg_heights = ((expanded_means_heights+expanded_all_heights)/2)
#avg_widths = ((expanded_means_widths+expanded_all_widths)/2)
avg_heights=avg_widths = (expanded_points1_heights+expanded_points1_widths)/2
#print point_deltas

normed_difference = (
    np.linalg.norm(point_deltas[:,:,0:2],2,2)/avg_widths +
    np.linalg.norm(point_deltas[:,:,2:4],2,2)/avg_widths +
    np.linalg.norm(point_deltas[:,:,4:6],2,2)/avg_heights +
    np.linalg.norm(point_deltas[:,:,6:8],2,2)/avg_heights
    )**2
np.fill_diagonal(normed_difference,float('inf'))
toRemove=[]
for i in range(len(anchors)-goalK):
    cord = np.argmin(normed_difference)
    a=cord//len(anchors)
    b=cord%len(anchors)
    #print('{} {} {}'.format(cord,a,b))

    normed_difference[a,b] = float('inf')
    normed_difference[b,a] = float('inf')
    #toRemove.append(a)
    #normed_difference[a,:] = float('inf')
    #normed_difference[:,a] = float('inf')

    if anchors[a]['popularity'] > anchors[b]['popularity']:
        toRemove.append(b)
        normed_difference[b,:] = float('inf')
        normed_difference[:,b] = float('inf')
    else:
        toRemove.append(a)
        normed_difference[a,:] = float('inf')
        normed_difference[:,a] = float('inf')

toRemove.sort(reverse=True)
#print(toRemove)
for idx in toRemove:
    del anchors[idx]

with open(outPath,'w') as out:
    out.write(json.dumps(anchors))

drawH=1000
drawW=4000
draw = np.zeros([drawH,drawW,3],dtype=np.float)
for anchor in anchors:
    color = np.random.uniform(0.2,1,3).tolist()
    h=anchor['height']
    w=anchor['width']
    rot=anchor['rot']
    tr = ( int(math.cos(rot)*w-math.sin(rot)*h)+(drawW//2),   int(math.sin(rot)*w+math.cos(rot)*h)+(drawH//2) )
    tl = ( int(math.cos(rot)*-w-math.sin(rot)*h)+(drawW//2),  int(math.sin(rot)*-w+math.cos(rot)*h)+(drawH//2) )
    br = ( int(math.cos(rot)*w-math.sin(rot)*-h)+(drawW//2),  int(math.sin(rot)*w+math.cos(rot)*-h)+(drawH//2) )
    bl = ( int(math.cos(rot)*-w-math.sin(rot)*-h)+(drawW//2), int(math.sin(rot)*-w+math.cos(rot)*-h)+(drawH//2) )

    cv2.line(draw,tl,tr,color)
    cv2.line(draw,tr,br,color)
    cv2.line(draw,br,bl,color)
    cv2.line(draw,bl,tl,color,2)
cv2.imshow('pruned',draw)
cv2.waitKey()
cv2.waitKey()
