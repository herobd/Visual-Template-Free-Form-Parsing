import torch
from model.yolo_loss import bbox_iou
import math

def non_max_sup_iou(pred_boxes,thresh_conf=0.5, thresh_inter=0.5):
    #rearr = [0,1,2,5,4,3]
    #for i in range(6,pred_boxes.shape[2]):
    #    rearr.append(i)
    #pred_boxes = pred_boxes[:,:,rearr]
    to_return=[]
    for b in range(pred_boxes.shape[0]):
        
        #allIOU = bbox_iou(
        above_thresh = []
        for i in range(pred_boxes.shape[1]):
            if pred_boxes[b,i,0]>thresh_conf:
                above_thresh.append( (pred_boxes[b,i,0], i) )
        above_thresh.sort(key=lambda a: a[0], reverse=True)
        li = 0
        while li<len(above_thresh)-1:
            i=above_thresh[li][1]
            intersections = max_intersection(pred_boxes[b,i,1:6],pred_boxes[b,[x[1] for x in above_thresh[li+1:]],1:6])
            #ious = bbox_iou(pred_boxes[b,i:i+1,1:5],pred_boxes[b,[x[1] for x in above_thresh[li+1:]],1:5], x1y1x2y2=False)
            to_remove=[]
            for lj in range(len(above_thresh)-1,li,-1):
                j=above_thresh[lj][1]
                #if bbox_iou( pred_boxes[b,i:i+1,1:5], pred_boxes[b,j:j+1,1:5], x1y1x2y2=False) > thresh_iou:
                if intersections[lj-(li+1)] > thresh_inter:
                    to_remove.append(lj)
            #to_remove.reverse()
            for index in to_remove:
                del above_thresh[index]
            li+=1

        best = pred_boxes[b,[x[1] for x in above_thresh],:]
        to_return.append(best)#[:,rearr])
    return to_return


def max_intersection(query_box, candidate_boxes):
    q_x1, q_x2 = query_box[0]-query_box[4], query_box[0]+query_box[4]
    q_y1, q_y2 = query_box[1]-query_box[3], query_box[1]+query_box[3]
    c_x1, c_x2 = candidate_boxes[:,0]-candidate_boxes[:,4], candidate_boxes[:,0]+candidate_boxes[:,4]
    c_y1, c_y2 = candidate_boxes[:,1]-candidate_boxes[:,3], candidate_boxes[:,1]+candidate_boxes[:,3]

    inter_rect_x1 = torch.max(q_x1, c_x1)
    inter_rect_x2 = torch.min(q_x2, c_x2)
    inter_rect_y1 = torch.max(q_y1, c_y1)
    inter_rect_y2 = torch.min(q_y2, c_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0 )

    q_area = (q_x2 - q_x1 + 1) * (q_y2 - q_y1 + 1)
    c_area = (c_x2 - c_x1 + 1) * (c_y2 - c_y1 + 1)
    min_area = torch.min(q_area,c_area)
    #import pdb; pdb.set_trace()

    return inter_area/min_area

def allIOU(boxes1,boxes2):
    b1_x1, b1_x2 = boxes1[:,0]-boxes1[:,4], boxes1[:,0]+boxes1[:,4]
    b1_y1, b1_y2 = boxes1[:,1]-boxes1[:,3], boxes1[:,1]+boxes1[:,3]
    b2_x1, b2_x2 = boxes2[:,0]-boxes2[:,4], boxes2[:,0]+boxes2[:,4]
    b2_y1, b2_y2 = boxes2[:,1]-boxes2[:,3], boxes2[:,1]+boxes2[:,3]

    #expand to make two dimensional, allowing every instance of boxes1
    #to be compared with every intsance of boxes2
    b1_x1 = b1_x1[:,None].expand(boxes1.size(0), boxes2.size(0))
    b1_y1 = b1_y1[:,None].expand(boxes1.size(0), boxes2.size(0))
    b1_x2 = b1_x2[:,None].expand(boxes1.size(0), boxes2.size(0))
    b1_y2 = b1_y2[:,None].expand(boxes1.size(0), boxes2.size(0))
    b2_x1 = b2_x1[None,:].expand(boxes1.size(0), boxes2.size(0))
    b2_y1 = b2_y1[None,:].expand(boxes1.size(0), boxes2.size(0))
    b2_x2 = b2_x2[None,:].expand(boxes1.size(0), boxes2.size(0))
    b2_y2 = b2_y2[None,:].expand(boxes1.size(0), boxes2.size(0))

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0 )

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou
 
#input is tensors of shape [instance,(conf,x,y,rot,h,w)]
def AP_iou(target,pred,iou_thresh,numClasses=2):
    #mAP=0.0
    aps=[]
    precisions=[]
    recalls=[]

    #how many classes are there?
    if len(target.size())>1:
        numClasses=target.size(1)-13
    elif len(pred.size())>1:
        #if there are no targets, we shouldn't be pred anything
        numClasses=pred.size(1)-6
        for cls in range(numClasses):
            if (torch.argmax(pred[:,cls+6:],dim=1)==cls).any():
                aps.append(0) #but we did for this class :(
                precisions.append(0)
            else:
                aps.append(1) #we didn't for this class :)
                precisions.append(1)
            recalls.append(1)
        return aps, precisions, recalls
    else:
        return [1]*numClasses, [1]*numClasses, [1]*numClasses #we didn't for all classes :)

    #by class
    #import pdb; pdb.set_trace()
    for cls in range(numClasses):
        scores=[]
        clsTargInd = target[:,cls+13]==1
        if len(pred.size())>1:
            clsPredInd = torch.argmax(pred[:,6:],dim=1)==cls
        else:
            clsPredInd = torch.empty(0,dtype=torch.uint8)
        if clsTargInd.any() and clsPredInd.any():
            clsTarg = target[clsTargInd]
            clsPred = pred[clsPredInd]
            clsIOUs = allIOU(clsTarg[:,0:],clsPred[:,1:])
            hits = clsIOUs>iou_thresh

            clsIOUs *= hits.float()
            ps = torch.argmax(clsIOUs,dim=1)
            left_ps = torch.ones(clsPred.size(0),dtype=torch.uint8)
            left_ps[ps]=0
            truePos=0
            for t in range(clsTarg.size(0)):
                p=ps[t]
                if hits[t,p]:
                    scores.append( (clsPred[p,0],True) )
                    #hits[t,p]=0
                    truePos+=1
                else:
                    scores.append( (0,True) )
            
            left_conf = clsPred[left_ps,0]
            for i in range(left_conf.size(0)):
                scores.append( (left_conf[i],False) )
            
            rank=[]
            for conf,rel in scores:
                if rel:
                    better=0
                    equal=-1 # as we'll iterate over this instance here
                    for conf2,rel2 in scores:
                        if conf2>conf:
                            better+=1
                        elif conf2==conf:
                            equal+=1
                    rank.append(better+math.floor(equal/2.0))
            rank.sort()
            ap=0.0
            for i in range(len(rank)):
                ap += float(i+1)/(rank[i]+1)
            ap/=len(rank)
            aps.append(ap)

            precisions.append( truePos/clsPred.size(0) )
            recalls.append( truePos/clsTarg.size(0) )

        elif clsPredInd.any() or clsTargInd.any():
            aps.append(0)
            if clsPredInd.any():
                recalls.append(1)
                precisions.append(0)
            else:
                precisions.append(0)
                recalls.append(0)
        else:
            aps.append(1)
            precisions.append(1)
            recalls.append(1)

    return aps, precisions, recalls


