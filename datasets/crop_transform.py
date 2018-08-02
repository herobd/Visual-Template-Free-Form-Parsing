from sol import crop_utils
import numpy as np
import cv2

def perform_crop(img, crop):
    cs = crop['crop_size']
    cropped_gt_img = img[crop['dim0'][0]:crop['dim0'][1], crop['dim1'][0]:crop['dim1'][1]]
    scaled_gt_img = cv2.resize(cropped_gt_img, (cs, cs), interpolation = cv2.INTER_CUBIC)
    return scaled_gt_img


def generate_random_crop(img, gt, params):

    contains_label = np.random.random() < params['prob_label']
    cs = params['crop_size']

    cnt = 0
    while True:

        dim0 = np.random.randint(0,img.shape[0]-cs)
        dim1 = np.random.randint(0,img.shape[1]-cs)

        crop = {
            "dim0": [dim0, dim0+cs],
            "dim1": [dim1, dim1+cs],
            "crop_size": cs
        }

        #TODO: this only works for the center points
        gt_match = np.zeros_like(gt[...,0:2])
        gt_match[...,0][gt[...,0] < dim1] = 1
        gt_match[...,0][gt[...,0] > dim1+cs] = 1

        gt_match[...,1][gt[...,1] < dim0] = 1
        gt_match[...,1][gt[...,1] > dim0+cs] = 1

        gt_match = 1-gt_match
        gt_match = np.logical_and(gt_match[...,0], gt_match[...,1])

        if gt_match.sum() > 0 and contains_label or cnt > 100:
            cropped_gt_img = perform_crop(img, crop)
            return crop, cropped_gt_img, np.where(gt_match != 0)

        if gt_match.sum() == 0 and not contains_label:
            cropped_gt_img = perform_crop(img, crop)
            return crop, cropped_gt_img, np.where(gt_match != 0)

        cnt += 1

class CropTransform(object):
    def __init__(self, crop_params):
        crop_size = crop_params['crop_size']
        self.random_crop_params = crop_params
        self.pad_params =  ((crop_size,crop_size),(crop_size,crop_size),(0,0))

    def __call__(self, sample):
        org_img = sample['img']
        gt = sample['sol_gt']

        org_img = np.pad(org_img, self.pad_params, 'mean')

        gt[:,:,0] = gt[:,:,0] + self.pad_params[0][0]
        gt[:,:,1] = gt[:,:,1] + self.pad_params[1][0]

        gt[:,:,2] = gt[:,:,2] + self.pad_params[0][0]
        gt[:,:,3] = gt[:,:,3] + self.pad_params[1][0]

        crop_params, org_img, gt_match = crop_utils.generate_random_crop(org_img, gt, self.random_crop_params)

        gt = gt[gt_match][None,...]
        gt[...,0] = gt[...,0] - crop_params['dim1'][0]
        gt[...,1] = gt[...,1] - crop_params['dim0'][0]

        gt[...,2] = gt[...,2] - crop_params['dim1'][0]
        gt[...,3] = gt[...,3] - crop_params['dim0'][0]

        return {
            "img": org_img,
            "sol_gt": gt
        }
