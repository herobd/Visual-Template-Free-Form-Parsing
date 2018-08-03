import numpy as np
import cv2

def perform_crop(img, crop):
    cs = crop['crop_size']
    cropped_gt_img = img[crop['dim0'][0]:crop['dim0'][1], crop['dim1'][0]:crop['dim1'][1]]
    scaled_gt_img = cv2.resize(cropped_gt_img, (cs, cs), interpolation = cv2.INTER_CUBIC)
    return scaled_gt_img


def generate_random_crop(img, gts, params):
    
    contains_label = np.random.random() < params['prob_label'] if 'prob_label' in params else None
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

        gt_match={}
        hit=False
        for name, gt in gts.items():
            gt_match[name] = np.zeros_like(gt)
            gt_match[name][...,0][gt[...,0] < dim1] = 1
            gt_match[name][...,0][gt[...,0] > dim1+cs] = 1

            gt_match[name][...,1][gt[...,1] < dim0] = 1
            gt_match[name][...,1][gt[...,1] > dim0+cs] = 1

            gt_match[name][...,0][gt[...,2] < dim1] = 1
            gt_match[name][...,0][gt[...,2] > dim1+cs] = 1

            gt_match[name][...,1][gt[...,3] < dim0] = 1
            gt_match[name][...,1][gt[...,3] > dim0+cs] = 1

            gt_match[name] = 1-gt_match[name]
            gt_match[name] = np.logical_and(gt_match[name][...,0], gt_match[name][...,1], gt_match[name][...,2], gt_match[name][...,3])
            if gt_match[name].sum() > 0:
                hit=True
        
        if contains_label is not None:
            if hit and contains_label or cnt > 100:
                cropped_gt_img = perform_crop(img, crop)
                return crop, cropped_gt_img, np.where(gt_match != 0)

            if not hit and not contains_label:
                cropped_gt_img = perform_crop(img, crop)
                return crop, cropped_gt_img, np.where(gt_match != 0)
        else:
            cropped_gt_img = perform_crop(img, crop)
            return crop, cropped_gt_img, np.where(gt_match != 0)

        cnt += 1

class CropTransform(object):
    def __init__(self, crop_params):
        crop_size = crop_params['crop_size']
        self.random_crop_params = crop_params
        self.pad_params = ((crop_size,crop_size),(crop_size,crop_size),(0,0))

    def __call__(self, sample):
        org_img = sample['img']
        gts = sample['sol_eol_gt']

        #pad out to allow random samples to take space off of the page
        org_img = np.pad(org_img, self.pad_params, 'mean')
        
        #pad the points accordingly
        for name, gt in gts.items():
            gt[:,:,0] = gt[:,:,0] + self.pad_params[0][0]
            gt[:,:,1] = gt[:,:,1] + self.pad_params[1][0]

            gt[:,:,2] = gt[:,:,2] + self.pad_params[0][0]
            gt[:,:,3] = gt[:,:,3] + self.pad_params[1][0]

        crop_params, org_img, gt_match = generate_random_crop(org_img, gt, self.random_crop_params)

        for name, gt in gts.items():
            gt = gt[gt_match][None,...]
            gt[...,0] = gt[...,0] - crop_params['dim1'][0]
            gt[...,1] = gt[...,1] - crop_params['dim0'][0]

            gt[...,2] = gt[...,2] - crop_params['dim1'][0]
            gt[...,3] = gt[...,3] - crop_params['dim0'][0]

        return {
            "img": org_img,
            "sol_eol_gt": gts
        }
