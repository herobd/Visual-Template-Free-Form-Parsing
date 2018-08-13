import cv2
import numpy as np

def tensmeyer_brightness(img, foreground=0, background=0):
    ret,th = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    th = (th.astype(np.float32) / 255)[...,None]

    img = img.astype(np.float32)
    img = img + (1.0 - th) * foreground
    img = img + th * background

    img[img>255] = 255
    img[img<0] = 0

    return img.astype(np.uint8)

def apply_tensmeyer_brightness(img, sigma=30, **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    foreground = random_state.normal(0,sigma)
    background = random_state.normal(0,sigma)

    img = tensmeyer_brightness(img, foreground, background)

    return img


def increase_brightness(img, brightness=0, contrast=1):
    img = img.astype(np.float32)
    img = img * contrast + brightness
    img[img>255] = 255
    img[img<0] = 0

    return img.astype(np.uint8)

def apply_random_brightness(img, b_range=[-50,51], **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    brightness = random_state.randint(b_range[0], b_range[1])

    img = increase_brightness(img, brightness)

    return input_data

def apply_random_color_rotation(img, **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    shift = random_state.randint(0,255)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[...,0] = hsv[...,0] + shift
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img
