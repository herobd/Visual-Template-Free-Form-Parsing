"""
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
import cv2
import numpy as np

def tensmeyer_brightness(img, foreground=0, background=0):
    if img.shape[2]==3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    ret,th = cv2.threshold(gray ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    th = (th.astype(np.float32) / 255)[...,None]

    img = img.astype(np.float32)
    img = img + (1.0 - th) * foreground
    img = img + th * background

    img[img>255] = 255
    img[img<0] = 0

    return img.astype(np.uint8)

def apply_tensmeyer_brightness(img, sigma=20, **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    foreground = random_state.normal(0,sigma)
    background = random_state.normal(0,sigma)
    #print('fore {}, back {}'.format(foreground,background))

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
