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
import os
import struct
import torch
from . import string_utils


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def pt_xyrs_2_xyxy(state):
    out = torch.ones(state.data.shape[0], 5).type(state.data.type())

    x = state[:,:,1:2]
    y = state[:,:,2:3]
    r = state[:,:,3:4]
    s = state[:,:,4:5]

    x0 = -torch.sin(r) * s + x
    y0 = -torch.cos(r) * s + y
    x1 =  torch.sin(r) * s + x
    y1 =  torch.cos(r) * s + y

    return torch.cat([
        state[:,:,0:1],
        x0, y0, x1, y1
    ], 2)
def pt_xyxy_2_xyrs(state):
    out = torch.ones(state.data.shape[0], 5).type(state.data.type())
            
    x0 = state[:,0:1]
    y0 = state[:,1:2]
    x1 = state[:,2:3]
    y1 = state[:,3:4]

    dx = x0-x1
    dy = y0-y1

    d = torch.sqrt(dx**2.0 + dy**2.0)/2.0

    mx = (x0+x1)/2.0
    my = (y0+y1)/2.0

    theta = -torch.atan2(dx, -dy)

    return torch.cat([
        mx, my, theta, d,
        state[:,4:5]
    ], 1)
#-------------------------------------------------------------------------------
# Name:        get_image_size
# Purpose:     extract image dimensions given a file path using just
#              core modules
#
# Author:      Paulo Scardine (based on code from Emmanuel VAÏSSE)
#
# Created:     26/09/2013
# Copyright:   (c) Paulo Scardine 2013
# Licence:     MIT
# From:        https://stackoverflow.com/questions/15800704/get-image-size-without-loading-image-into-memory
#-------------------------------------------------------------------------------
class UnknownImageFormat(Exception):
    pass

def get_image_size(file_path):
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct modules from core
    """
    size = os.path.getsize(file_path)

    with open(file_path) as input:
        height = -1
        width = -1
        data = input.read(25)

        if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
            # GIFs
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith('\211PNG\r\n\032\n')
              and (data[12:16] == 'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith('\211PNG\r\n\032\n'):
            # older PNGs?
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith('\377\330'):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while (b and ord(b) != 0xDA):
                    while (ord(b) != 0xFF): b = input.read(1)
                    while (ord(b) == 0xFF): b = input.read(1)
                    if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(int(struct.unpack(">H", input.read(2))[0])-2)
                    b = input.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat(
                "Sorry, don't know how to get information from this file."
            )

    return width, height

def decode_handwriting(out, idx_to_char):
    hw_out = out#['hw']
    list_of_pred = []
    list_of_raw_pred = []
    for i in range(hw_out.shape[0]):
        logits = hw_out[i,...]
        pred, raw_pred = string_utils.naive_decode(logits)
        pred_str = string_utils.label2str_single(pred, idx_to_char, False)
        raw_pred_str = string_utils.label2str_single(raw_pred, idx_to_char, True)
        list_of_pred.append(pred_str)
        list_of_raw_pred.append(raw_pred_str)

    return list_of_pred, list_of_raw_pred

