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
import numpy as np


def my_metric(y_input, y_target):
    assert len(y_input) == len(y_target)
    correct = 0
    for y0, y1 in zip(y_input, y_target):
        if np.array_equal(y0, y1):
            correct += 1
    return correct / len(y_input)


def meanIOU(y_output, y_target):
    assert len(y_output) == len(y_target)
    epsilon = 0.001
    iouSum = 0
    for out, targ in zip(y_output, y_target):
        binary = out>0 #torch.where(out>0,1,0)
        #binary = torch.round(y_input) #threshold at 0.5
        intersection = (binary * targ).sum()
        union = (binary + targ).sum() - intersection
        iouSum += (intersection+epsilon) / (union+epsilon)
    return iouSum / float(len(y_output))

def mean_xy(xyrs_output, xyrs_target):
    assert len(xyrs_output) == len(xyrs_target)
    dists=0
    for out, targ in zip(xyrs_output, xyrs_target):
        dists+=( (out[0:2]-targ[0:2]).linalg.norm() )
    return dists/float(len(xyrs_output))
def std_xy(xyrs_output, xyrs_target):
    assert len(xyrs_output) == len(xyrs_target)
    dists=[]
    for out, targ in zip(xyrs_output, xyrs_target):
        dists.append( (out[0:2]-targ[0:2]).linalg.norm() )
    return np.std(dists)
def mean_rot(xyrs_output, xyrs_target):
    assert len(xyrs_output) == len(xyrs_target)
    rotDiffs=0
    for out, targ in zip(xyrs_output, xyrs_target):
        rotDiffs+=(targ[2]-out[2])
    return rotDiffs/float(len(xyrs_output))
def std_rot(xyrs_output, xyrs_target):
    assert len(xyrs_output) == len(xyrs_target)
    rotDiffs=[]
    for out, targ in zip(xyrs_output, xyrs_target):
        rotDiffs.append(targ[2]-out[2])
    return np.std(rotDiffs)
def mean_scale(xyrs_output, xyrs_target):
    assert len(xyrs_output) == len(xyrs_target)
    scaleDiffs=0
    for out, targ in zip(xyrs_output, xyrs_target):
        scaleDiffs+=(targ[3]-out[3])
    return scaleDiffs/float(len(xyrs_output))
def std_scale(xyrs_output, xyrs_target):
    assert len(xyrs_output) == len(xyrs_target)
    scaleDiffs=[]
    for out, targ in zip(xyrs_output, xyrs_target):
        scaleDiffs.append(targ[3]-out[3])
    return np.std(scaleDiffs)
