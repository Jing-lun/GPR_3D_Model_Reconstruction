# Copyright 2021, Robotics Lab, City College of New York

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  Originating Author: Jinglun Feng, (jfeng1@ccny.cuny.edu)

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from open3d import *

def array2samples_distance(array1, array2):
    """
    arguments:
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
    """
    # print(type(array1),type(array2))
    num_point1, num_features1 = array1.shape
    num_point2, num_features2 = array2.shape
    expanded_array1 = array1.repeat(num_point2, 1)
    expanded_array2 = torch.reshape(
            torch.unsqueeze(array2, 1).repeat(1, num_point1, 1),
            (-1, num_features2))
    distances = (expanded_array1-expanded_array2)* (expanded_array1-expanded_array2)
#    distances = torch.sqrt(distances)
    distances = torch.sum(distances,dim=1)
    distances = torch.reshape(distances, (num_point2, num_point1))
    distances = torch.min(distances,dim=1)[0]
    distances = torch.mean(distances)
    return distances

def chamfer_distance_numpy(array1, array2):
    batch_size, num_point, num_features = array1.shape
    dist = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist = dist + (0.5*av_dist1+0.5*av_dist2)/batch_size
    return dist*100

def chamfer_distance_numpy_test(array1, array2):
    batch_size, num_point, num_features = array1.shape
    dist_all = 0
    dist1 = 0
    dist2 = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist_all = dist_all + (av_dist1+av_dist2)/batch_size
        dist1 = dist1+av_dist1/batch_size
        dist2 = dist2+av_dist2/batch_size
    return dist_all, dist1, dist2

class PointLoss(nn.Module):
    def __init__(self):
        super(PointLoss,self).__init__()

    def forward(self,array1,array2):
        return chamfer_distance_numpy(array1,array2)

class PointLoss_test(nn.Module):
    def __init__(self):
        super(PointLoss_test,self).__init__()

    def forward(self,array1,array2):
        return chamfer_distance_numpy_test(array1,array2)
if __name__=='__main__':
    a = torch.randn(1,256,3)
    b = torch.randn(1,256,3)
    c = torch.randn(1,64,3)
    d = torch.randn(1,64,3)
    p = PointLoss()
    print(p(a,b))
    print(p(c,d)*0.25)
