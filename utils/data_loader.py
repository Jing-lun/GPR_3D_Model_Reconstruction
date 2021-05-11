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

import os
from os.path import splitext
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
import open3d as o3d
import sys
import h5py
import scipy.io
import logging
import matplotlib.pyplot as plt

gt_path    = '../../new_mat/'
input_path = '../../resnet_range/'

class BasicDataset(Dataset):
    def __init__(self, input_dir, gt_dir):
        self.input_dir = input_dir
        self.gt_dir    = gt_dir

        self.ids = [splitext(file)[0] for file in listdir(input_dir)
                    if not file.startswith('.')]
        # print(self.ids)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        input_file = glob(self.input_dir + idx + '.mat')
        gt_file    = glob(self.gt_dir + idx + '.mat')
        # print(input_file)
        # print(gt_file)
        '''
        h5py method
        '''
        with h5py.File(input_file[0], 'r') as f:
            # dset = f['imageHN']
            dset = f['RawImage']
            mat_input = dset[:]
        mat_input = mat_input.transpose(1,0,2)
        '''
        scipy method
        '''
        f = scipy.io.loadmat(gt_file[0])
        occ_map = f['occupancy_map']
        occ_map = occ_map.transpose(1,0,2)
        shape = occ_map.shape
        occ_map = occ_map.reshape(1, shape[0], shape[1], shape[2])

        return {'mat_input': torch.from_numpy(mat_input), 'mat_gt': torch.from_numpy(occ_map)}

if __name__ == '__main__':
    mat = BasicDataset(input_path, gt_path)
    input = mat[14]
    print(input['mat_input'].shape)
    print(input['mat_gt'].shape)
    gt = np.array(input['mat_gt'])
    print(np.max(gt))
