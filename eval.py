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
from tqdm import tqdm
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

def eval_net(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    L1_loss = nn.L1Loss()
    L1_loss.to(device)
    net.eval()
    tot = 0
    with tqdm(total=n_val, desc='Validation round', unit='mat', leave=False) as pbar:
        for batch in loader:
            mats = batch['mat_input']
            pcds = batch['mat_gt']

            mats = mats.to(device=device, dtype=torch.float32)
            pcds = pcds.to(device=device, dtype=torch.float32)
            

            with torch.no_grad():
                mats_pred = net(mats)

            tot += L1_loss(mats_pred, pcds)

            pbar.update(mats.shape[0])
    return tot / n_val
