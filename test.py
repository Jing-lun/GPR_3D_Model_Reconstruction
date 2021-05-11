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
from tqdm import tqdm
import h5py
import scipy.io
from utils.utils import PointLoss
import argparse
from model import UNet3D
import logging
import torch.nn.functional as F
import open3d as o3d
import numpy as np

def predict_net(net, mat, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mat = torch.from_numpy(mat)
    mat = mat.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        mat_pred = net(mat)
    mat_pred = mat_pred.squeeze().cpu().numpy()
    mat_pred = mat_pred.transpose(1,0,2)
    pcd = o3d.geometry.PointCloud()

    points = []
    a, b, c = mat_pred.shape
    for i in range(a):
        for j in range(b):
            for k in range(c):
                if mat_pred[i][j][k] >= 0.7:
                    points.append([j*0.002, i*0.002, k*0.002])
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd]) 
    pcd_name = 'test_pcd/nips/' + args.file[:-4] + '.pcd'
    print(pcd_name)
    o3d.io.write_point_cloud(pcd_name, pcd)

    
    print(mat_pred.shape)
    mat_name = 'test_pcd/nips/' + args.file[:-4] + '.mat'
    scipy.io.savemat(mat_name, {'occupancy_map':mat_pred})

def args_setting():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='check_points/CP_epoch101.pth', metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', dest='input', default='../test_data/input/',
                        help='input path')
    parser.add_argument('--gt', '-g', metavar='GT', dest='gt', default='../test_data/gt/',
                        help='gt path')                   
    parser.add_argument('--file', '-f', metavar='FILE', dest = 'file', default='16.mat',
                        help='filename')

    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = args_setting()

    net = UNet3D(residual='conv')

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)

    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    print(args.input + args.file)
    with h5py.File(args.input + args.file, 'r') as f:
        dset = f['RawImage']
        mat = dset[:]
        mat = mat.transpose(1,0,2)
        shape = mat.shape
        mat = mat.reshape(1, shape[0], shape[1], shape[2])
    predict_pcd = predict_net(net=net,
                           mat=mat,
                           device=device)
    
    print(args.gt + args.file)                       
    f = scipy.io.loadmat(args.gt + args.file)
    occ_map = f['occupancy_map']
    pcd = o3d.geometry.PointCloud()
    points = []
    a, b, c = occ_map.shape
    for i in range(a):
        for j in range(b):
            for k in range(c):
                if occ_map[i][j][k] != 0:
                    points.append([j*0.002, i*0.002, k*0.002])
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    pcd_name = 'test_pcd/gt/' + args.file[:-4] + '.pcd'
    print(pcd_name)
    o3d.io.write_point_cloud(pcd_name, pcd) 
