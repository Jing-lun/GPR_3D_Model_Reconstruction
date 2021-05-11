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

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from model import UNet3D
from utils.data_loader import BasicDataset
from utils.utils import PointLoss
from eval import eval_net

def train_net(net,
              epochs,
              batch_size,
              lr,
              device,
              save_cp = True):
    dset = BasicDataset(args.input, args.gt)
    n_train = int(len(dset) * 0.85)
    n_val = len(dset) - n_train
    train, val = random_split(dset, [n_train, n_val])
    dset_train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    dset_valid = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    writer = SummaryWriter(comment=f'BS_{2}')

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)
    L1_loss = nn.L1Loss()
    L1_loss.to(device)
    global_step = 0

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch+1}/{epochs}', unit='mat') as pbar:
            for batch in dset_train:
                mats = batch['mat_input']
                pcds = batch['mat_gt']

                mats = mats.to(device=device, dtype=torch.float32)
                pcds = pcds.to(device=device, dtype=torch.float32)

                test = pcds*6 + 1

                optimizer.zero_grad()
                mats_pred = net(mats)
                new_predict = test * mats_pred
                new_ground_truth = 7*pcds

                loss = L1_loss(new_predict, new_ground_truth)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                loss.backward()
                optimizer.step()

                pbar.update(mats.shape[0])
                global_step += 1
            val_score = eval_net(net, dset_valid, device, n_val)
            logging.info(f'Validation L1 Distance: {val_score}')
            writer.add_scalar('Loss/test', val_score, global_step)


        scheduler.step()
        if epoch % 20 == 0:
            torch.save(net.state_dict(),
                           'check_points/' + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

def args_setting():
    parser = argparse.ArgumentParser(description='Train the net on gpr data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=101,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default='check_points/good_627/CP_epoch101.pth',
                        help='Load model from a .pth file')
    parser.add_argument('-i', '--input', default='../resnet_range/',
                        type=str, metavar='PATH', help='path to input dataset', dest='input')
    parser.add_argument('-g', '--ground-truth', default='../new_mat_gt/',
                        type=str, metavar='PATH', help='path to gt dataset', dest='gt')
    parser.add_argument('-c', '--checkpoint', default='check_point/',
                        type=str, metavar='PATH', help='path to gt dataset', dest='cp')

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = args_setting()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Let\'s use {torch.cuda.device_count()} GPUs!')

    net = UNet3D(residual='conv')
    net = torch.nn.DataParallel(net)
    if args.load != '':
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    logging.info(f'Network Structure:\n'
                 f'\t{net}\n')
    net.to(device=device)


    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
