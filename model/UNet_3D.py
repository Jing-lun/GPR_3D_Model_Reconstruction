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

import scipy.io
from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d
from torch.nn import ReLU, Sigmoid
import torch
from torch import Tensor
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
# from .utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


class UNet3D(Module):
    # __                            __
    #  1|__   ________________   __|1
    #     2|__  ____________  __|2
    #        3|__  ______  __|3
    #           4|__ __ __|4
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity

    def __init__(self, num_channels=1, feat_channels=[4, 8, 16, 32, 64], residual='conv'):

        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

        super(UNet3D, self).__init__()

        layers = [2, 2, 2, 2]
        block = BasicBlock
        self.inplanes = 16
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(9, self.inplanes, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2,
                                       dilate=False)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                       dilate=False)

        # Encoder downsamplers
        self.pool1 = MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.pool2 = MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.pool3 = MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.pool4 = MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(2*feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block(2*feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2*feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2*feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv     = Conv3d(feat_channels[0], num_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.one_one_conv = Conv3d(8, num_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Activation function
        self.activation = Sigmoid()
    
    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        # norm_layer = self._norm_layer
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet
        # Encoder part
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        mat_pred = x.squeeze().cpu().numpy()

        shape = x.shape
        x = x.reshape(shape[0], 1, shape[1], shape[2], shape[3])
        
        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)

        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)

        # Decoder part
        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)
        res = self.one_conv(d_high1)
        seg = self.activation(self.one_conv(d_high1))

        return seg

class Conv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block, self).__init__()
        self.conv1 = Sequential(
                        Conv3d(inp_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        self.conv2 = Sequential(
                        Conv3d(out_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)

class Deconv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):

        super(Deconv3D_Block, self).__init__()

        self.deconv = Sequential(
                        ConvTranspose3d(inp_feat, out_feat, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                        ReLU())

    def forward(self, x):

        return self.deconv(x)

def sigmoid(x):
    s=1/(1+torch.exp(-6*x))
    return s

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet3D(residual='pool')
    net = net.to(device)
    # print(net)
    L1_loss = torch.nn.L1Loss()
    L1_loss.to(device)
    y = torch.randn(1, 1, 128, 176, 128)
    # x = torch.randn(1, 1, 352, 256, 256)
    # x = torch.randn(1, 1, 176, 128, 128)
    x = torch.randn(1, 9, 1404, 1024)
    x = x.cuda()
    y = y.cuda()
    output = net(x)
    loss = L1_loss(output, y)
    print(loss)
    print(output.shape)

    # test_input = torch.randint(0,2,(2,3,4))
    # print(test_input)
    # test_output = torch.randn(2,3,4)
    # print(test_output)
    # test = 6*(test_input) + 1
    # print(test)
    # test_in = test_output*test
    # print(test_in)
    # test_o  = 7*test_output
    # print(test_o)
