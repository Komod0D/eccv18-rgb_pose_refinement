import warnings
from typing import Optional, List, Callable, Tuple

import torch
from torch import nn
from torch import Tensor
from RealInception import InceptionA, BasicConv2d, InceptionOutputs
from torchvision.models.inception import *


class Architecture(nn.Module):
    def __init__(self):
        super(Architecture, self).__init__()

        self.head = InceptionHead()

        self.mid_conv = BasicConv2d(288 * 2, 192, kernel_size=3, stride=2)
        self.mid_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.inception_mid1 = InceptionA(288 * 2 + 192, pool_features=64)
        self.inception_mid2 = InceptionA(288, pool_features=64)

        self.inception_r_1 = InceptionA(288, pool_features=64)
        self.inception_r_2 = InceptionA(288, pool_features=64)

        self.conv_r_mid = BasicConv2d(288, 64, kernel_size=3, stride=2)  # to change
        self.conv_r_out = BasicConv2d(64, 4, kernel_size=6, stride=1)

        self.inception_t_1 = InceptionA(288, pool_features=64)
        self.inception_t_1 = InceptionA(288, pool_features=64)

        self.conv_t_mid = BasicConv2d(288, 64, kernel_size=3, stride=2)  # to change
        self.conv_t_out = BasicConv2d(64, 3, kernel_size=6, stride=1)

    def forward(self, real: Tensor, render: Tensor) -> Tensor:
        real = self.head(real)
        render = self.head(render)

        x = torch.cat((real, render), 1)
        x = torch.cat((self.mid_conv(x), self.mid_pool(x)), 1)
        x = self.inception_mid2(self.inception_mid1(x))

        r = self.inception_r_2(self.inception_r_1(x))
        r = self.conv_r_out(self.conv_r_mid(r))

        t = self.inception_r_2(self.inception_r_1(x))
        t = self.conv_t_mid(self.conv_t_out(t))

        return r, t


class InceptionHead(nn.Module):
    def __init__(self) -> None:
        super(InceptionHead, self).__init__()
        conv_block = BasicConv2d
        inception_a = InceptionA

        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._forward(x)
        if torch.jit.is_scripting():
            return x


