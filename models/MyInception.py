import torch
from torch import nn
from RealInception import Inception3, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE
from torchvision.models.inception import *


class MyInception(nn.Module):
    def __init__(self):
        super(MyInception, self).__init__()

        self.inceptionv4_1 = None
        self.inceptionv4_2 = None
        InceptionE(a
