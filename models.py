#!/usr/bin/env Python
# coding=utf-8

import torch
import math
import torchvision
import torch.nn as nn
from torch.nn import Linear, ReLU, Tanh
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import itertools
import numpy as np


class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    # torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std= 0.1)

    def forward(self, x):
        feat1 = self.relu(self.fc1(x))
        # feat1 = feat1 + self.relu(self.fc2(self.dropout(feat1)))
        hid = self.fc_encode(self.dropout(feat1))
        code = torch.tanh(self.alpha * hid)

        return x, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len=512 ):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std=0.3)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat))
        code = torch.tanh(self.alpha * hid)

        return feat, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
