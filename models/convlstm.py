# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:37:55 2026

@author: AZROULMAHLI Chaimae
"""

import torch
import torch.nn as nn


class ConvLSTM(nn.Module):

    def __init__(self, window):

        super().__init__()

        self.conv1 = nn.Conv2d(window, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)

        return x
