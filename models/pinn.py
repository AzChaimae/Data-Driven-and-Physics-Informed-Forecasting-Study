# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:37:55 2026

@author: AZROULMAHLI Chaimae
"""

import torch
import torch.nn as nn


class PINN(nn.Module):

    def __init__(self, window, nx, ny):

        super().__init__()

        self.window = window
        self.nx = nx
        self.ny = ny

        input_dim = window * nx * ny

        self.net = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.Tanh(),

            nn.Linear(256, 256),
            nn.Tanh(),

            nn.Linear(256, nx * ny)
        )

    def forward(self, x):

        batch = x.shape[0]

        # flatten input
        x = x.view(batch, -1)

        out = self.net(x)

        # reshape to spatial map
        out = out.view(batch, 1, self.nx, self.ny)

        return out