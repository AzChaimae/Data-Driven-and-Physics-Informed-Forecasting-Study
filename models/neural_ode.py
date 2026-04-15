# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:37:55 2026

@author: AZROULMAHLI Chaimae
"""
import torch
import torch.nn as nn


class ODEFunc(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):

        return self.net(x)


class NeuralODE(nn.Module):

    def __init__(self, window, nx, ny):

        super().__init__()

        self.window = window
        self.nx = nx
        self.ny = ny

        input_dim = window * nx * ny

        self.func = ODEFunc(input_dim)

        # projection to next infection map
        self.output_layer = nn.Linear(input_dim, nx * ny)

    def forward(self, x):

        batch = x.shape[0]

        # flatten
        x = x.view(batch, -1)

        dx = self.func(x)

        out = self.output_layer(dx)

        # reshape into spatial map
        out = out.view(batch, 1, self.nx, self.ny)

        return out