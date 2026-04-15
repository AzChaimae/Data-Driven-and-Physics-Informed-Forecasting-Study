# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:37:55 2026

@author: AZROULMAHLI Chaimae
"""

import numpy as np
from config import CONFIG

def gaussian_seed(nx, ny, sigma=4):
    x = np.arange(nx)
    y = np.arange(ny)
    X,Y = np.meshgrid(x,y)
    cx,cy = nx//2, ny//2
    G = np.exp(-((X-cx)**2 + (Y-cy)**2)/(2*sigma**2))

    return G/G.max()*0.1


def laplacian(Z):
    return (
        np.roll(Z,1,0) +
        np.roll(Z,-1,0) +
        np.roll(Z,1,1) +
        np.roll(Z,-1,1) -
        4*Z
    )


def simulate_spatial_seir():
    
    nx = CONFIG["grid_size"]
    ny = CONFIG["grid_size"]
    T = CONFIG["time_steps"]
    
    beta = CONFIG["beta"]
    kappa = CONFIG["kappa"]
    gamma = CONFIG["gamma"]

    DE = CONFIG["diffusion_E"]
    DI = CONFIG["diffusion_I"]

    dt = 0.1

    S = np.ones((nx,ny))
    E = np.zeros((nx,ny))
    I = gaussian_seed(nx,ny)
    R = np.zeros((nx,ny))

    S -= I

    dataset = []

    for t in range(T):

        lap_I = laplacian(I)
        lap_E = laplacian(E)

        S_new = S - dt*beta*S*I
        E_new = E + dt*(beta*S*I - kappa*E + DE*lap_E)
        I_new = I + dt*(kappa*E - gamma*I + DI*lap_I)
        R_new = R + dt*(gamma*I)

        S,E,I,R = S_new,E_new,I_new,R_new

        dataset.append(I.copy())

    return np.array(dataset)