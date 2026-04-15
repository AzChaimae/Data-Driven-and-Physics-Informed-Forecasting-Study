# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:37:55 2026

@author: AZROULMAHLI Chaimae
"""

import numpy as np

def build_dataset(data, window):

    """
    Convert spatio-temporal epidemic dataset into
    supervised forecasting samples.

    Parameters
    ----------
    data : numpy array
        shape (T, Nx, Ny)

    window : int
        history window size

    Returns
    -------
    X : array
        shape (samples, window, Nx, Ny)

    Y : array
        shape (samples, 1, Nx, Ny)
    """

    X = []
    Y = []

    for t in range(len(data) - window - 1):

        X.append(data[t:t + window])

        Y.append(data[t + window])

    X = np.array(X)
    Y = np.array(Y)

    # add channel dimension
    Y = Y[:, None, :, :]

    return X, Y