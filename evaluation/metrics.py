# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:37:55 2026

@author: AZROULMAHLI Chaimae
"""

import numpy as np
from sklearn.metrics import mean_squared_error


def compute_metrics(pred, true):

    """
    Compute evaluation metrics for forecasting models.

    Automatically reshapes predictions and targets
    so they can be compared regardless of model format.
    """

    # convert predictions to numpy
    pred = np.array(pred)
    true = np.array(true)

    # reshape both tensors to 2D (samples, features)
    pred_flat = pred.reshape(pred.shape[0], -1)
    true_flat = true.reshape(true.shape[0], -1)

    # Mean Squared Error
    mse = mean_squared_error(true_flat, pred_flat)

    # Relative L2 Error
    rel = np.linalg.norm(pred_flat - true_flat) / np.linalg.norm(true_flat)

    # Peak infection error
    peak_error = abs(pred_flat.max() - true_flat.max())

    return {
        "MSE": mse,
        "Relative_L2": rel,
        "PeakError": peak_error
    }