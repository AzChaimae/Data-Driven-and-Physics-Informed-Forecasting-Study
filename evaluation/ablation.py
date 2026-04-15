# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:37:55 2026

@author: AZROULMAHLI Chaimae
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Import dataset builder
from data.dataset_builder import build_dataset

# Import metrics
from evaluation.metrics import compute_metrics


def ablation_history(dataset, windows, trainer=None):
    """
    Runs an ablation study on the history window size.

    Parameters
    ----------
    dataset : numpy array
        Spatial epidemic dataset (T, Nx, Ny)

    windows : list
        List of history window sizes to test

    trainer : optional
        Trainer function if using deep learning models

    Returns
    -------
    DataFrame containing ablation results
    """

    results = []

    print("\n===== Ablation Study: History Window Size =====")

    for w in windows:

        print(f"\nRunning ablation with window = {w}")

        # build forecasting dataset
        X, Y = build_dataset(dataset, w)

        # train/test split
        split = int(0.7 * len(X))

        X_train = X[:split]
        X_test = X[split:]

        Y_train = Y[:split]
        Y_test = Y[split:]

        # Flatten for classical ML model
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_test_flat = X_test.reshape(len(X_test), -1)

        Y_train_flat = Y_train.reshape(len(Y_train), -1)
        Y_test_flat = Y_test.reshape(len(Y_test), -1)

        # Use Random Forest for ablation
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10
        )

        model.fit(X_train_flat, Y_train_flat)

        pred = model.predict(X_test_flat)

        # reshape predictions
        pred = pred.reshape(Y_test.shape)

        metrics = compute_metrics(pred, Y_test)

        results.append({

            "HistoryWindow": w,
            "MSE": metrics["MSE"],
            "RelativeL2": metrics["Relative_L2"],
            "PeakError": metrics["PeakError"]

        })

        print(
            f"Window={w} | MSE={metrics['MSE']:.6f} | "
            f"RelL2={metrics['Relative_L2']:.6f}"
        )

    results_df = pd.DataFrame(results)

    return results_df