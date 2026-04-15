# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:37:55 2026

@author: AZROULMAHLI Chaimae
"""

import os
import time
import torch
#import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG

# simulation
from simulation.spatial_seir import simulate_spatial_seir

# data utilities
from data.dataset_builder import build_dataset
from data.data_io import save_numpy_dataset, save_csv_dataset

# models
from models.random_forest import RandomForestModel
from models.convlstm import ConvLSTM
from models.neural_ode import NeuralODE
from models.pinn import PINN

# training
from training.trainer import train_model

# evaluation
from evaluation.metrics import compute_metrics
from evaluation.statistics import compare_models
from evaluation.ablation import ablation_history
from evaluation.complexity import model_size, inference_time

# visualization
from visualization.plots import (
    save_heatmaps,
    save_prediction_plot
)

device = CONFIG["device"]


def prepare_dataloader(X,Y):

    X = torch.tensor(X).float()
    Y = torch.tensor(Y).float()

    dataset = TensorDataset(X,Y)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    return loader


def evaluate_model(model,X_test,Y_test,model_name):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X_test).float().to(device)
        pred = model(X).cpu().numpy()

    metrics = compute_metrics(pred,Y_test)
    save_prediction_plot(pred,Y_test,model_name)
    return metrics,pred


def benchmark_model(model,model_name,X_train,Y_train,X_test,Y_test):

    loader = prepare_dataloader(X_train,Y_train)

    model,losses,train_time = train_model(
        model,
        loader,
        CONFIG["epochs"],
        model_name
    )

    metrics,pred = evaluate_model(
        model,
        X_test,
        Y_test,
        model_name
    )

    params,size = model_size(model)

    sample = torch.tensor(X_test[:1]).float().to(device)

    infer_time = inference_time(model,sample)

    results = {

        "Model":model_name,
        "TrainTime":train_time,
        "InferenceTime":infer_time,
        "Parameters":params,
        "ModelSizeMB":size,
        "MSE":metrics["MSE"],
        "RelativeL2":metrics["Relative_L2"],
        "PeakError":metrics["PeakError"]
    }

    return results,pred


def run_all_models(X_train,Y_train,X_test,Y_test):

    results = []
    predictions = {}

    # ConvLSTM
    conv_model = ConvLSTM(CONFIG["history_window"]).to(device)

    r,p = benchmark_model(
        conv_model,
        "ConvLSTM",
        X_train,
        Y_train,
        X_test,
        Y_test
    )

    results.append(r)
    predictions["ConvLSTM"] = p


    # Neural ODE
    #dim = X_train.shape[2]*X_train.shape[3]

    window = X_train.shape[1]
    nx = X_train.shape[2]
    ny = X_train.shape[3]

    ode_model = NeuralODE(window, nx, ny).to(device)

    r,p = benchmark_model(
        ode_model,
        "NeuralODE",
        X_train,
        Y_train,
        X_test,
        Y_test
    )

    results.append(r)
    predictions["NeuralODE"] = p


    # PINN
    window = X_train.shape[1]
    nx = X_train.shape[2]
    ny = X_train.shape[3]

    pinn_model = PINN(window, nx, ny).to(device)
    

    r,p = benchmark_model(
        pinn_model,
        "PINN",
        X_train,
        Y_train,
        X_test,
        Y_test
    )

    results.append(r)
    predictions["PINN"] = p

    
    # Random Forest
    rf = RandomForestModel()

    print("\nTraining model: RandomForest")

    start = time.time()
    print("\nBlaaah")
    rf.train(X_train,Y_train)
    print("\nBlaaah")

    train_time = time.time()-start

    start = time.time()

    pred = rf.predict(X_test)

    infer_time = time.time()-start

    metrics = compute_metrics(pred,Y_test)

    results.append({
        "Model":"RandomForest",
        "TrainTime":train_time,
        "InferenceTime":infer_time,
        "Parameters":"N/A",
        "ModelSizeMB":"N/A",
        "MSE":metrics["MSE"],
        "RelativeL2":metrics["Relative_L2"],
        "PeakError":metrics["PeakError"]
    })

    predictions["RandomForest"] = pred

    return results,predictions


def run_statistics(predictions,Y_test):

    models = list(predictions.keys())

    stats_results = []

    for i in range(len(models)):
      for j in range(i+1,len(models)):
        A = predictions[models[i]].flatten()
        B = predictions[models[j]].flatten()

        stat,p = compare_models(A,B)
        stats_results.append({
          "ModelA":models[i],
          "ModelB":models[j],
          "t_stat":stat,
          "p_value":p
        })

    return pd.DataFrame(stats_results)


def main():

    print("\n========== Epidemic Forecasting Benchmark ==========")

    print("\nGenerating spatial epidemic dataset...")

    data = simulate_spatial_seir()

    save_numpy_dataset(data,"spatial_seir_dataset")
    save_csv_dataset(data,"spatial_seir_dataset_flat")

    print("Dataset saved.")

    print("\nGenerating heatmap visualization...")

    save_heatmaps(data)

    print("Heatmaps saved.")

    print("\nConstructing forecasting dataset...")

    window1 = CONFIG["history_window"]

    X, Y = build_dataset(data, window1)


    split = int(CONFIG["train_split"] * len(X))

    X_train,X_test = X[:split],X[split:]
    Y_train,Y_test = Y[:split],Y[split:]

    print("\nRunning model benchmarks...")

    results,predictions = run_all_models(
        X_train,
        Y_train,
        X_test,
        Y_test
    )

    results_df = pd.DataFrame(results)

    os.makedirs("outputs/results",exist_ok=True)

    results_path = "outputs/results/benchmark_results.csv"

    results_df.to_csv(results_path,index=False)

    print("\nBenchmark results saved:",results_path)

    print("\nRunning statistical tests...")

    stats_df = run_statistics(predictions,Y_test)

    stats_path = "outputs/results/statistical_tests.csv"

    stats_df.to_csv(stats_path,index=False)

    print("Statistical results saved:",stats_path)

    print("\nRunning ablation experiments...")

    ablation_df = ablation_history(
        data,
        [3,5,10],
        None
    )
    
    ablation_path = "outputs/results/ablation_results.csv"
    ablation_df.to_csv(ablation_path,index=False)
    print("Ablation results saved:",ablation_path)
    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()