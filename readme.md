# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:45:35 2026

@author: AZROUMAHLI Chaimae
"""

# Machine Learning Benchmark for Forecasting Agricultural Disease Epidemics

This repository provides a **research-grade experimental framework** for benchmarking machine learning and deep learning models for **forecasting agricultural disease epidemics**.

The framework combines **mechanistic epidemiological simulation**, **data-driven forecasting models**, and **systematic benchmarking tools**. It is designed to support reproducible experiments for evaluating forecasting methods in agricultural epidemiology.

The implementation follows the experimental pipeline described in the accompanying research paper.

---

# Project Objective

The goal of this framework is to compare different machine learning approaches for forecasting plant disease outbreaks using a **synthetic spatio-temporal dataset generated from a spatial SEIR epidemiological model**.

The framework enables evaluation of models based on:

- forecasting accuracy
- computational performance
- robustness to data conditions
- statistical significance of results

---

# Main Features

The framework includes:

• Synthetic **spatial epidemic simulation** using a reaction–diffusion SEIR model  
• Construction of **spatio-temporal forecasting datasets**  
• Implementation of several forecasting models:

- Random Forest (baseline ML)
- ConvLSTM (spatio-temporal deep learning)
- Neural Ordinary Differential Equation model
- Physics-Informed Neural Network (PINN)

• Automatic **experiment runner and benchmarking pipeline**

• Evaluation metrics including:

- Mean Squared Error (MSE)
- Relative L2 error
- Peak infection error

• **Statistical model comparison** using t-tests

• **Ablation experiments** to analyze model sensitivity

• Automatic generation and saving of:

- datasets
- figures
- experiment results

---

# Project Structure


---

# Spatial Epidemic Simulation

The synthetic dataset is generated using a **spatial SEIR epidemiological model** describing the spread of plant disease across an agricultural landscape.

The model includes:

- susceptible plants
- exposed plants
- infectious plants
- removed plants

Spatial diffusion terms simulate pathogen dispersal between neighboring crop fields.

The simulation produces **spatio-temporal infection maps**: $I(x,y,t)$


representing disease intensity across the simulated agricultural region.

---

# Dataset Construction

The spatial epidemic simulation generates infection maps over time.

These maps are converted into a forecasting dataset using a **sliding temporal window**:

Input: $[I(t-k), ..., I(t)]$
Target : $I(t + Δt)$


This produces a supervised learning dataset suitable for training forecasting models.

---

# Forecasting Models

The framework currently includes the following models:

### Random Forest

Classical machine learning baseline using ensemble decision trees.

Advantages:
- fast training
- low computational cost

Limitations:
- limited ability to capture spatial dependencies

---

### ConvLSTM

Deep learning architecture for spatio-temporal sequence prediction.

Advantages:
- captures spatial propagation patterns
- effective for epidemic forecasting

Limitations:
- higher GPU and memory requirements

---

### Neural Ordinary Differential Equation

Neural ODE models learn continuous-time epidemic dynamics.

Advantages:
- flexible representation of dynamical systems
- suitable for time-continuous prediction

---

### Physics-Informed Neural Network

PINNs incorporate epidemiological equations directly into the training process.

Advantages:
- physically consistent predictions
- integrates mechanistic knowledge with data-driven learning

---

# Running the Framework

## 1 Install dependencies

Required packages include:
numpy
pandas
matplotlib
scikit-learn
torch
scipy


Install with:


pip install numpy pandas matplotlib scikit-learn torch scipy


---

## 2 Run the experiment pipeline

Execute:


python main.py


The script will:

1 Generate a synthetic spatial epidemic dataset  
2 Construct forecasting samples  
3 Train forecasting models  
4 Evaluate prediction performance  
5 Save datasets, figures, and results

---

# Output Files

Running the experiment automatically generates the following outputs.

## Datasets


outputs/datasets/

spatial_seir_dataset.npy
spatial_seir_dataset_flat.csv


These files contain the synthetic epidemic simulation.

---

## Figures


outputs/figures/

epidemic_heatmaps.png


These heatmaps visualize spatial epidemic propagation.

---

## Benchmark Results


outputs/results/

benchmark_results.csv


This file contains evaluation metrics and computational performance for each forecasting model.

---

# Evaluation Metrics

The following metrics are used for model evaluation.

Mean Squared Error

Measures average prediction error between predicted and true infection levels.

Relative L2 Error

Measures normalized prediction error relative to the magnitude of the epidemic.

Peak Infection Error

Measures the difference between predicted and true epidemic peak intensity.

---

# Statistical Analysis

The framework includes statistical comparison between models using paired t-tests.

This analysis determines whether observed performance differences between models are statistically significant.

---

# Ablation Experiments

Ablation experiments analyze the influence of different factors on forecasting performance.

Examples include:

- history window size
- dataset size
- noise levels

These experiments help identify which factors most strongly affect model performance.

---

# Reproducibility

The framework is designed for reproducible research:

- datasets are saved locally
- figures are automatically generated
- experiment results are exported as CSV files

All experiments can therefore be replicated and verified.

---

# Using te framework

"requierment.txt" defines all Python dependencies required to run the experiments.

Users can install everything with: pip install -r requirements.txt



# Intended Use

This framework is intended for:

- Benchmarking forecasting methods in epidemiology
- studying machine learning approaches for agricultural disease prediction
- generating controlled synthetic datasets for research

---

# License

This repository is intended for academic and research use.



