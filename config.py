# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:37:55 2026

@author: AZROULMAHLI Chaimae
"""
import torch

CONFIG = {

"grid_size": 40,
"time_steps": 120,

"history_window": 10,

"train_split": 0.7,

"beta": 1.2,
"kappa": 0.5,
"gamma": 0.3,

"diffusion_E": 0.02,
"diffusion_I": 0.04,

"epochs": 30,
"learning_rate": 1e-3,

"device": "cuda" if torch.cuda.is_available() else "cpu",

"output_dir": "outputs"

}