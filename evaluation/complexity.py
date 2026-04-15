# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 02:33:23 2026

@author: AZROUMAHLI Chaimae
"""

import torch
#import os


def model_size(model):

    param_size = 0

    for param in model.parameters():
        param_size += param.numel()

    size_mb = param_size * 4 / (1024**2)

    return param_size, size_mb


def memory_usage():

    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)

    return 0


def inference_time(model, sample):

    import time

    start = time.time()

    model(sample)

    return time.time() - start