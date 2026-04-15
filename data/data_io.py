# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:37:55 2026

@author: AZROULMAHLI Chaimae
"""

import os
import numpy as np
import pandas as pd
from config import CONFIG

def save_numpy_dataset(data, name):
    path = f"{CONFIG['output_dir']}/datasets/{name}.npy"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)


def save_csv_dataset(data, name):
    path = f"{CONFIG['output_dir']}/datasets/{name}.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = data.reshape(data.shape[0], -1)
    df = pd.DataFrame(flat)
    df.to_csv(path, index=False)