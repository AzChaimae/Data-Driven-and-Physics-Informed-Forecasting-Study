# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:37:55 2026

@author: AZROULMAHLI Chaimae
"""
import pandas as pd
import os


def save_results(results):

    path = "outputs/results/benchmark_results.csv"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame(results)

    df.to_csv(path, index=False)

    print("\nBenchmark results saved:", path)