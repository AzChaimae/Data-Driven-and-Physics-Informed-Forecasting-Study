# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:37:55 2026

@author: AZROULMAHLI Chaimae
"""

from scipy.stats import ttest_rel
import pandas as pd


def compare_models(errorsA, errorsB):

    stat,p = ttest_rel(errorsA, errorsB)

    return stat,p


def save_statistical_results(results, path):

    df = pd.DataFrame(results)

    df.to_csv(path, index=False)