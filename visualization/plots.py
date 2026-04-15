# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:37:55 2026

@author: AZROULMAHLI Chaimae
"""

import matplotlib.pyplot as plt
import os
from config import CONFIG

def save_heatmaps(data):

    times=[0,30,60,90]

    fig,axes=plt.subplots(1,4,figsize=(14,4))

    for ax,t in zip(axes,times):

        im=ax.imshow(data[t],cmap="hot")
        ax.set_title(f"time={t}")

    fig.colorbar(im,ax=axes.ravel().tolist())

    path=f"{CONFIG['output_dir']}/figures/epidemic_heatmaps.png"

    os.makedirs(os.path.dirname(path),exist_ok=True)

    plt.savefig(path,dpi=300)
    plt.close()
    

def save_prediction_plot(pred, true, name):

    plt.figure()

    plt.plot(pred.flatten()[:200], label="Prediction")
    plt.plot(true.flatten()[:200], label="Truth")

    plt.legend()

    path = f"outputs/figures/{name}.png"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.savefig(path)

    plt.close()