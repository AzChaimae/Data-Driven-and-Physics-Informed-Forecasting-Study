# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 01:37:55 2026

@author: AZROULMAHLI Chaimae
"""

import torch
import time
import os
from config import CONFIG


def train_model(model, train_loader, epochs, model_name):

    device = CONFIG["device"]
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = torch.nn.MSELoss()

    losses = []

    print(f"\n==============================")
    print(f"Training model: {model_name}")
    print(f"==============================")

    start_time = time.time()

    for epoch in range(epochs):

        epoch_loss = 0

        print(f"\nEpoch {epoch+1}/{epochs}")

        for i,(X,Y) in enumerate(train_loader):

            X,Y = X.to(device), Y.to(device)

            pred = model(X)

            #loss = criterion(pred.squeeze(), Y)
            loss = criterion(pred.view_as(Y), Y) #To avoid mismatches during training

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"Batch {i} | Loss: {loss.item():.6f}")

        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)

        print(f"Epoch Loss: {epoch_loss:.6f}")

    training_time = time.time() - start_time

    save_path = f"{CONFIG['output_dir']}/models/{model_name}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(model.state_dict(), save_path)

    print(f"\nModel saved to {save_path}")

    return model, losses, training_time