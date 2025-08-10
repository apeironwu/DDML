# -*- coding: utf-8 -*-
import torch
import numpy as np

from model import *

device = torch.device("cpu")

def draw_spectral_SE(d, D):
  return torch.randn((d,D))

def den_X(train_x, D = 100, learning_rate = 1e-3, reg = 1e-3, n_iter = 5000):
    
    # Generate omega, shared parameters for SE kernel
    omega = draw_spectral_SE(d = train_x.shape[1], D = D)

    model_x = ModelX(
        x = train_x, 
        omega = omega,
        D = D, 
        reg = reg
    ).to(device) 

    optimizer = torch.optim.Adam(model_x.parameters(), lr = learning_rate)

    model_x.train()
    for iter in range(n_iter):
        optimizer.zero_grad()
        loss = model_x.loss()
        loss.backward()
        optimizer.step()

        if iter % int(n_iter / 10) == 0: 
            print(
              f"Iteration {iter}, Loss: {loss.item()}, ell: {torch.exp(model_x.ell_log).item()}"
            )

    return model_x, optimizer, omega
