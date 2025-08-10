# -*- coding: utf-8 -*-
import torch
import numpy as np

from den_est_rkhs_rff import *

device = torch.device("cpu")

def draw_spectral_SE(d, D):
  return torch.randn((d,D))

def den_est_rkhs_rff_train(train_x, D = 100, learning_rate = 1e-3, reg = 1e-3, n_iter = 1000):
    
  # Generate omega, shared parameters for SE kernel
  omega = draw_spectral_SE(d = train_x.shape[1], D = D)

  model_den_est = Den_est_rkhs_rff(
    x = train_x, 
    omega = omega,
    D = D, 
    reg = reg
  ).to(device) 

  optimizer = torch.optim.Adam(model_den_est.parameters(), lr = learning_rate)

  model_den_est.train()
  for iter in range(n_iter):
    optimizer.zero_grad()
    loss = model_den_est.loss()
    loss.backward()
    optimizer.step()

    # if iter % int(n_iter / 10) == 0: 
    #   print(
    #     # f"Iteration {iter}, Loss: {loss.item()}, ell: {torch.exp(model_den_est.ell_log).item()}"
    #     f"Iteration {iter}, Loss: {loss.item()}"
    #   )

  return model_den_est, optimizer, omega