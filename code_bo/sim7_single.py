## loading packages
import dml
import numpy as np
import torch
import matplotlib.pyplot as plt

## set parameters
K_fold = 3
# n = 1000; p = 10
n = 1000; p = 2
# beta = 1.5
beta = -2.0
# beta = -5.0
learning_rate = 0.01

seed_np = 2026
# if True:
# for seed_np in range(2025, 2030): 

np.random.seed(seed_np)
torch.manual_seed(seed_np)

## data generation
mat_x = np.random.randn(n, p)
vec_d = mat_x[:, 0] + np.random.randn(n)
vec_y_prob = dml.fun_expit(vec_d * beta + 1 * mat_x[:, 1])
vec_y = np.random.binomial(1, vec_y_prob)

site1 = dml.DataSite(vec_y, vec_d, mat_x, K_fold)

site1.model_train_m()

site1.model_est_m()

site1.model_train_gam()

site1.model_est_gam()

model, optimizer = dml.train_logistic_dml(site1, learning_rate)

print(">> optimizer: ", optimizer)

print(">> beta estimate: ", model.beta.item())

