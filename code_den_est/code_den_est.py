## loading packages
import numpy as np
import scipy.stats
import torch
import scipy
import matplotlib.pyplot as plt

from model_train import *
from model import *

## configuration
D = 200
learning_rate = 1e-3
n = 1000
p_x = 1

device = torch.device("cpu")

## generate data
seed = 42
np.random.seed(seed)

mat_x = np.random.randn(n, p_x) * 2.0 + 1.0
mat_x_pred = np.random.randn(n, p_x) * 2.0 + 1.0

ts_x = torch.from_numpy(mat_x).float().to(device)
ts_x_pred = torch.from_numpy(mat_x_pred).float().to(device)

model_x, optimizer, omega = den_X(
    ts_x, D = D, learning_rate = learning_rate, 
    reg = 20 * np.log(n)
    # reg = 1e-5
)

den_x_est = model_x.pred(ts_x_pred).detach().numpy()

den_x_est[den_x_est < np.quantile(den_x_est, 0.05)] = np.quantile(den_x_est, 0.05)

den_x_truth = scipy.stats.multivariate_normal.pdf(
    mat_x_pred, 
    mean = np.ones(p_x), 
    cov = np.eye(p_x) * 2.0
)

# plt.scatter(
#     den_x_truth, 
#     den_x_est, 
#     s = 1, alpha = 0.5
# )    

plt.scatter(
    mat_x_pred, 
    den_x_est, 
    s = 1, alpha = 0.5
)

plt.savefig(
    "den_x_cpr_test.pdf", 
    format = "pdf"
)
