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

mat_x = np.random.randn(n, p_x) + np.random.binomial(
    n = 1, p = 0.5, size = (n, p_x)
) * 10.0 - 5.0
mat_x_pred = np.random.randn(n, p_x) + np.random.binomial(
    n = 1, p = 0.5, size = (n, p_x)
) * 10.0 - 5.0

# plt.hist(mat_x_pred, bins = 30, density = True, alpha = 0.5, label = "x_pred")
# plt.savefig(
#     "den_x_mixGau_hist.pdf", 
#     format = "pdf"
# )

# exit("stop")

ts_x = torch.from_numpy(mat_x).float().to(device)
ts_x_pred = torch.from_numpy(mat_x_pred).float().to(device)

model_x, optimizer, omega = den_X(
    ts_x, D = D, learning_rate = learning_rate, 
    # reg = 20 * np.log(n)
    reg = 1
    # reg = 1e-2
    # reg = 1e-3
)

den_x_est = model_x.pred(ts_x_pred).detach().numpy()

den_x_est[den_x_est < np.quantile(den_x_est, 0.05)] = np.quantile(den_x_est, 0.05)

fig, axs = plt.subplots(2)

# den_x_truth = scipy.stats.multivariate_normal.pdf(
#     mat_x_pred, 
#     mean = np.ones(p_x), 
#     cov = np.eye(p_x) * 2.0
# )

# axs[0].scatter(
#     den_x_truth, 
#     den_x_est, 
#     s = 1, alpha = 0.5
# )    
    
axs[0].scatter(
    mat_x_pred, 
    den_x_est, 
    s = 1, alpha = 0.5
)

axs[1].hist(
    (model_x.beta).detach().numpy(), 
    bins = 30, density = True, alpha = 0.5, label = "x_pred"
)

fig.savefig(
    "out_den_est_mixGau_test.pdf", 
    format = "pdf"
)


