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
ts_x = torch.from_numpy(mat_x).float().to(device)

model_x, optimizer, omega = den_X(ts_x, D = D, learning_rate = learning_rate, reg = np.log(n))

den_x_est = model_x.pred(ts_x).detach().numpy()

plt.scatter(mat_x, den_x_est, alpha=0.5, s=1)

plt.savefig(
    "den_x_test.pdf", 
    format = "pdf"
)
