## loading packages
import dml_bt as dml
import numpy as np
import torch
import matplotlib.pyplot as plt

## set parameters
K_fold = 3; n = 1000; p = 2; beta = -2.0

learning_rate = 0.01

seed_np = 2026
# if True:
# for seed_np in range(2025, 2030): 

np.random.seed(seed_np)
torch.manual_seed(seed_np)

## data generation
mat_x = np.random.randn(n, p)

vec_d_prob = dml.fun_expit(-1.0 + mat_x[:, 0])
vec_d = np.random.binomial(1, vec_d_prob)

vec_y_prob = dml.fun_expit(vec_d * beta + 1.0 * mat_x[:, 1])
vec_y = np.random.binomial(1, vec_y_prob)

print(">> Pr(d = 1), Pr(y = 1): ", vec_d.mean().round(4), vec_y.mean().round(4))

site1 = dml.DataSite(vec_y, vec_d, mat_x, K_fold)

site1.model_train_m()

site1.model_est_m()

site1.model_train_gam()

site1.model_est_gam()

model, optimizer, _ = dml.train_logistic_dml(site1, learning_rate)

# print(">> optimizer: ", optimizer)

print(">> beta estimate: ", model.beta.item())




# ## sanity check - w_hat
# dml.fig_check(
#     site1.mat_w[site1.lab_ds != 0, 0], 
#     vec_y[site1.lab_ds != 0]
# )

# plt.hist(
#     [
#         site1.mat_w[site1.lab_ds != 0, 0][
#             vec_y[site1.lab_ds != 0] == 1
#         ], 
#         site1.mat_w[site1.lab_ds != 0, 0][
#             vec_y[site1.lab_ds != 0] == 0
#         ] 
#     ], 
#     bins=30, color=['skyblue', 'lightcoral'], edgecolor='black'
# ); plt.show()






# ## sanity check - beta
# bd = 0.5
# vec_beta_test = np.linspace(beta - bd, beta + bd, 200)
# vec_score_test = np.zeros_like(vec_beta_test, dtype=np.float32)

# for i in range(len(vec_beta_test)):
#     model.beta = torch.nn.Parameter(torch.tensor(vec_beta_test[i], dtype=torch.float32))
#     # vec_score_test[i] = model.score().detach().numpy()
#     vec_score_test[i] = model.score2().detach().numpy()

# dml.fig_check(vec_beta_test, vec_score_test)







