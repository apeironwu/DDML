## loading packages
import dml
import numpy as np
import torch
import matplotlib.pyplot as plt

## set parameters
K_fold = 3
n = 2000; p = 5
beta = -2.0
learning_rate = 0.01
S = 5

# seed_np = 2026
# if True:
for seed_np in range(2025, 2030): 

    np.random.seed(seed_np)
    torch.manual_seed(seed_np)

    ls_site = [None] * S
    ls_dml_loc = [None] * S

    vec_beta_loc = np.zeros(S, dtype=np.float32)

    ## data generation and local estimation
    for s in range(S): 
        mat_x = np.random.randn(n, p)
        vec_d = mat_x[:, 0] + np.random.randn(n)
        vec_y_prob = dml.fun_expit(0.5 + vec_d * beta + 1 * mat_x[:, 1])
        vec_y = np.random.binomial(1, vec_y_prob)

        ls_site[s] = dml.DataSite(vec_y, vec_d, mat_x, K_fold)
        ls_site[s].model_train_est()

        model_dml_cur, _, bl_conv_cur = dml.train_logistic_dml(ls_site[s], learning_rate)

        vec_beta_loc[s] = model_dml_cur.beta.detach().numpy()
        ls_dml_loc[s] = model_dml_cur

    print(">> beta_loc:  ", vec_beta_loc.round(3), np.mean(vec_beta_loc).round(3))

    vec_weight = np.ones(S) / S
    # print(vec_weight)

    beta_ini = np.sum(vec_beta_loc * vec_weight)

    vec_score_loc = np.zeros(S, dtype=np.float32)

    for s in range(S): 
        ls_dml_loc[s].beta = torch.nn.Parameter(torch.tensor(beta_ini, dtype=torch.float32))
        vec_score_loc[s] = ls_dml_loc[s].score().detach().numpy()
    score_loc = np.sum(vec_score_loc * vec_weight)

    ls_op_site_cen = [dml.OperaSiteCen(s, ls_site) for s in range(S)]

    ls_model_fdml = [None] * S
    vec_beta_fdml = np.zeros(S, dtype=np.float32)

    for s in range(S): 
        model_fdml_cur, _, bl_conv_cur = dml.train_logistic_fdml(
            ls_site[s], ls_op_site_cen[s], 
            score_loc, beta_ini, learning_rate
        )
        ls_model_fdml[s] = model_fdml_cur
        vec_beta_fdml[s] = model_fdml_cur.beta.detach().numpy()

    print(">> beta_fdml: ", vec_beta_fdml.round(3), np.mean(vec_beta_fdml).round(3))




# vec_beta_test = np.linspace(-2.5, -1.5, 1000)
# vec_score_test = np.zeros_like(vec_beta_test, dtype=np.float32)

# for i in range(len(vec_beta_test)):
#     model_fdml_cur.beta = torch.nn.Parameter(torch.tensor(vec_beta_test[i], dtype=torch.float32))
#     vec_score_test[i] = model_fdml_cur.score2().detach().numpy()

# dml.fig_check(vec_beta_test, vec_score_test)







# print(
#     ">> beta: {} - score: {}".format(
#         ls_dml_loc[0].beta.detach().numpy(), 
#         ls_dml_loc[0].score()
#     )
# )

# ls_dml_loc[0].beta = torch.nn.Parameter(torch.tensor(beta_ini, dtype=torch.float32))
# ls_dml_loc[0].score()

# print(
#     ">> beta: {} - score: {}".format(
#         ls_dml_loc[0].beta.detach().numpy(), 
#         ls_dml_loc[0].score()
#     )
# )





