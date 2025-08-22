## loading packages
import dml
import numpy as np
import torch
import matplotlib.pyplot as plt

## set parameters
K_fold = 3

n = 2000; p = 2; S = 50
# n = 2000; p = 5; S = 2

beta = -2.0
learning_rate = 0.01
n_iter = 2

seed_np = 2026
if True:
# for seed_np in range(2025, 2030): 

    np.random.seed(seed_np)
    torch.manual_seed(seed_np)

    ls_site = [None] * S
    ls_dml_loc = [None] * S

    vec_beta_loc = np.zeros(S, dtype=np.float32)
    vec_var_est = np.zeros_like(vec_beta_loc)

    ## data generation and local estimation
    # if True:
    #     s = 1
    for s in range(S): 
        
        mat_x = np.random.randn(n, p)
        vec_d = mat_x[:, 0] + np.random.randn(n)
        vec_y_prob = dml.fun_expit(vec_d * beta + dml.fun_expit(mat_x[:, 1]))
        vec_y = np.random.binomial(1, vec_y_prob)

        ls_site[s] = dml.DataSite(vec_y, vec_d, mat_x, K_fold)
        ls_site[s].model_train_est()

        model_dml_cur, _, bl_conv_cur = dml.train_logistic_dml(ls_site[s], learning_rate)

        if bl_conv_cur is False:
            print("Model did not converge")
            continue

        print(
            ">> s =", s, "  |  ", 
            bl_conv_cur, "  |  ", 
            model_dml_cur.beta.detach().numpy().round(4), 
            model_dml_cur.var().detach().numpy().round(4), 
        )

        vec_beta_loc[s] = model_dml_cur.beta.detach().numpy()
        vec_var_est[s] = model_dml_cur.var().detach().numpy()
        ls_dml_loc[s] = model_dml_cur
    
    dml.fig_check(vec_beta_loc, vec_var_est)


