## loading packages
import dml
import numpy as np
import torch
import matplotlib.pyplot as plt

## set parameters
K_fold = 3
n = 2000; p = 2; S = 2
# n = 2000; p = 5; S = 2
beta = -2.0
learning_rate = 0.01
n_iter = 2

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
        vec_y_prob = dml.fun_expit(vec_d * beta + dml.fun_expit(mat_x[:, 1]))
        vec_y = np.random.binomial(1, vec_y_prob)

        ls_site[s] = dml.DataSite(vec_y, vec_d, mat_x, K_fold)
        ls_site[s].model_train_est()

        model_dml_cur, _, bl_conv_cur = dml.train_logistic_dml(ls_site[s], learning_rate)

        vec_beta_loc[s] = model_dml_cur.beta.detach().numpy()
        ls_dml_loc[s] = model_dml_cur
    
    vec_beta_test = np.linspace(beta - 1, beta + 1, 100)
    vec_var_test = np.zeros_like(vec_beta_test)
    vec_eh2_test = np.zeros_like(vec_beta_test)
    vec_i02_test = np.zeros_like(vec_beta_test)

    print(">> beta_cur: ", model_dml_cur.beta.detach().numpy())

    # ls_dml_loc[1].vec_m_est = torch.tensor(ls_site[1].mat_x[:, 0])
    # ls_dml_loc[1].vec_gam_est = torch.tensor(dml.fun_expit(ls_site[1].mat_x[:, 1]))

    for i in range(vec_beta_test.__len__()):
        ls_dml_loc[1].beta = torch.nn.Parameter(torch.tensor(vec_beta_test[i], dtype=torch.float32))
        # vec_var_test[i] = ls_dml_loc[1].var().detach().numpy()
        vec_eh2_test[i], vec_i02_test[i] = ls_dml_loc[1].var()

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)  # Create 3 subplots stacked vertically

    # First subplot for Eh2
    axes[0].scatter(vec_beta_test, vec_eh2_test, label="Eh2", color="blue")
    axes[0].set_ylabel("Eh2")
    axes[0].legend()
    axes[0].grid(True)

    # Second subplot for I02
    axes[1].scatter(vec_beta_test, vec_i02_test, label="I02", color="orange")
    axes[1].set_ylabel("I02")
    axes[1].legend()
    axes[1].grid(True)

    # Third subplot for Var (Eh2 / I02)
    axes[2].scatter(vec_beta_test, vec_eh2_test / vec_i02_test, label="Var", color="green")
    axes[2].set_xlabel("Beta Test")
    axes[2].set_ylabel("Var (Eh2 / I02)")
    axes[2].legend()
    axes[2].grid(True)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

    print(">> beta_loc:  ", vec_beta_loc.round(3), np.mean(vec_beta_loc).round(3))

    vec_weight = np.ones(S) / S

    vec_beta_fdml = vec_beta_loc.copy() 

    for r in range(n_iter): 
        beta_ini = np.sum(vec_beta_fdml * vec_weight)

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

        print(">> >> beta_fdml_", r, ": ", vec_beta_fdml.round(3), np.mean(vec_beta_fdml).round(3))







