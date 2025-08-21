## loading packages
import dml
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, getopt

import data_gen



def main(argv): 

    ## set default parameters
    K_fold = 3
    n = 1000; p = 5; S = 5; n_rnd = 2000; ver = 0; path_out = None

    beta = -2.0
    learning_rate = 0.01
    n_iter = 2


    ## prompt
    prompt_help = [
        "Usage: sim7.py [options]",
        "Options:",
        "  -h, --help          Show this help message",
        "  --ver=VERSION       Set the version"
    ]
    try:
        opts, args = getopt.getopt(
            argv, "h",
            [
                "ver=", 
                "path_out="
            ]
        )
    except getopt.GetoptError:
        print("\n    ".join(prompt_help))
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("\n    ".join(prompt_help))
            sys.exit()
        elif opt == '--ver':
            ver = int(arg)
        elif opt == '--path_out':
            path_out = arg

    ## output
    if path_out is None:
        print(
            "rnd", "Avg", "Avg_var", "M1_single", 
            ",".join(["M" + str(r + 1) for r in range(n_iter)]), 
            sep = ","
        )
    else: 
        print(
            "rnd", "avg", "var_avg", "m1_single", 
            ",".join(["M" + str(r + 1) for r in range(n_iter)]), 
            sep = ",", 
            file = open(path_out, "w")
        )

    for seed_np in range(2025, 2025+n_rnd): 

        np.random.seed(seed_np)
        torch.manual_seed(seed_np)

        vec_beta_iter = np.zeros(n_iter, dtype=np.float32)

        ls_site = [None] * S
        ls_dml_loc = [None] * S

        vec_beta_loc = np.zeros(S, dtype=np.float32)
        vec_beta_var_loc = np.zeros(S, dtype=np.float32)

        ## data generation and local estimation
        for s in range(S): 
            vec_y, vec_d, mat_x = data_gen.data_gen(n, p, beta, ver=ver)
            # mat_x = np.random.randn(n, p)
            # vec_d = mat_x[:, 0] + np.random.randn(n)
            # vec_y_prob = dml.fun_expit(0.5 + vec_d * beta + 1 * mat_x[:, 1])
            # vec_y = np.random.binomial(1, vec_y_prob) 

            ls_site[s] = dml.DataSite(vec_y, vec_d, mat_x, K_fold)
            ls_site[s].model_train_est()

            model_dml_cur, _, bl_conv_cur = dml.train_logistic_dml(ls_site[s], learning_rate)

            vec_beta_loc[s] = model_dml_cur.beta.detach().numpy()
            vec_beta_var_loc[s] = model_dml_cur.var().detach().numpy()
            ls_dml_loc[s] = model_dml_cur

        vec_weight = np.ones(S) / S

        vec_weight_var = 1.0 / vec_beta_var_loc 
        vec_weight_var /= np.sum(vec_weight_var)

        print(
            ">> beta_loc:  ", 
            vec_beta_loc.round(3), 
            np.mean(vec_beta_loc).round(3), 
            np.sum(vec_beta_loc * vec_weight_var).round(3)
        )

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

            if r == 0:
                vec_beta_m1 = vec_beta_fdml[0]

            vec_beta_iter[r] = np.mean(vec_beta_fdml)

            print(">> >> beta_fdml_", r, ": ", vec_beta_fdml.round(3), np.mean(vec_beta_fdml).round(3))

        if path_out is None:
            print(
                seed_np,
                np.mean(vec_beta_loc).round(3),
                np.sum(vec_beta_loc * vec_weight_var).round(3),
                vec_beta_m1.round(3),
                vec_beta_iter.round(3),
                sep = ",",
            )
        else: 
            print(
                seed_np,
                np.mean(vec_beta_loc).round(3),
                np.sum(vec_beta_loc * vec_weight_var).round(3),
                vec_beta_m1.round(3),
                vec_beta_iter.round(3),
                sep = ",",
                file = open(path_out, "a")
            )


if __name__ == "__main__":
    main(sys.argv[1:])





