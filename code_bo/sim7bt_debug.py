## loading packages
import dml_bt as dml
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, getopt

import data_gen



def main(argv): 

    ## set default parameters
    K_fold = 3
    # n = 500; p = 5; S = 5; n_rnd = 500; ver = 0; path_out = None
    n = 500; p = 5; S = 10; n_rnd = 1; ver = 0; path_out = None; bl_log = False; bl_conv_step = False

    beta = -1.0
    learning_rate = 1e-3
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
                "path_out=", 
                "bl_log="
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
        elif opt == '--bl_log':
            bl_log = arg.lower() == 'true'

    ## output
    if path_out is None:
        print(
            "rnd", "Avg", "IVWAvg", "M1s", 
            ",".join(["M" + str(r + 1) for r in range(n_iter)]), 
            sep = ","
        )
    else: 
        print(
            "rnd", "Avg", "IVWAvg", "M1s", 
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

        mat_gam = np.random.rand(S, 2)
        
        ## data generation and local estimation
        for s in range(S): 
            vec_y, vec_d, mat_x = data_gen.data_gen_bt_het(n, p, beta, ver=ver, gam0=mat_gam[s, 0], gam1=mat_gam[s, 1])
            # vec_y, vec_d, mat_x = data_gen.data_gen_bt(n, p, beta, ver=ver)

            if bl_log & (s == 0):
                print(" == " * 4, ">> Pr(d=1), Pr(y=1): ", np.mean(vec_d).round(3), np.mean(vec_y).round(3))

            ls_site[s] = dml.DataSite(vec_y, vec_d, mat_x, K_fold)
            ls_site[s].model_train_est()

            model_dml_cur, _, bl_conv_cur = dml.train_logistic_dml(
                ls_site[s], learning_rate, eps=1e-8, bl_log=bl_log
            )

            exit()

            vec_beta_loc[s] = model_dml_cur.beta.detach().numpy()
            vec_beta_var_loc[s] = model_dml_cur.var().detach().numpy()
            ls_dml_loc[s] = model_dml_cur

        vec_weight = np.ones(S) / S

        vec_weight_var = 1.0 / vec_beta_var_loc 
        vec_weight_var /= np.sum(vec_weight_var)

        if bl_log: 
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

            # if bl_log:
            #     print(">> scale of loc score: ", score_loc)

            ls_op_site_cen = [dml.OperaSiteCen(s, ls_site) for s in range(S)]

            ls_model_fdml = [None] * S
            vec_beta_fdml = np.zeros(S, dtype=np.float32)

            for s in range(S): 
                model_fdml_cur, _, bl_conv_cur = dml.train_logistic_fdml(
                    ls_site[s], ls_op_site_cen[s], 
                    score_loc, beta_ini, learning_rate, eps=1e-10, bl_conv_step = bl_conv_step
                )
                ls_model_fdml[s] = model_fdml_cur
                vec_beta_fdml[s] = model_fdml_cur.beta.detach().numpy()

            if r == 0:
                beta_m1s = vec_beta_fdml[0]

            vec_beta_iter[r] = np.mean(vec_beta_fdml)

            if bl_log:
                print(
                    ">> >> beta_fdml_", r, ": ", 
                    vec_beta_fdml.round(3), np.mean(vec_beta_fdml).round(3)
                )

        if path_out is None:
            print(
                seed_np,
                np.mean(vec_beta_loc).round(3),
                np.sum(vec_beta_loc * vec_weight_var).round(3),
                beta_m1s.round(3),
                ",".join(str(beta_est.round(3)) for beta_est in vec_beta_iter.round(3)),
                sep = ",",
            )
        else: 
            print(
                seed_np,
                np.mean(vec_beta_loc).round(5),
                np.sum(vec_beta_loc * vec_weight_var).round(5),
                beta_m1s.round(5),
                ",".join(str(beta_est.round(5)) for beta_est in vec_beta_iter.round(5)),
                sep = ",",
                file = open(path_out, "a")
            )


if __name__ == "__main__":
    main(sys.argv[1:])





