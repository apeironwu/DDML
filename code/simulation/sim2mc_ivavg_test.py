import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
import time
import sys, getopt

import matplotlib.pyplot as plt

def fun_sigm(X): 
    out = np.exp(X) / (1 + np.exp(X)) 
    return out

## nonlinear case - return the probability of rare event
def fun_mu(X, j, alpha = 1, beta = 10, ip = 2.0):
    j = 0
    p = len(X)
    jp_step = 1  ## default
    j = np.mod(j, p) 
    jp = np.mod(j + jp_step, p) 
    out = .25 * X[jp] + fun_sigm(X[j]) 
    # out = fun_sigm(beta * (out - ip))  ## rareCov 
    return out

def fun_gamma(X, j, vec_gam = [1.0, 2.0]): 
    j = 0
    p = len(X)
    jp_step = 1  ## default
    jp = np.mod(j + jp_step, p)
    out = vec_gam[0] * X[j] + vec_gam[1] * fun_sigm(X[jp])  ## default
    return out
    # out = out * 2.0  ## double gamma
    # out = out * 0.5  ## half gamma


def main(argv): 

    ## default values
    n = 300
    K = 10
    p = 5
    n_rnp = 100
    n_rds = 10
    n_iter = 2
    n_rft = 100
    mu_alpha = 1.0
    mu_beta = 2.0 
    mu_ip = 0.0
    vgam = 7
    rnp_np_ini = 128
    path_out = None
    bl_cor_X = True
    den_est = "single"
    
    prompt_help = [
        'sim2m_rareCov_test.py',
        '--n <# sample size>',
        '--K <# of sites>',
        '--p <# of confounders>', 
        '--n_rnp <# replication>',
        '--n_rds <# data splitting>',
        '--n_iter <# iteration>',
        '--n_rft <# random forest trees>',
        '--mu_alpha <shift scale in mu>',
        '--mu_beta <max slope in mu>',
        '--mu_ip <inflection point in mu>',
        '--vgam <vgam setting>',
        '--rnp_np_ini <rnp np initial>',
        '--path_out <output path>',
        '--bl_cor_X <correlated X (True) or not (False)>',
        '--den_est <density estimation method, "single", or "ora_dou">'
    ]

    try:
        opts, args = getopt.getopt(
            argv, "h",
            [
                "n=",
                "K=",
                "p=",
                "n_rnp=",
                "n_rds=", 
                "n_iter=",
                "n_rft=",
                "mu_alpha=",
                "mu_beta=",
                "mu_ip=",
                "vgam=",
                "rnp_np_ini=",
                "path_out=",
                "bl_cor_X=",
                "den_est="
            ]
        )
    except getopt.GetoptError:
        print("\n    ".join(prompt_help))
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("\n    ".join(prompt_help))
            sys.exit()
        elif opt in ("--K"):
            K = int(arg)
        elif opt in ("--p"):
            p = int(arg)
        elif opt in ("--n"):
            n = int(arg)
        elif opt in ("--n_rnp"):
            n_rnp = int(arg)
        elif opt in ("--n_rds"):
            n_rds = int(arg)
        elif opt in ("--n_iter"):
            n_iter = int(arg)
        elif opt in ("--n_rft"):
            n_rft = int(arg)
        elif opt in ("--mu_alpha"):
            mu_alpha = float(arg)
        elif opt in ("--mu_beta"):
            mu_beta = float(arg)
        elif opt in ("--mu_ip"):
            mu_ip = float(arg)
        elif opt in ("--vgam"):
            vgam = int(arg)
        elif opt in ("--rnp_np_ini"):
            rnp_np_ini = int(arg)
        elif opt in ("--path_out"):
            path_out = arg
        elif opt in ("--bl_cor_X"):
            bl_cor_X = arg == "True"
        elif opt in ("--den_est"):
            den_est = arg

    while den_est not in ["single", "double", "ora_dou"]:
        sys.exit("den_est must be either 'single', 'double' or 'ora_dou'.")
    
    print('=' * 20, "Parameter Setting", '=' * 20)
    print('>> n: ', n)
    print('>> K: ', K)
    print('>> p: ', p)
    print('>> n_rnp: ', n_rnp)
    print('>> n_rds: ', n_rds)
    print('>> n_iter: ', n_iter)
    print('>> n_rft: ', n_rft)
    print('>> mu_alpha: ', mu_alpha)
    print('>> mu_beta: ', mu_beta)
    print('>> mu_ip: ', mu_ip)
    print('>> vgam: ', vgam)
    print('>> rnp_np_ini: ', rnp_np_ini)
    print('>> path_out: ', path_out)
    print('>> bl_cor_X: ', bl_cor_X)
    print('>> den_est: ', den_est)

    time_start = time.time()

    ## parameter setting
    vec_beta_est_iter = np.zeros(n_iter)
    vec_beta_ora_est_iter = np.zeros(n_iter)

    beta = -2 ## default

    # large variance
    psi_u = 1.0   ## default
    psi_v = 1.0   ## default

    psi_u_inv = 1 / psi_u
    psi_v_inv = 1 / psi_v

    fig = plt.figure(figsize=(12, 6))
    ax_weight, ax_var = fig.subplots(1, 2, sharex=True)

    fig_ora = plt.figure(figsize=(12, 6))
    ax_weight_ora, ax_var_ora = fig_ora.subplots(1, 2, sharex=True)

    if path_out is None: 
        print(
            "rnd_np", "rnd_ds", "Average", 
            ",".join(["M" + str(i + 1) for i in range(n_iter)]),
            ",".join(["Mo" + str(i + 1) for i in range(n_iter)]),
            sep=",", 
        )
    else:
        print(
            "rnd_np", "rnd_ds", "Average", 
            ",".join(["M" + str(i + 1) for i in range(n_iter)]),
            ",".join(["Mo" + str(i + 1) for i in range(n_iter)]),
            sep=",", 
            file = open(path_out, "w")
        )

    ## data generation 
    #### randomization
    for rnd_np in (rnp_np_ini + np.array(range(n_rnp))): 
        for rnd_ds in (128 + np.array(range(n_rds))):
            try:

                # print(rnd_np, rnd_ds)
            
                ## set `numpy` random seed
                np.random.seed(rnd_np)
                
                if bl_cor_X is False:
                    #### uncorrelated X
                    arr_X = np.random.randn(K, n, p)
                else:
                    #### correlated X
                    covmat_X = np.fromfunction(lambda i, j: np.power(0.7, np.abs(i - j)), (p, p))
                    arr_X = np.random.multivariate_normal(np.zeros(p), covmat_X, K * n).reshape(K, n, p)

                mat_U = np.random.randn(n, K) * np.sqrt(psi_u)
                psi_u_est = psi_u
                psi_u_inv_est = 1.0 / psi_u_est

                mat_V = np.random.randn(n, K) * np.sqrt(psi_v)

                # mat_U = np.random.standard_t(df_t, (n, K)) * np.sqrt(psi_u)
                # psi_u_est = np.var(mat_U)
                # psi_u_inv_est = 1.0 / psi_u_est

                # print(">> psi_u_est: ", psi_u_est.round(5))

                mat_D = np.zeros((n, K))
                mat_Y = np.zeros((n, K))

                # if vgam == 1:
                #     mat_gam = np.random.uniform(1.0, 2.0, (2, K))  ## gam setting 1
                # elif vgam == 2:
                #     mat_gam = np.random.uniform(0.0, 1.0, (2, K))  ## gam setting 2
                # elif vgam == 3:
                #     mat_gam = np.random.uniform(-2.0, 2.0, (2, K)) ## gam setting 3
                # elif vgam == 4:
                #     mat_gam = np.random.uniform(-1.0, 1.0, (2, K))       ## gam setting 4
                #     mat_gam = np.sign(mat_gam) * (np.abs(mat_gam) + 1.0) ## gam setting 4
                # elif vgam == 5:
                #     mat_gam = np.random.uniform(0.0, 0.5, (2, K))  ## gam setting 5
                # elif vgam == 6:
                #     mat_gam = np.random.uniform(-0.5, 0.5, (2, K))  ## gam setting 6
                if vgam == 7:
                    mat_gam = np.random.uniform(0.5, 1.0, (2, K))       ## gam setting 7
                # elif vgam == 8:
                #     mat_gam = np.random.uniform(-0.5, 0.5, (2, K))       ## gam setting 8
                #     mat_gam = np.sign(mat_gam) * (np.abs(mat_gam) + 0.5) ## gam setting 8
                # elif vgam == 9: 
                #     mat_gam = (np.array(range(K)) / float(K-1) - 0.5) * [[1.0], [1.0]] ## gam setting 9
                # elif vgam == 10: 
                #     mat_gam = np.random.binomial(n = 1, p = 0.5, size = (2, K)) / 2.0 ## gam setting 10
                # elif vgam == 11: 
                #     mat_gam = np.random.binomial(n = 1, p = 0.5, size = (2, K)) - 1.0 ## gam setting 11
                else:
                    exit(">> invalid vgam input")

                # print(mat_gam.round(5))

                for j in range(K): 
                    ## continuous treatment
                    mat_D[:, j] = np.array(list(map(lambda X: fun_mu(X, j, mu_alpha, mu_beta, mu_ip), arr_X[j]))) 
                    mat_D[:, j] = mat_D[:, j] + mat_V[:, j]

                    mat_Y[:, j] = mat_D[:, j] * beta \
                        + np.array(list(map(lambda X: fun_gamma(X, j, mat_gam[:, j]), arr_X[j]))) + mat_U[:, j]

                ## randomly data splitting

                ## set `random` random seed
                random.seed(int(rnd_ds))
                
                n_est = int(n / 3)

                # idx_est = np.array(list(range(n_est))) ## non-random splitting
                idx_est = np.array(list(set(random.sample(range(n), n_est)))) ## random splitting
                idx_nui = np.array(list(set(range(n)) - set(idx_est)))

                ## initial estimation of beta and estimation of nuisance parameter
                vec_beta_est_local = np.zeros(K)

                ## variance estimation
                vec_beta_var_local_ora = np.zeros(K)

                vec_beta_est_local = np.linspace(-5, 1, K)

                ## variance estimation ===========================================
                for j in range(K): 
                    # var_beta_score2 = 0.0
                    # var_beta_j0 = 0.0

                    ## estimating
                    mat_X_est = arr_X[j][idx_est, :]
                    vec_D_est = mat_D[idx_est, j]
                    vec_Y_est = mat_Y[idx_est, j]

                    vec_D_diff_ora = vec_D_est - np.array(list(map(lambda X: fun_mu(X, j, mu_alpha, mu_beta, mu_ip), mat_X_est))) 

                    ## PREVIOUS ERROR HERE
                    vec_Y_gam_diff_ora = vec_Y_est - \
                        np.array(list(map(lambda X: fun_gamma(X, j, mat_gam[:, j]), mat_X_est)))

                    var_beta_score2_ora = np.mean(
                        (
                            (vec_Y_gam_diff_ora - vec_D_est * vec_beta_est_local[j]) * vec_D_diff_ora
                        ) ** 2
                    )
                    var_beta_j0_ora = np.mean(vec_D_est * vec_D_diff_ora)

                    vec_beta_var_local_ora[j] = var_beta_score2_ora / (var_beta_j0_ora ** 2) / n_est

                vec_weight_iv_ora = 1.0 / vec_beta_var_local_ora
                vec_weight_iv_ora /= np.sum(vec_weight_iv_ora)

                ax_weight_ora.scatter(
                    vec_beta_est_local,
                    vec_weight_iv_ora
                )

                ax_var_ora.scatter(
                    vec_beta_est_local, 
                    vec_beta_var_local_ora
                )
                
                if path_out is None: 
                    print(
                        rnd_np,
                        rnd_ds,
                        np.mean(vec_beta_est_local).round(5), " | ",
                        sep=",",
                    )
                else:
                    print(
                        rnd_np,
                        rnd_ds,
                        np.mean(vec_beta_est_local).round(5),
                        sep=",",
                        file=open(path_out, "a")
                    )

            except ValueError:
                print("ValueError:  ", "rnp_", rnd_np, " rds_", rnd_ds, sep="")
                continue

    print("running time: ", time.time() - time_start, " seconds ")

    fig_ora.savefig(
        "output/sim2mc_ivavg_scatter_ldml_ivw_ora_test.pdf", 
        format="pdf"
    )

if __name__ == "__main__":
    main(sys.argv[1:])