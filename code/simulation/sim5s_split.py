import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import clone
import time
import sys, getopt

def fun_sigm(X): 
    out = np.exp(X) / (1 + np.exp(X)) 
    return out

def fun_tanh(X): 
    out = (np.exp(X) - 1) / (np.exp(X) + 1)
    return out

## nonlinear case
def fun_gam0(X): 
    out = X[0] + fun_tanh(X[1])
    return out

def fun_gam1(X): 
    out = X[2] + fun_tanh(X[3])
    return out

def fun_gamma(vec_D_X): 
    D = vec_D_X[0]
    X = vec_D_X[1:]

    if D == 0:
        out = fun_gam0(X)
    elif D == 1:
        out = fun_gam1(X)
    else:
        raise ValueError("D should be either 0 or 1")

    return out

def fun_mu(X):
    out = X[4] + .25 * fun_tanh(X[5])
    return out

def main(argv): 

    ## default values
    n = 1000
    K = 5
    n_rnp = 100
    n_rds = 10
    n_rft = 300
    n_iter = 1
    p = 10
    psi_d = .1
    path_out = None
    bl_cor_X = False
    
    prompt_help = [
        'sim5_args.py',
        '--K <# of sites>',
        '--n <# sample size>',
        '--n_rnp <# replication>',
        '--n_rds <# data splitting>',
        '--n_rft <# trees in RF>',
        '--n_iter <# iteration>',
        '--p <# covariates>',
        '--psi_d <var of error in treatment>',
        '--path_out <output path>',
        '--bl_cor_X <correlated X (True) or not (False)>' 
    ]

    try:
        opts, args = getopt.getopt(
            argv, "h",
            [
                "n=",
                "K=",
                "n_rnp=",
                "n_rds=", 
                "n_rft=",
                "n_iter=",
                "p=",
                "psi_d=",
                "path_out=",
                "bl_cor_X="
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
        elif opt in ("--n"):
            n = int(arg)
        elif opt in ("--n_rnp"):
            n_rnp = int(arg)
        elif opt in ("--n_rds"):
            n_rds = int(arg)
        elif opt in ("--n_rft"):
            n_rft = int(arg)
        elif opt in ("--n_iter"):
            n_iter = int(arg)
        elif opt in ("--p"):
            p = int(arg)
        elif opt in ("--psi_d"):
            psi_d = float(arg)
        elif opt in ("--path_out"):
            path_out = arg
        elif opt in ("--bl_cor_X"):
            bl_cor_X = arg == "True"
    
    print('=' * 20, "Parameter Setting", '=' * 20)
    print('>> K: ', K)
    print('>> n: ', n)
    print('>> n_rnp: ', n_rnp)
    print('>> n_rds: ', n_rds)
    print('>> n_rft: ', n_rds)
    print('>> n_iter: ', n_iter)
    print('>> p: ', p)
    print('>> psi_d: ', psi_d)
    print('>> path_out: ', path_out)
    print('>> bl_cor_X: ', bl_cor_X)

    time_start = time.time()

    ## parameter setting
    vec_beta_est_iter = np.zeros(n_iter)

    beta = 0.5

    # large variance
    psi_u = 1
    # psi_u_inv = 1 / psi_u


    rho_U_delta = 0.75 ## default value
    cormat_U_delta = np.fromfunction(lambda i, j: np.power(rho_U_delta, np.abs(i - j)), (2, 2))
    diagmat_psi_ud_sqrt = np.sqrt(np.diag([psi_u, psi_d]))
    covmat_U_delta = diagmat_psi_ud_sqrt @ cormat_U_delta @ diagmat_psi_ud_sqrt
    covmat_U_delta_inv = np.linalg.inv(covmat_U_delta)

    rf_reg = RandomForestRegressor(n_estimators=n_rft)
    rf_clf = RandomForestClassifier(n_estimators=n_rft)

    if path_out is None: 
        print(
            "rnd_np", "rnd_ds", "Average", 
            ",".join(["M" + str(i + 1) for i in range(n_iter)]),
            sep=",", 
        )
    else:
        print(
            "rnd_np", "rnd_ds", "Average", 
            ",".join(["M" + str(i + 1) for i in range(n_iter)]),
            sep=",", 
            file = open(path_out, "w")
        )

    #### randomization
    for rnd_np in (2023 + np.array(range(n_rnp))): 
        for rnd_ds in (2023 + np.array(range(n_rds))):

            try:
                # print(rnd_np, rnd_ds)

                ## data generation ----

                ## set `numpy` random seed
                np.random.seed(rnd_np)
                
                if bl_cor_X is False:
                    #### uncorrelated X
                    arr_X = np.random.randn(K, n, p)
                else:
                    #### correlated X
                    covmat_X = np.fromfunction(lambda i, j: np.power(0.7, np.abs(i - j)), (p, p))
                    arr_X = np.random.multivariate_normal(np.zeros(p), covmat_X, K * n).reshape(K, n, p)

                # mat_U = np.random.randn(n, K) * np.sqrt(psi_u)
                # mat_V = np.zeros((n, K))
                # mat_d = np.zeros((n, K)) ## error of treatment

                ## correlated 
                arr_U_delta = np.random.multivariate_normal(np.zeros(2), cormat_U_delta, K * n).T.reshape(2, n, K)

                mat_U = arr_U_delta[0] * np.sqrt(psi_u)
                mat_delta = arr_U_delta[1] * np.sqrt(psi_d)

                mat_Z = np.zeros((n, K))
                mat_D = np.zeros((n, K))
                mat_Y = np.zeros((n, K))

                for j in range(K): 
                    mat_Z[:, j] = np.random.rand(n) < fun_sigm(
                        np.array(list(map(lambda X: fun_mu(X), arr_X[j])))
                    )

                    # mat_V[:, j] = mat_Z[:, j] - np.array(list(map(lambda X: 0.5 + fun_mu(X), arr_X[j])))

                    mat_D[:, j] = -0.5 + mat_Z[:, j] + mat_delta[:, j]
                    mat_D[:, j] = mat_D[:, j] > 0

                    # mat_d[:, j] = mat_D[:, j] - mat_Z[:, j]
                    
                    mat_Y[:, j] = mat_D[:, j] * beta \
                        + np.array(
                            list(map(fun_gamma, np.column_stack((mat_D[:, j], arr_X[j]))))
                        ) + mat_U[:, j]

                ## randomly data splitting

                ## set `random` random seed
                random.seed(int(rnd_ds))
                
                n_est = int(n / 2)

                # idx_est = np.array(list(range(n_est))) ## non-random splitting
                idx_est = np.array(list(set(random.sample(range(n), n_est)))) ## random splitting
                idx_nui = np.array(list(set(range(n)) - set(idx_est)))

                ## initial estimation of beta and estimation of nuisance parameter
                vec_beta_est_local = np.zeros(K)

                list_mu_est = list()
                list_nu0_est = list()
                list_nu1_est = list()
                list_gam0_est = list()
                list_gam1_est = list()

                for j in range(K):
                    ## training part
                    mat_X_nui = arr_X[j][idx_nui, :]
                    vec_Z_nui = mat_Z[idx_nui, j]
                    vec_D_nui = mat_D[idx_nui, j]
                    vec_Y_nui = mat_Y[idx_nui, j]

                    vec_Z0_nui = vec_Z_nui == 0
                    vec_Z1_nui = vec_Z_nui == 1

                    mat_X_Z0_nui = mat_X_nui[vec_Z0_nui, :]
                    mat_X_Z1_nui = mat_X_nui[vec_Z1_nui, :]

                    ## estimating part
                    mat_X_est = arr_X[j][idx_est, :]
                    vec_Z_est = mat_Z[idx_est, j]
                    vec_D_est = mat_D[idx_est, j]
                    vec_Y_est = mat_Y[idx_est, j]


                    ## estimating nuisance parameter
                    model_mu = clone(rf_clf)
                    model_mu.fit(mat_X_nui, vec_Z_nui)

                    
                    model_nu0 = clone(rf_clf)
                    model_nu0.fit(mat_X_Z0_nui, vec_D_nui[vec_Z0_nui])
                    model_nu1 = clone(rf_clf)
                    model_nu1.fit(mat_X_Z1_nui, vec_D_nui[vec_Z1_nui])

                    model_gam0 = clone(rf_reg)
                    model_gam0.fit(mat_X_Z0_nui, vec_Y_nui[vec_Z0_nui])
                    model_gam1 = clone(rf_reg)
                    model_gam1.fit(mat_X_Z1_nui, vec_Y_nui[vec_Z1_nui])

                    ## estimating beta
                    vec_mu_est = model_mu.predict_proba(mat_X_est)[:, 1]

                    vec_nu0_est = model_nu0.predict_proba(mat_X_est)[:, 1]
                    vec_nu1_est = model_nu1.predict_proba(mat_X_est)[:, 1]

                    vec_gam0_est = model_gam0.predict(mat_X_est)
                    vec_gam1_est = model_gam1.predict(mat_X_est)

                    # print(">> diff: ", np.mean(vec_gam1_est - vec_gam0_est))

                    beta_est_local_itcp = np.mean(
                        vec_gam1_est - vec_gam0_est + \
                            vec_Z_est * (vec_Y_est - vec_gam1_est) / vec_mu_est - \
                                (1 - vec_Z_est) * (vec_Y_est - vec_gam0_est) / (1 - vec_mu_est)
                    )

                    beta_est_local_slp = np.mean(
                        vec_nu1_est - vec_nu0_est + \
                            vec_Z_est * (vec_D_est - vec_nu1_est) / vec_mu_est - \
                                (1 - vec_Z_est) * (vec_D_est - vec_nu0_est) / (1 - vec_mu_est)
                    )

                    beta_est_local = beta_est_local_itcp / beta_est_local_slp

                    list_mu_est.append(model_mu)
                    list_nu0_est.append(model_nu0)
                    list_nu1_est.append(model_nu1)
                    list_gam0_est.append(model_gam0)
                    list_gam1_est.append(model_gam1)

                    vec_beta_est_local[j] = beta_est_local
                
                # print(">> mean: ", np.mean(vec_beta_est_local))
                # print(">> var:  ", np.sqrt(np.var(vec_beta_est_local)))

                beta_est_ini = np.mean(vec_beta_est_local)

                # print("AVE: ", beta_est_ini)

                i_iter = 0

                while i_iter < n_iter:

                    ## statistics from other sites
                    vec_s = np.zeros(n_est)
                    vec_s_itcp = np.zeros(n_est)
                    vec_s_slp = np.zeros(n_est)

                    vec_S_est = np.zeros(K)
                    vec_S_slp_est = np.zeros(K)

                    vec_mu_est = np.zeros(n_est)
                    vec_nu0_est = np.zeros(n_est)
                    vec_nu1_est = np.zeros(n_est)
                    vec_gam0_est = np.zeros(n_est)
                    vec_gam1_est = np.zeros(n_est)

                    vec_beta_est_cen = np.zeros(K)
                    

                    for j in range(K): 
                        ## estimating
                        mat_X_est = arr_X[j][idx_est, :]
                        vec_Z_est = mat_Z[idx_est, j]
                        vec_D_est = mat_D[idx_est, j]
                        vec_Y_est = mat_Y[idx_est, j]

                        model_mu = list_mu_est[j]
                        model_nu0 = list_nu0_est[j]
                        model_nu1 = list_nu1_est[j]
                        model_gam0 = list_gam0_est[j]
                        model_gam1 = list_gam1_est[j]

                        vec_mu_est = model_mu.predict_proba(mat_X_est)[:, 1]

                        vec_nu0_est = model_nu0.predict_proba(mat_X_est)[:, 1]
                        vec_nu1_est = model_nu1.predict_proba(mat_X_est)[:, 1]

                        vec_gam0_est = model_gam0.predict(mat_X_est)
                        vec_gam1_est = model_gam1.predict(mat_X_est)

                        vec_s_slp = vec_nu1_est - vec_nu0_est + \
                            vec_Z_est * (vec_D_est - vec_nu1_est) / vec_mu_est - \
                                (1 - vec_Z_est) * (vec_D_est - vec_nu0_est) / (1 - vec_mu_est)
                        
                        vec_s_itcp = vec_gam1_est - vec_gam0_est + \
                            vec_Z_est * (vec_Y_est - vec_gam1_est) / vec_mu_est - \
                                (1 - vec_Z_est) * (vec_Y_est - vec_gam0_est) / (1 - vec_mu_est)

                        vec_s = vec_s_itcp - vec_s_slp * beta_est_ini

                        vec_S_est[j] = np.mean(vec_s)
                        vec_S_slp_est[j] = np.mean(vec_s_slp)

                    S = np.mean(vec_S_est)
                    
                    vec_beta_est_cen = beta_est_ini + S / vec_S_slp_est

                    beta_est = np.mean(vec_beta_est_cen)

                    # print("M1:  ", beta_est)
                    
                    vec_beta_est_iter[i_iter] = beta_est

                    i_iter += 1

                if path_out is None: 
                    print(
                        rnd_np,
                        rnd_ds,
                        np.mean(vec_beta_est_local),
                        ','.join(str(b) for b in vec_beta_est_iter),
                        sep=",",
                    )
                else:
                    print(
                        rnd_np,
                        rnd_ds,
                        np.mean(vec_beta_est_local),
                        ','.join(str(b) for b in vec_beta_est_iter),
                        sep=",",
                        file=open(path_out, "a")
                    )
            
            except ValueError:
                print("ValueError:  ", "rnp_", rnd_np, " rds_", rnd_ds, sep="")
                continue

    print("running time: ", time.time() - time_start, " seconds ")

if __name__ == "__main__":
    main(sys.argv[1:])