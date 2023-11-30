import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import time
import sys, getopt

def fun_sigm(X): 
    out = np.exp(X) / (1 + np.exp(X)) 
    return out

## nonlinear case
def fun_mu(X, j):
    p = len(X)
    jp = np.mod(j + 2, p)
    out = fun_sigm(X[j]) + .25 * X[jp]
    return out

def fun_gamma(X, j): 
    p = len(X)
    jp = np.mod(j + 2, p)
    out = X[j] + .25 * fun_sigm(X[jp])
    return out

def fun_quad(vec_U, vec_eps, mat_cov_inv): 
    mat_U_eps = np.vstack((vec_U, vec_eps))
    out = np.diag(mat_U_eps.T @ mat_cov_inv @ mat_U_eps)

    ## alternative realization
    # mat_U_eps = np.vstack((vec_U, vec_eps))
    # out = np.apply_along_axis(lambda row: row @ mat_cov_inv @ row.T, 1, mat_U_eps.T)

    return out

def main(argv):
    
    ## default values
    n = 1000
    K = 5
    p = 20
    n_rnp_gen = 100
    n_rnp_ds = 10
    n_iter = 2
    path_out = None
    bl_cor_X = True
    den_est = "joint"
    psi_d = 1
    
    prompt_help = [
        'sim3_args.py',
        '--K <# of sites>',
        '--n <# sample size>',
        '--p <dim of covariates>',
        '--n_rnp_gen <# replication>',
        '--n_rnp_ds <# data splitting>',
        '--n_iter <# iteration>',
        '--path_out <output path>',
        '--bl_cor_X <correlated X (True) or not (False)>', 
        '--den_est <joint or single>',
        '--psi_d <variance of D>'
    ]

    try:
        opts, args = getopt.getopt(
            argv, "h",
            [
                "n=",
                "K=",
                "p=",
                "n_rnp_gen=",
                "n_rnp_ds=", 
                "n_iter=",
                "path_out=",
                "bl_cor_X=", 
                "den_est=", 
                "psi_d="
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
        elif opt in ("--p"):
            p = int(arg)
        elif opt in ("--n_rnp_gen"):
            n_rnp_gen = int(arg)
        elif opt in ("--n_rnp_ds"):
            n_rnp_ds = int(arg)
        elif opt in ("--n_iter"):
            n_iter = int(arg)
        elif opt in ("--path_out"):
            path_out = arg
        elif opt in ("--bl_cor_X"):
            bl_cor_X = arg == "True"
        elif opt in ("--den_est"):
            den_est = arg
        elif opt in ("--psi_d"):
            psi_d = float(arg)

    while den_est not in ["joint", "single"]:
        sys.exit("den_est must be either 'joint' or 'single'.")
    
    print('=' * 20, "Parameter Setting", '=' * 20)
    print('>> K: ', K)
    print('>> n: ', n)
    print('>> p: ', p)
    print('>> n_rnp_gen: ', n_rnp_gen)
    print('>> n_rnp_ds: ', n_rnp_ds)
    print('>> n_iter: ', n_iter)
    print('>> path_out: ', path_out)
    print('>> bl_cor_X: ', bl_cor_X)
    print('>> den_est: ', den_est)
    print('>> psi_d: ', psi_d)

    time_start = time.time()

    vec_beta_est_iter = np.zeros(n_iter)

    ## parameter setting
    beta = 0.5

    ## variance
    psi_u = 1
    psi_v = 1
    
    # random forest setting 
    rf_gamma = RandomForestRegressor(n_estimators=200)
    rf_mu = RandomForestRegressor(n_estimators=200)

    psi_u_inv = 1 / psi_u
    psi_v_inv = 1 / psi_v

    rho_U_eps = 0.75 ## default value
    cormat_U_eps = np.fromfunction(lambda i, j: np.power(rho_U_eps, np.abs(i - j)), (2, 2))
    diagmat_psi_ud_sqrt = np.sqrt(np.diag([psi_u, psi_d]))
    covmat_U_eps = diagmat_psi_ud_sqrt @ cormat_U_eps @ diagmat_psi_ud_sqrt
    covmat_U_eps_inv = np.linalg.inv(covmat_U_eps)

    if path_out is None: 
        print(
            "rnd_gen", "rnd_ds", "Average", 
            ",".join(["M" + str(i + 1) for i in range(n_iter)]),
            sep=",", 
        )
    else:
        print(
            "rnd_gen", "rnd_ds", "Average", 
            ",".join(["M" + str(i + 1) for i in range(n_iter)]),
            sep=",", 
            file = open(path_out, "w")
        )

    #### randomization
    for rnd_gen in (2023 + np.array(range(n_rnp_gen))): 
        for rnd_ds in (2023 + np.array(range(n_rnp_ds))):

            try: 
                ## set `numpy` random seed
                np.random.seed(rnd_gen)

                if not bl_cor_X:
                    #### uncorrelated X
                    arr_X = np.random.randn(K, n, p)
                elif bl_cor_X:
                    #### correlated X
                    covmat_X = np.fromfunction(lambda i, j: np.power(0.7, np.abs(i - j)), (p, p))
                    arr_X = np.random.multivariate_normal(np.zeros(p), covmat_X, K * n).reshape(K, n, p)

                #### correlated U and eps
                arr_U_eps = np.random.multivariate_normal(np.zeros(2), cormat_U_eps, K * n).T.reshape(2, n, K)

                mat_U = arr_U_eps[0] * np.sqrt(psi_u)
                mat_eps = arr_U_eps[1] * np.sqrt(psi_d)
                
                mat_V = np.random.randn(n, K) * np.sqrt(psi_v)

                mat_Z = np.zeros((n, K))
                mat_D = np.zeros((n, K))
                mat_Y = np.zeros((n, K))

                for j in range(K): 
                    mat_Z[:, j] = np.array(list(map(lambda X: fun_mu(X, j), arr_X[j]))) + mat_V[:, j]
                    mat_D[:, j] = mat_Z[:, j] + mat_eps[:, j]
                    mat_Y[:, j] = mat_D[:, j] * beta \
                        + np.array(list(map(lambda X: fun_gamma(X, j), arr_X[j]))) + mat_U[:, j]

                ## randomly data splitting
                K_fold = 2

                #### set `random` random seed
                np.random.seed(rnd_ds)
                idx_K_fold = np.random.choice(range(K_fold), n, replace=True)

                i_iter = 0

                list_gamma_est = list()
                list_mu_est = list()

                ## initial estimation of beta and estimation of nuisance parameter
                mat_beta_est_local = np.zeros((K, K_fold))

                for splt in range(K_fold):
                    idx_est = np.where(idx_K_fold == splt)[0]
                    idx_nui = np.where(idx_K_fold != splt)[0]

                    n_est = len(idx_est)

                    for j in range(K):
                        ## data for training ML model
                        mat_X_nui = arr_X[j][idx_nui, :]
                        vec_Z_nui = mat_Z[idx_nui, j]
                        vec_D_nui = mat_D[idx_nui, j]
                        vec_Y_nui = mat_Y[idx_nui, j]
                        
                        ## estimating
                        mat_X_est = arr_X[j][idx_est, :]
                        vec_Z_est = mat_Z[idx_est, j]
                        vec_D_est = mat_D[idx_est, j]
                        vec_Y_est = mat_Y[idx_est, j]

                        model_mu = clone(rf_mu)
                        model_mu.fit(mat_X_nui, vec_Z_nui)
                        
                        ## estimation of beta based on partialling out score function
                        model_zeta = clone(rf_mu)
                        model_zeta.fit(mat_X_nui, vec_D_nui)
                        
                        model_xi = clone(rf_gamma)
                        model_xi.fit(mat_X_nui, vec_Y_nui)
                        
                        vec_Z_diff = vec_Z_est - model_mu.predict(mat_X_est)
                        vec_D_diff = vec_D_est - model_zeta.predict(mat_X_est)
                        vec_Y_diff = vec_Y_est - model_xi.predict(mat_X_est)
                        
                        beta_est_local = np.mean(vec_Y_diff * vec_Z_diff) / np.mean(vec_D_diff * vec_Z_diff)

                        model_gamma = clone(rf_gamma)
                        model_gamma.fit(mat_X_nui, vec_Y_nui - vec_D_nui * beta_est_local)

                        list_mu_est.append(model_mu)
                        list_gamma_est.append(model_gamma)

                        mat_beta_est_local[j, splt] = beta_est_local

                beta_est_ini = np.mean(mat_beta_est_local)

                mat_beta_est_cen = np.zeros((K, K_fold))

                for splt in range(K_fold):
                    idx_est = np.where(idx_K_fold == splt)[0]
                    idx_nui = np.where(idx_K_fold != splt)[0]
                    
                    n_est = len(idx_est)
                    
                    ## statistics from other sites
                    vec_s = np.zeros(n_est)
                    vec_S_est = np.zeros(K)

                    for j in range(K): 
                        idx_ls = j + splt * K

                        ## estimating
                        mat_X_est = arr_X[j][idx_est, :]
                        vec_Z_est = mat_Z[idx_est, j]
                        vec_D_est = mat_D[idx_est, j]
                        vec_Y_est = mat_Y[idx_est, j]

                        vec_Z_diff = vec_Z_est - list_mu_est[idx_ls].predict(mat_X_est)
                        vec_Y_diff = vec_Y_est - list_gamma_est[idx_ls].predict(mat_X_est)
                        
                        vec_s = (vec_Y_diff - vec_D_est * beta_est_ini) * vec_Z_diff
                        
                        vec_S_est[j] = np.mean(vec_s)

                    S = np.mean(vec_S_est)

                    ## operation in the central site
                    for j_cen in range(K):
                        idx_ls_cen = j_cen + splt * K

                        vec_Y_cen = mat_Y[idx_est, j_cen]
                        vec_Z_cen = mat_Z[idx_est, j_cen]
                        vec_D_cen = mat_D[idx_est, j_cen]
                        mat_X_cen = arr_X[j_cen][idx_est,]

                        mat_U_slope = np.zeros((n_est, K))

                        vec_Z_cen_diff = vec_Z_cen - list_mu_est[idx_ls_cen].predict(mat_X_cen)
                        vec_Y_cen_diff = vec_Y_cen - list_gamma_est[idx_ls_cen].predict(mat_X_cen)

                        vec_U_cen_est = vec_Y_cen_diff - vec_D_cen * beta_est_ini

                        for j in range(K):
                            idx_ls = j + splt * K

                            vec_Z_loc_diff = vec_Z_cen - list_mu_est[idx_ls].predict(mat_X_cen)
                            vec_Y_loc_diff = vec_Y_cen - list_gamma_est[idx_ls].predict(mat_X_cen)
                            
                            vec_U_loc_est = vec_Y_loc_diff - vec_D_cen * beta_est_ini
                            
                            if den_est == "single":
                                ## single density estimation
                                mat_U_slope[:, j] = np.power(vec_U_loc_est, 2) - np.power(vec_U_cen_est, 2)
                                mat_U_slope[:, j] = np.exp(-.5 * psi_u_inv * mat_U_slope[:, j]) ## density ratio
                            elif den_est == "joint":
                                ## joint density estimation
                                vec_D_res = vec_D_cen - vec_Z_cen

                                mat_U_slope[:, j] = fun_quad(vec_U_loc_est, vec_D_res, covmat_U_eps_inv) - \
                                    fun_quad(vec_U_cen_est, vec_D_res, covmat_U_eps_inv)
                                mat_U_slope[:, j] = mat_U_slope[:, j] + psi_v_inv * \
                                    (np.power(vec_Z_loc_diff, 2) - np.power(vec_Z_cen_diff, 2))
                                mat_U_slope[:, j] = np.exp(-.5 * mat_U_slope[:, j]) ## density ratio

                            # print("density ratio: ", np.median(mat_U_slope[:, j]))
                            
                            mat_U_slope[:, j] = mat_U_slope[:, j] * vec_D_cen * vec_Z_loc_diff

                            # print(" " * 40, "U_slope:       ", np.median(mat_U_slope[:, j]))

                        U_slope = np.mean(mat_U_slope)

                        beta_est_cen = beta_est_ini + S / U_slope
                        mat_beta_est_cen[j_cen, splt] = beta_est_cen

                ## final estimation
                beta_est = np.mean(mat_beta_est_cen)

                vec_beta_est_iter[i_iter] = beta_est
                i_iter += 1

                ## iteration
                while i_iter < n_iter:
                    
                    mat_beta_est_cen = np.zeros((K, K_fold))
                    beta_est_ini = beta_est

                    for splt in range(K_fold):
                        idx_est = np.where(idx_K_fold == splt)[0]
                        idx_nui = np.where(idx_K_fold != splt)[0]
                        
                        n_est = len(idx_est)
                    
                        ## updating estimation of nuisance parameter
                        for j in range(K):
                            idx_ls = j + splt * K

                            ## training ML model
                            mat_X_nui = arr_X[j][idx_nui, :]
                            vec_Z_nui = mat_Z[idx_nui, j]
                            vec_D_nui = mat_D[idx_nui, j]
                            vec_Y_nui = mat_Y[idx_nui, j]
                            
                            ## updating estimation of gamma
                            model_gamma = clone(rf_gamma)
                            model_gamma.fit(mat_X_nui, vec_Y_nui - vec_D_nui * beta_est)

                            list_gamma_est[idx_ls] = model_gamma

                        ## statistics from other sites
                        vec_s = np.zeros(n_est)
                        vec_S_est = np.zeros(K)

                        for j in range(K): 
                            idx_ls = j + splt * K

                            ## estimating
                            mat_X_est = arr_X[j][idx_est, :]
                            vec_Z_est = mat_Z[idx_est, j]
                            vec_D_est = mat_D[idx_est, j]
                            vec_Y_est = mat_Y[idx_est, j]

                            vec_Z_diff = vec_Z_est - list_mu_est[idx_ls].predict(mat_X_est)
                            vec_Y_diff = vec_Y_est - list_gamma_est[idx_ls].predict(mat_X_est)
                            
                            vec_s = (vec_Y_diff - vec_D_est * beta_est_ini) * vec_Z_diff
                            
                            vec_S_est[j] = np.mean(vec_s)

                        S = np.mean(vec_S_est)

                        ## operation in the central site
                        for j_cen in range(K):
                            idx_ls_cen = j_cen + splt * K
                            
                            vec_Y_cen = mat_Y[idx_est, j_cen]
                            vec_Z_cen = mat_Z[idx_est, j_cen]
                            vec_D_cen = mat_D[idx_est, j_cen]
                            mat_X_cen = arr_X[j_cen][idx_est,]

                            mat_U_slope = np.zeros((n_est, K))

                            vec_Z_cen_diff = vec_Z_cen - list_mu_est[idx_ls_cen].predict(mat_X_cen)
                            vec_Y_cen_diff = vec_Y_cen - list_gamma_est[idx_ls_cen].predict(mat_X_cen)

                            vec_U_cen_est = vec_Y_cen_diff - vec_D_cen * beta_est_ini

                            for j in range(K):
                                idx_ls = j + splt * K

                                vec_Z_loc_diff = vec_Z_cen - list_mu_est[idx_ls].predict(mat_X_cen)
                                vec_Y_loc_diff = vec_Y_cen - list_gamma_est[idx_ls].predict(mat_X_cen)
                                
                                vec_U_loc_est = vec_Y_loc_diff - vec_D_cen * beta_est_ini
                                
                                if den_est == "single":
                                    ## single density estimation
                                    mat_U_slope[:, j] = np.power(vec_U_loc_est, 2) - np.power(vec_U_cen_est, 2)
                                    mat_U_slope[:, j] = np.exp(-.5 * psi_u_inv * mat_U_slope[:, j]) ## density ratio
                                elif den_est == "joint":
                                    ## joint density estimation
                                    vec_D_res = vec_D_cen - vec_Z_cen

                                    mat_U_slope[:, j] = fun_quad(vec_U_loc_est, vec_D_res, covmat_U_eps_inv) - \
                                        fun_quad(vec_U_cen_est, vec_D_res, covmat_U_eps_inv)
                                    mat_U_slope[:, j] = mat_U_slope[:, j] + psi_v_inv * \
                                        (np.power(vec_Z_loc_diff, 2) - np.power(vec_Z_cen_diff, 2))

                                    mat_U_slope[:, j] = np.exp(-.5 * mat_U_slope[:, j]) ## density ratio

                                # print("density ratio: ", np.median(mat_U_slope[:, j]))
                                
                                mat_U_slope[:, j] = mat_U_slope[:, j] * vec_D_cen * vec_Z_loc_diff

                                # print(" " * 40, "U_slope:       ", np.median(mat_U_slope[:, j]))

                            U_slope = np.mean(mat_U_slope)

                            beta_est_cen = beta_est_ini + S / U_slope
                            mat_beta_est_cen[j_cen, splt] = beta_est_cen
                        
                    beta_est = np.mean(mat_beta_est_cen)
                    vec_beta_est_iter[i_iter] = beta_est
                    i_iter += 1

                if path_out is None: 
                    print(
                        rnd_gen,
                        rnd_ds,
                        np.mean(mat_beta_est_local),
                        ','.join(str(b) for b in vec_beta_est_iter),
                        sep=",",
                    )
                else:
                    print(
                        rnd_gen,
                        rnd_ds,
                        np.mean(mat_beta_est_local),
                        ','.join(str(b) for b in vec_beta_est_iter),
                        sep=",",
                        file=open(path_out, "a")
                    )
                
            except ValueError:
                print("ValueError:  ", "rnp_", rnd_gen, " rds_", rnd_ds, sep="")
                continue

    print("running time: ", time.time() - time_start, " seconds ")

if __name__ == "__main__":
    main(sys.argv[1:])

