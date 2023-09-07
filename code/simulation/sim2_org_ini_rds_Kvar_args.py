import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
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
    out = out * 3
    return out

def fun_gamma(X, j): 
    p = len(X)
    jp = np.mod(j + 2, p)
    out = X[j] + .25 * fun_sigm(X[jp])
    out = out * 12
    return out


def main(argv): 

    ## default values
    n = 100
    K = 5
    p = K + 3
    n_rnp = 100
    n_rds = 10
    n_iter = 2
    n_rft = 100
    path_out = None
    bl_cor_X = False
    
    prompt_help = [
        'sim2_org_ini_rds_args.py',
        '--n <# sample size>',
        '--K <# of sites>',
        '--p <# of confounders>', 
        '--n_rnp <# replication>',
        '--n_rds <# data splitting>',
        '--n_iter <# iteration>',
        '--n_rft <# random forest trees>',
        '--path_out <output path>',
        '--bl_cor_X <correlated X (True) or not (False)>' 
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
        elif opt in ("--path_out"):
            path_out = arg
        elif opt in ("--bl_cor_X"):
            bl_cor_X = arg == "True"
    
    print('=' * 20, "Parameter Setting", '=' * 20)
    print('>> n: ', n)
    print('>> K: ', K)
    print('>> p: ', p)
    print('>> n_rnp: ', n_rnp)
    print('>> n_rds: ', n_rds)
    print('>> n_iter: ', n_iter)
    print('>> n_rft: ', n_rft)
    print('>> path_out: ', path_out)
    print('>> bl_cor_X: ', bl_cor_X)

    time_start = time.time()

    ## parameter setting
    vec_beta_est_iter = np.zeros(n_iter)

    beta = 2

    # large variance
    psi_u = 36
    psi_v = 9

    psi_u_inv = 1 / psi_u

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

    ## data generation 
    #### randomization
    for rnd_np in (128 + np.array(range(n_rnp))): 
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
                mat_V = np.random.randn(n, K) * np.sqrt(psi_v)

                mat_D = np.zeros((n, K))
                mat_Y = np.zeros((n, K))

                for j in range(K): 
                    mat_D[:, j] = np.array(list(map(lambda X: fun_mu(X, j), arr_X[j]))) + mat_V[:, j]
                    mat_Y[:, j] = mat_D[:, j] * beta \
                        + np.array(list(map(lambda X: fun_gamma(X, j), arr_X[j]))) + mat_U[:, j]

                ## randomly data splitting

                ## set `random` random seed
                random.seed(int(rnd_ds))
                
                n_est = int(n / 2)

                # idx_est = np.array(list(range(n_est))) ## non-random splitting
                idx_est = np.array(list(set(random.sample(range(n), n_est)))) ## random splitting
                idx_nui = np.array(list(set(range(n)) - set(idx_est)))

                i_iter = 0

                ## initial estimation of beta and estimation of nuisance parameter
                vec_beta_est_local = np.zeros(K)

                list_gamma_est = list()
                list_mu_est = list()

                for j in range(K):
                    ## training ML model
                    mat_X_nui = arr_X[j][idx_nui, :]
                    vec_D_nui = mat_D[idx_nui, j]
                    vec_Y_nui = mat_Y[idx_nui, j]
                    
                    ## estimating
                    mat_X_est = arr_X[j][idx_est, :]
                    vec_D_est = mat_D[idx_est, j]
                    vec_Y_est = mat_Y[idx_est, j]

                    model_mu = RandomForestRegressor(n_estimators=n_rft)
                    model_mu.fit(mat_X_nui, vec_D_nui)
                    
                    ## naive estimation of beta
                    # beta_tilde = np.mean(vec_Y_nui.T * vec_D_nui) / np.mean(vec_D_nui.T * vec_D_nui)
                    # print("beta_tilde: ", beta_tilde)
                    
                    ## true value of beta
                    # beta_tilde = 2

                    ## estimation of beta based on partialling out score function
                    model_xi = RandomForestRegressor(n_estimators=n_rft)
                    model_xi.fit(mat_X_nui, vec_Y_nui)
                    
                    vec_D_diff = vec_D_est - model_mu.predict(mat_X_est)
                    vec_Y_diff = vec_Y_est - model_xi.predict(mat_X_est)
                    
                    beta_est_local = np.mean(vec_Y_diff * vec_D_diff) / np.mean(vec_D_diff * vec_D_diff)

                    model_gamma = RandomForestRegressor(n_estimators=n_rft)
                    model_gamma.fit(mat_X_nui, vec_Y_nui - vec_D_nui * beta_est_local)

                    list_mu_est.append(model_mu)
                    list_gamma_est.append(model_gamma)

                    vec_beta_est_local[j] = beta_est_local

                beta_est_ini = np.mean(vec_beta_est_local)

                ## statistics from other sites
                vec_s = np.zeros(n_est)
                vec_S_est = np.zeros(K)

                # vec_s_ora = np.zeros(n_est)
                # vec_S_ora_est = np.zeros(K)

                for j in range(K): 
                    ## estimating
                    mat_X_est = arr_X[j][idx_est, :]
                    vec_D_est = mat_D[idx_est, j]
                    vec_Y_est = mat_Y[idx_est, j]

                    vec_D_diff = vec_D_est - list_mu_est[j].predict(mat_X_est)
                    vec_Y_diff = vec_Y_est - list_gamma_est[j].predict(mat_X_est)
                    
                    vec_s = (vec_Y_diff - vec_D_est * beta_est_ini) * vec_D_diff
                    
                    vec_S_est[j] = np.mean(vec_s)

                    # vec_D_ora_diff = vec_D_est - np.array(list(map(lambda X: fun_mu(X, j), mat_X_est))) 
                    # vec_Y_ora_diff = vec_Y_est - np.array(list(map(lambda X: fun_gamma(X, j), mat_X_est)))
                    
                    # vec_s_ora = (vec_Y_ora_diff - vec_D_est * beta_est_ini) * vec_D_ora_diff
                    # vec_S_ora_est[j] = np.mean(vec_s_ora)


                S = np.mean(vec_S_est)

                # S_ora = np.mean(vec_S_ora_est)



                ## operation in the central site
                vec_beta_est_cen = np.zeros(K)
                # vec_beta_ora_est_cen = np.zeros(K)

                for j_cen in range(K):
                # for j_cen in [0]:
                    vec_Y_cen = mat_Y[idx_est, j_cen]
                    vec_D_cen = mat_D[idx_est, j_cen]
                    mat_X_cen = arr_X[j_cen][idx_est,]

                    mat_U_slope = np.zeros((n_est, K))

                    vec_D_cen_diff = vec_D_cen - list_mu_est[j_cen].predict(mat_X_cen)
                    vec_Y_cen_diff = vec_Y_cen - list_gamma_est[j_cen].predict(mat_X_cen)

                    vec_U_cen_est = vec_Y_cen_diff - vec_D_cen * beta_est_ini

                    for j in range(K):

                        vec_D_loc_diff = vec_D_cen - list_mu_est[j].predict(mat_X_cen)
                        vec_Y_loc_diff = vec_Y_cen - list_gamma_est[j].predict(mat_X_cen)
                        
                        vec_U_loc_est = vec_Y_loc_diff - vec_D_cen * beta_est_ini
                        
                        mat_U_slope[:, j] = np.power(vec_U_loc_est, 2) - np.power(vec_U_cen_est, 2)
                        mat_U_slope[:, j] = np.exp(-.5 * psi_u_inv * mat_U_slope[:, j]) ## density ratio

                        # print("density ratio: ", np.median(mat_U_slope[:, j]))
                        
                        mat_U_slope[:, j] = mat_U_slope[:, j] * vec_D_cen * vec_D_loc_diff

                        # print(" " * 40, "U_slope:       ", np.median(mat_U_slope[:, j]))


                    U_slope = np.mean(mat_U_slope)

                    beta_est_cen = beta_est_ini + S / U_slope
                    vec_beta_est_cen[j_cen] = beta_est_cen

                    # beta_ora_est_cen = beta_est_ini + S_ora / U_slope
                    # vec_beta_ora_est_cen[j_cen] = beta_ora_est_cen

                ## final estimation
                beta_est = np.mean(vec_beta_est_cen)

                # beta_est_ora = np.mean(vec_beta_ora_est_cen)

                vec_beta_est_iter[i_iter] = beta_est
                i_iter += 1


                ## iteration
                while i_iter < n_iter:
                    
                    ## updating estimation of nuisance parameter
                    for j in range(K):
                        ## training ML model
                        mat_X_nui = arr_X[j][idx_nui, :]
                        vec_D_nui = mat_D[idx_nui, j]
                        vec_Y_nui = mat_Y[idx_nui, j]
                        
                        ## updating estimation of gamma
                        model_gamma = RandomForestRegressor(n_estimators=n_rft)
                        model_gamma.fit(mat_X_nui, vec_Y_nui - vec_D_nui * beta_est)

                        list_gamma_est[j] = model_gamma

                    beta_est_ini = beta_est

                    ## statistics from other sites
                    vec_s = np.zeros(n_est)
                    vec_S_est = np.zeros(K)

                    # vec_s_ora = np.zeros(n_est)
                    # vec_S_ora_est = np.zeros(K)

                    for j in range(K): 
                        ## estimating
                        mat_X_est = arr_X[j][idx_est, :]
                        vec_D_est = mat_D[idx_est, j]
                        vec_Y_est = mat_Y[idx_est, j]

                        vec_D_diff = vec_D_est - list_mu_est[j].predict(mat_X_est)
                        vec_Y_diff = vec_Y_est - list_gamma_est[j].predict(mat_X_est)
                        
                        vec_s = (vec_Y_diff - vec_D_est * beta_est_ini) * vec_D_diff
                        
                        vec_S_est[j] = np.mean(vec_s)

                        # vec_D_ora_diff = vec_D_est - np.array(list(map(lambda X: fun_mu(X, j), mat_X_est))) 
                        # vec_Y_ora_diff = vec_Y_est - np.array(list(map(lambda X: fun_gamma(X, j), mat_X_est)))
                        
                        # vec_s_ora = (vec_Y_ora_diff - vec_D_est * beta_est_ini) * vec_D_ora_diff
                        # vec_S_ora_est[j] = np.mean(vec_s_ora)


                    S = np.mean(vec_S_est)

                    # S_ora = np.mean(vec_S_ora_est)



                    ## operation in the central site
                    vec_beta_est_cen = np.zeros(K)
                    # vec_beta_ora_est_cen = np.zeros(K)

                    for j_cen in range(K):
                    # for j_cen in [0]:
                        vec_Y_cen = mat_Y[idx_est, j_cen]
                        vec_D_cen = mat_D[idx_est, j_cen]
                        mat_X_cen = arr_X[j_cen][idx_est,]

                        mat_U_slope = np.zeros((n_est, K))

                        vec_D_cen_diff = vec_D_cen - list_mu_est[j_cen].predict(mat_X_cen)
                        vec_Y_cen_diff = vec_Y_cen - list_gamma_est[j_cen].predict(mat_X_cen)

                        vec_U_cen_est = vec_Y_cen_diff - vec_D_cen * beta_est_ini

                        for j in range(K):

                            vec_D_loc_diff = vec_D_cen - list_mu_est[j].predict(mat_X_cen)
                            vec_Y_loc_diff = vec_Y_cen - list_gamma_est[j].predict(mat_X_cen)
                            
                            vec_U_loc_est = vec_Y_loc_diff - vec_D_cen * beta_est_ini
                            
                            mat_U_slope[:, j] = np.power(vec_U_loc_est, 2) - np.power(vec_U_cen_est, 2)
                            mat_U_slope[:, j] = np.exp(-.5 * psi_u_inv * mat_U_slope[:, j]) ## density ratio

                            # print("density ratio: ", np.median(mat_U_slope[:, j]))
                            
                            mat_U_slope[:, j] = mat_U_slope[:, j] * vec_D_cen * vec_D_loc_diff

                            # print(" " * 40, "U_slope:       ", np.median(mat_U_slope[:, j]))

                        U_slope = np.mean(mat_U_slope)

                        beta_est_cen = beta_est_ini + S / U_slope
                        vec_beta_est_cen[j_cen] = beta_est_cen
                    
                    beta_est = np.mean(vec_beta_est_cen)
                    vec_beta_est_iter[i_iter] = beta_est
                    i_iter += 1

                # print("np rnd:       ", rnd_np)
                # print("ds rnd:       ", rnd_ds)
                # print("Average:      ", np.mean(vec_beta_est_local))
                # print("vec_beta_est: ", vec_beta_est_iter)

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