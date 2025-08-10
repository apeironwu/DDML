import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import time
import sys, getopt

def fun_sigm(X): 
    out = np.exp(X) / (1 + np.exp(X)) 
    return out

## nonlinear case
def fun_mu(X, j, alpha = 10, beta = 10, ip = 2.0):
    j = 1
    p = len(X)
    jp_step = 2  ## default
    j = np.mod(j, p) 
    jp = np.mod(j + jp_step, p) 
    # out = alpha * fun_sigm(beta * (X[j] - ip)) + X[jp]   ## rareCov 
    # out = alpha * (out > (alpha / 2.0)).astype(float)    ## binary shift
    # out = alpha * fun_sigm(beta * (X[j] - ip))  ## rareCov 
    out = fun_sigm(beta * (X[j] - ip))
    out = np.random.binomial(1, out)
    # out = (out < 0.5).astype(int)
    return out

def fun_gamma(X, j): 
    j = 1
    p = len(X)
    jp_step = 1  ## default
    jp = np.mod(j + jp_step, p)
    out = X[j] + 2.0 * fun_sigm(X[jp])
    return out


def main(argv): 

    ## default values
    n = 1000
    K = 5
    p = 10
    n_rnp = 100
    n_rds = 10
    n_iter = 2
    n_rft = 100
    mu_alpha = 10.0
    mu_beta = 2.0 
    mu_ip = 0.0
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
        elif opt in ("--rnp_np_ini"):
            rnp_np_ini = int(arg)
        elif opt in ("--path_out"):
            path_out = arg
        elif opt in ("--bl_cor_X"):
            bl_cor_X = arg == "True"
        elif opt in ("--den_est"):
            den_est = arg

    while den_est not in ["single", "ora_dou"]:
        sys.exit("den_est must be either 'single' or 'ora_dou'.")
    
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
    print('>> rnp_np_ini: ', rnp_np_ini)
    print('>> path_out: ', path_out)
    print('>> bl_cor_X: ', bl_cor_X)
    print('>> den_est: ', den_est)

    time_start = time.time()

    ## parameter setting
    vec_beta_est_iter = np.zeros(n_iter)
    vec_beta_ora_est_iter = np.zeros(n_iter)

    # beta = 0.5 ## default
    # beta = 0.2 ## default
    # beta = 0.1 ## default
    beta = mu_alpha

    # large variance
    # psi_u = 1 ## default
    psi_u = 4 
    psi_v = 1 ## default
    # psi_v = 4
    # psi_v = .25
    # psi_v = .1

    psi_u_inv = 1 / psi_u
    psi_v_inv = 1 / psi_v

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
                # mat_V = np.random.randn(n, K) * np.sqrt(psi_v)

                # print("sd mat_v: ", np.std(mat_V).round(5))

                mat_D = np.zeros((n, K))
                mat_Y = np.zeros((n, K))

                for j in range(K): 
                    mat_D[:, j] = np.array(list(map(lambda X: fun_mu(X, j, mu_alpha, mu_beta, mu_ip), arr_X[j]))) 

                    mat_Y[:, j] = mat_D[:, j] * beta \
                        + np.array(list(map(lambda X: fun_gamma(X, j), arr_X[j]))) + mat_U[:, j]
                
                print(">> Pr(D=1): ", np.mean(mat_D == 1).round(5))

                ## randomly data splitting

                ## set `random` random seed
                random.seed(int(rnd_ds))
                
                n_est = int(n / 2)

                # idx_est = np.array(list(range(n_est))) ## non-random splitting
                idx_est = np.array(list(set(random.sample(range(n), n_est)))) ## random splitting
                idx_nui = np.array(list(set(range(n)) - set(idx_est)))

                i_iter = 0

                ## initial estimation of beta and estimation of nuisance parameter
                vec_beta_est_po_local = np.zeros(K)
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


                    model_mu = RandomForestClassifier(n_estimators=n_rft)
                    model_mu.fit(mat_X_nui, vec_D_nui)

                    # vec_mu_proba_est = model_mu.predict_proba(mat_X_est)[:, 1]
                    # print(
                    #     np.mean(vec_mu_proba_est - vec_D_est), 
                    #     np.var(vec_mu_proba_est - vec_D_est).round(5),
                    # )
                    # exit("Test")

                    # print(np.mean(vec_D_est - model_mu.predict_proba(mat_X_est)))
                    
                    ## estimation of beta based on partialling out score function
                    model_xi = RandomForestRegressor(n_estimators=n_rft)
                    model_xi.fit(mat_X_nui, vec_Y_nui)
                    
                    vec_D_diff = vec_D_est - (1 - model_mu.predict_proba(mat_X_est)[:, 0])
                    vec_Y_xi_diff = vec_Y_est - model_xi.predict(mat_X_est)
                    
                    ## first partialling-out estimation
                    beta_est_local = np.mean(vec_Y_xi_diff * vec_D_diff) / np.mean(vec_D_diff * vec_D_diff)
                    vec_beta_est_po_local[j] = beta_est_local

                    for i_local in range(2): 
                        model_gamma = RandomForestRegressor(n_estimators=n_rft)
                        model_gamma.fit(mat_X_nui, vec_Y_nui - vec_D_nui * beta_est_local)

                        vec_Y_gam_diff = vec_Y_est - model_gamma.predict(mat_X_est)

                        ## second orthogonal estimation
                        beta_est_local = np.mean(vec_Y_gam_diff * vec_D_diff) / np.mean(vec_D_est * vec_D_diff)

                    model_gamma = RandomForestRegressor(n_estimators=n_rft)
                    model_gamma.fit(mat_X_nui, vec_Y_nui - vec_D_nui * beta_est_local)

                    list_mu_est.append(model_mu)
                    list_gamma_est.append(model_gamma)

                    vec_beta_est_local[j] = beta_est_local

                # print(">> vec_beta_est_po_local: ", vec_beta_est_po_local)
                # print(">> vec_beta_est_local: ", vec_beta_est_local)

                beta_est_ini = np.mean(vec_beta_est_local)
                # beta_est_ini = np.median(vec_beta_est_local)

                # exit("Test")

                ## statistics from other sites
                vec_s = np.zeros(n_est)
                vec_S_est = np.zeros(K)

                vec_s_ora = np.zeros(n_est)
                vec_S_ora_est = np.zeros(K)

                for j in range(K): 
                    ## estimating
                    mat_X_est = arr_X[j][idx_est, :]
                    vec_D_est = mat_D[idx_est, j]
                    vec_Y_est = mat_Y[idx_est, j]

                    vec_D_diff = vec_D_est - (1 - list_mu_est[j].predict_proba(mat_X_est)[:, 0])
                    vec_Y_diff = vec_Y_est - list_gamma_est[j].predict(mat_X_est)
                    
                    vec_s = (vec_Y_diff - vec_D_est * beta_est_ini) * vec_D_diff
                    
                    vec_S_est[j] = np.mean(vec_s)

                    vec_D_ora_diff = vec_D_est - np.array(list(map(lambda X: fun_mu(X, j, mu_alpha, mu_beta, mu_ip), mat_X_est))) 
                    vec_Y_ora_diff = vec_Y_est - np.array(list(map(lambda X: fun_gamma(X, j), mat_X_est)))
                    
                    vec_s_ora = (vec_Y_ora_diff - vec_D_est * beta_est_ini) * vec_D_ora_diff
                    vec_S_ora_est[j] = np.mean(vec_s_ora)

                # print(">> vec_S_est: ", vec_S_est)
                # print(">> vec_S_ora_est: ", vec_S_ora_est)

                S = np.mean(vec_S_est)
                S_ora = np.mean(vec_S_ora_est)

                ## operation in the central site
                vec_beta_est_cen = np.zeros(K)
                vec_beta_ora_est_cen = np.zeros(K)

                for j_cen in range(K):
                    vec_Y_cen = mat_Y[idx_est, j_cen]
                    vec_D_cen = mat_D[idx_est, j_cen]
                    mat_X_cen = arr_X[j_cen][idx_est,]

                    mat_U_slope = np.zeros((n_est, K))

                    if den_est == "single":
                        # vec_D_cen_diff = vec_D_cen - list_mu_est[j_cen].predict(mat_X_cen)
                        # print(">>>> var_rf_v_est", np.var(vec_D_cen_diff).round(5))
                        vec_Y_cen_diff = vec_Y_cen - list_gamma_est[j_cen].predict(mat_X_cen)

                        vec_U_cen_est = vec_Y_cen_diff - vec_D_cen * beta_est_ini
                        # print("   " * 10, ">>>> var_rf_u_est", np.var(vec_U_cen_est).round(5))

                        for j in range(K):

                            vec_D_loc_diff = vec_D_cen - (1 - list_mu_est[j].predict_proba(mat_X_cen)[:, 0])
                            vec_Y_loc_diff = vec_Y_cen - list_gamma_est[j].predict(mat_X_cen)
                            
                            vec_U_loc_est = vec_Y_loc_diff - vec_D_cen * beta_est_ini
                            
                            mat_U_slope[:, j] = np.power(vec_U_loc_est, 2) - np.power(vec_U_cen_est, 2)
                            mat_U_slope[:, j] = np.exp(-.5 * psi_u_inv * mat_U_slope[:, j]) ## density ratio

                            # print("density ratio: ", np.median(mat_U_slope[:, j]))
                            
                            mat_U_slope[:, j] = mat_U_slope[:, j] * vec_D_cen * vec_D_loc_diff

                    elif den_est == "ora_dou":
                        ## oracle central estimation
                        vec_D_cen_diff_ora = vec_D_cen - np.array(list(map(lambda X: fun_mu(X, j_cen, mu_alpha, mu_beta, mu_ip), mat_X_cen)))
                        # print(">>>> var_rf_v_est", np.var(vec_D_cen_diff_ora))
                        vec_Y_cen_diff_ora = vec_Y_cen - np.array(list(map(lambda X: fun_gamma(X, j_cen), mat_X_cen)))

                        vec_U_cen_est_ora = vec_Y_cen_diff_ora - vec_D_cen * beta_est_ini
                        # print(">>>> " * 4, "var_rf_u_est", np.var(vec_U_cen_est_ora))
                        
                        for j in range(K):

                            ## oracle local estimation
                            vec_D_loc_diff_ora = vec_D_cen - np.array(list(map(lambda X: fun_mu(X, j, mu_alpha, mu_beta, mu_ip), mat_X_cen)))
                            vec_Y_loc_diff_ora = vec_Y_cen - np.array(list(map(lambda X: fun_gamma(X, j), mat_X_cen)))

                            vec_U_loc_est_ora = vec_Y_loc_diff_ora - vec_D_cen * beta_est_ini
                            
                            ## oracle double density estimation
                            mat_U_slope[:, j] = psi_u_inv * \
                                (np.power(vec_U_loc_est_ora, 2) - np.power(vec_U_cen_est_ora, 2))
                            mat_U_slope[:, j] = mat_U_slope[:, j] + psi_v_inv * \
                                (np.power(vec_D_loc_diff_ora, 2) - np.power(vec_D_cen_diff_ora, 2))
                            mat_U_slope[:, j] = np.exp(-.5 * mat_U_slope[:, j]) 
                            
                            # print("density ratio: ", np.median(mat_U_slope[:, j]))
                            
                            mat_U_slope[:, j] = mat_U_slope[:, j] * vec_D_cen * vec_D_loc_diff_ora

                    U_slope = np.mean(mat_U_slope)

                    beta_est_cen = beta_est_ini + S / U_slope
                    vec_beta_est_cen[j_cen] = beta_est_cen

                    beta_ora_est_cen = beta_est_ini + S_ora / U_slope
                    vec_beta_ora_est_cen[j_cen] = beta_ora_est_cen

                ## final estimation
                beta_est = np.mean(vec_beta_est_cen)
                beta_est_ora = np.mean(vec_beta_ora_est_cen)

                vec_beta_est_iter[i_iter] = beta_est
                vec_beta_ora_est_iter[i_iter] = beta_est_ora

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

                    vec_s_ora = np.zeros(n_est)
                    vec_S_ora_est = np.zeros(K)

                    for j in range(K): 
                        ## estimating
                        mat_X_est = arr_X[j][idx_est, :]
                        vec_D_est = mat_D[idx_est, j]
                        vec_Y_est = mat_Y[idx_est, j]

                        vec_D_diff = vec_D_est - (1 - list_mu_est[j].predict_proba(mat_X_est)[:, 0])
                        vec_Y_diff = vec_Y_est - list_gamma_est[j].predict(mat_X_est)
                        
                        vec_s = (vec_Y_diff - vec_D_est * beta_est_ini) * vec_D_diff
                        vec_S_est[j] = np.mean(vec_s)

                        vec_D_ora_diff = vec_D_est - np.array(list(map(lambda X: fun_mu(X, j, mu_alpha, mu_beta, mu_ip), mat_X_est))) 
                        vec_Y_ora_diff = vec_Y_est - np.array(list(map(lambda X: fun_gamma(X, j), mat_X_est)))

                        vec_s_ora = (vec_Y_ora_diff - vec_D_est * beta_est_ini) * vec_D_ora_diff
                        vec_S_ora_est[j] = np.mean(vec_s_ora)

                        # print(
                        #     ">> corr_Y_diff: ", 
                        #     np.corrcoef(
                        #         vec_Y_diff - vec_D_est * beta_est_ini, 
                        #         vec_Y_ora_diff - vec_D_est * beta_est_ini
                        #     ), 
                        #     " | >> corr_D_diff: ",
                        #     np.corrcoef(
                        #         vec_D_diff, 
                        #         vec_D_ora_diff
                        #     )
                        # )

                    S = np.mean(vec_S_est)
                    S_ora = np.mean(vec_S_ora_est)

                    ## operation in the central site
                    vec_beta_est_cen = np.zeros(K)
                    vec_beta_ora_est_cen = np.zeros(K)

                    for j_cen in range(K):
                    # for j_cen in [0]:
                        vec_Y_cen = mat_Y[idx_est, j_cen]
                        vec_D_cen = mat_D[idx_est, j_cen]
                        mat_X_cen = arr_X[j_cen][idx_est,]

                        mat_U_slope = np.zeros((n_est, K))

                        if den_est == "single":
                            vec_Y_cen_diff = vec_Y_cen - list_gamma_est[j_cen].predict(mat_X_cen)

                            vec_U_cen_est = vec_Y_cen_diff - vec_D_cen * beta_est_ini

                            for j in range(K):

                                vec_D_loc_diff = vec_D_cen - (1 - list_mu_est[j].predict_proba(mat_X_cen)[:, 0])
                                vec_Y_loc_diff = vec_Y_cen - list_gamma_est[j].predict(mat_X_cen)
                                
                                vec_U_loc_est = vec_Y_loc_diff - vec_D_cen * beta_est_ini
                                
                                mat_U_slope[:, j] = np.power(vec_U_loc_est, 2) - np.power(vec_U_cen_est, 2)
                                mat_U_slope[:, j] = np.exp(-.5 * psi_u_inv * mat_U_slope[:, j]) ## density ratio

                                # print("density ratio: ", np.median(mat_U_slope[:, j]))
                                
                                mat_U_slope[:, j] = mat_U_slope[:, j] * vec_D_cen * vec_D_loc_diff

                        elif den_est == "ora_dou":
                            ## oracle central estimation
                            vec_D_cen_diff_ora = vec_D_cen - np.array(list(map(lambda X: fun_mu(X, j_cen, mu_alpha, mu_beta, mu_ip), mat_X_cen)))
                            vec_Y_cen_diff_ora = vec_Y_cen - np.array(list(map(lambda X: fun_gamma(X, j_cen), mat_X_cen)))

                            vec_U_cen_est_ora = vec_Y_cen_diff_ora - vec_D_cen * beta_est_ini
                            
                            for j in range(K):
                                ## oracle local estimation
                                vec_D_loc_diff_ora = vec_D_cen - np.array(list(map(lambda X: fun_mu(X, j, mu_alpha, mu_beta, mu_ip), mat_X_cen)))
                                vec_Y_loc_diff_ora = vec_Y_cen - np.array(list(map(lambda X: fun_gamma(X, j), mat_X_cen)))

                                vec_U_loc_est_ora = vec_Y_loc_diff_ora - vec_D_cen * beta_est_ini
                                
                                ## oracle double density estimation
                                mat_U_slope[:, j] = psi_u_inv * \
                                    (np.power(vec_U_loc_est_ora, 2) - np.power(vec_U_cen_est_ora, 2))
                                mat_U_slope[:, j] = mat_U_slope[:, j] + psi_v_inv * \
                                    (np.power(vec_D_loc_diff_ora, 2) - np.power(vec_D_cen_diff_ora, 2))
                                mat_U_slope[:, j] = np.exp(-.5 * mat_U_slope[:, j]) 
                                
                                # print("density ratio: ", np.median(mat_U_slope[:, j]))
                                
                                mat_U_slope[:, j] = mat_U_slope[:, j] * vec_D_cen * vec_D_loc_diff_ora

                        U_slope = np.mean(mat_U_slope)

                        beta_est_cen = beta_est_ini + S / U_slope
                        vec_beta_est_cen[j_cen] = beta_est_cen

                        beta_ora_est_cen = beta_est_ini + S_ora / U_slope
                        vec_beta_ora_est_cen[j_cen] = beta_ora_est_cen
                    
                    ## final estimation
                    beta_est = np.mean(vec_beta_est_cen)
                    vec_beta_est_iter[i_iter] = beta_est

                    beta_est_ora = np.mean(vec_beta_ora_est_cen)
                    vec_beta_ora_est_iter[i_iter] = beta_est_ora

                    i_iter += 1

                # print("np rnd:       ", rnd_np)
                # print("ds rnd:       ", rnd_ds)
                # print("Average:      ", np.mean(vec_beta_est_local))
                # print("vec_beta_est: ", vec_beta_est_iter)

                if path_out is None: 
                    print(
                        rnd_np,
                        rnd_ds,
                        np.mean(vec_beta_est_local).round(5), " | ",
                        ','.join(str(b.round(5)) for b in vec_beta_est_iter), " | ",
                        ','.join(str(b.round(5)) for b in vec_beta_ora_est_iter),
                        sep=",",
                    )
                else:
                    print(
                        rnd_np,
                        rnd_ds,
                        np.mean(vec_beta_est_local),
                        ','.join(str(b.round(5)) for b in vec_beta_est_iter),
                        ','.join(str(b.round(5)) for b in vec_beta_ora_est_iter),
                        sep=",",
                        file=open(path_out, "a")
                    )
            
            except ValueError:
                print("ValueError:  ", "rnp_", rnd_np, " rds_", rnd_ds, sep="")
                continue

    print("running time: ", time.time() - time_start, " seconds ")

if __name__ == "__main__":
    main(sys.argv[1:])