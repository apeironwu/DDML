import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
import time
import sys, getopt

import torch
import scipy.stats
from den_est_rkhs_rff import *
from den_est_rkhs_rff_train import *

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
    out = fun_sigm(beta * (out - ip))  ## rareCov 
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
    n = 1000
    K = 5
    p = 2
    n_rnp = 1000
    n_rds = 10
    n_iter = 1
    n_rft = 100
    mu_alpha = 1.0
    mu_beta = 2.0 
    mu_ip = 2.0
    vgam = 7
    type_cs = 1
    rnp_np_ini = 128
    path_out = None
    bl_cor_X = True
    den_est = "double"
    
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
        '--typs_cs <typr of covariate shift>',
        '--rnp_np_ini <rnp np initial>',
        '--path_out <output path>',
        '--bl_cor_X <correlated X (True) or not (False)>',
        '--den_est <density estimation method, "single", "double" or "ora_dou">'
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
                "type_cs=",
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
        elif opt in ("--type_cs"):
            type_cs = int(arg)
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
    print('>> type_cs: ', type_cs)
    print('>> rnp_np_ini: ', rnp_np_ini)
    print('>> path_out: ', path_out)
    print('>> bl_cor_X: ', bl_cor_X)
    print('>> den_est: ', den_est)

    time_start = time.time()

    ## parameter setting
    vec_beta_est_iter = np.zeros(n_iter)
    vec_beta_csp_est_iter = np.zeros(n_iter)
    vec_beta_cs_est_iter = np.zeros(n_iter)
    vec_beta_ora_est_iter = np.zeros(n_iter)

    beta = -2 ## default

    # large variance
    psi_u = 4.0   ## default
    psi_v = 1.0   ## default

    psi_u_inv = 1 / psi_u
    psi_v_inv = 1 / psi_v



    if path_out is None: 
        print(
            "rnd_np", "rnd_ds", "Average", 
            ",".join(["M" + str(i + 1) for i in range(n_iter)]),
            ",".join(["Mp" + str(i + 1) for i in range(n_iter)]),
            ",".join(["Mnp" + str(i + 1) for i in range(n_iter)]),
            ",".join(["Mo" + str(i + 1) for i in range(n_iter)]),
            sep=",", 
        )
    else:
        print(
            "rnd_np", "rnd_ds", "Average", 
            ",".join(["M" + str(i + 1) for i in range(n_iter)]),
            ",".join(["Mp" + str(i + 1) for i in range(n_iter)]),
            ",".join(["Mnp" + str(i + 1) for i in range(n_iter)]),
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
                
                mat_X_shift = np.zeros((K, p))

                if type_cs == 1:
                    ## default shift type
                    mat_X_shift = np.random.randn(K, p) * 0.5
                elif type_cs == 2:
                    mat_X_shift[:, 0] = np.linspace(-1, 1, K) 
                    mat_X_shift[:, 1] = np.linspace(-1, 1, K) 
                elif type_cs == 3:
                    mat_X_shift = np.random.randn(K, p) * 2.0
                elif type_cs == 4:
                    mat_X_shift = np.random.randn(K, p) * 1.0
                    

                mat_X_shift[:, range(2, p)] = 0.0 

                # print(">> mat_X_shift: \n", mat_X_shift.round(3))

                if bl_cor_X is False:
                    #### uncorrelated X
                    arr_X = np.random.randn(K, n, p)
                else:
                    #### correlated X
                    covmat_X = np.fromfunction(lambda i, j: np.power(0.7, np.abs(i - j)), (p, p))
                    arr_X_cen = np.random.multivariate_normal(np.zeros(p), covmat_X, K * n).reshape(K, n, p)
                    arr_X = arr_X_cen + mat_X_shift[:, np.newaxis, :]
                
                # print(">> arr_X.shape: ", arr_X.shape)
                # print(">> mean_X: ", np.mean(arr_X, axis=(1)).shape)
                # print(">> mean_X: \n", np.mean(arr_X, axis=(1)).round(3))

                # exit("test")
                
                mat_U = np.random.randn(n, K) * np.sqrt(psi_u)
                psi_u_est = psi_u
                psi_u_inv_est = 1.0 / psi_u_est

                # mat_U = np.random.standard_t(df_t, (n, K)) * np.sqrt(psi_u)
                # psi_u_est = np.var(mat_U)
                # psi_u_inv_est = 1.0 / psi_u_est

                # print(">> psi_u_est: ", psi_u_est.round(5))

                mat_D = np.zeros((n, K))
                mat_Y = np.zeros((n, K))

                if vgam == 7:
                    mat_gam = np.random.uniform(0.5, 1.0, (2, K))       ## gam setting 7
                else:
                    exit(">> invalid vgam input")

                # print(mat_gam.round(5))

                for j in range(K): 
                    vec_prob_d = np.array(list(map(lambda X: fun_mu(X, j, mu_alpha, mu_beta, mu_ip), arr_X_cen[j]))) 

                    mat_D[:, j] = np.random.binomial(n = 1, p = vec_prob_d) ## rare event

                    # print(mat_D[:, j].round(5))
                    # print(">> mean_v: ", np.mean(mat_D[:, j] - vec_prob_d).round(5))

                    mat_Y[:, j] = mat_D[:, j] * beta \
                        + np.array(list(map(lambda X: fun_gamma(X, j, mat_gam[:, j]), arr_X[j]))) + mat_U[:, j]

                # print(">> Pr(D=1): ", np.mean(mat_D, axis=0).round(5))

                # continue
                ## randomly data splitting

                ## set `random` random seed
                random.seed(int(rnd_ds))
                torch.manual_seed(int(rnd_ds))
                
                n_est = int(n / 3)

                # idx_est = np.array(list(range(n_est))) ## non-random splitting
                idx_est = np.array(list(set(random.sample(range(n), n_est)))) ## random splitting
                idx_nui = np.array(list(set(range(n)) - set(idx_est)))

                i_iter = 0

                ## initial estimation of beta and estimation of nuisance parameter
                vec_beta_est_po_local = np.zeros(K)
                vec_beta_est_local = np.zeros(K)

                list_mu_est = list()

                list_gamma_est = list()
                list_gamma_cs_est = [None] * K
                list_gamma_ora_est = [None] * K

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
                    
                    ## estimation of beta based on partialling out score function
                    model_xi = RandomForestRegressor(n_estimators=n_rft)
                    model_xi.fit(mat_X_nui, vec_Y_nui)
                    
                    vec_D_diff = vec_D_est - model_mu.predict(mat_X_est)
                    vec_Y_xi_diff = vec_Y_est - model_xi.predict(mat_X_est)
                    
                    ## first partialling-out estimation
                    beta_est_local = np.mean(vec_Y_xi_diff * vec_D_diff) / np.mean(vec_D_diff * vec_D_diff)
                    vec_beta_est_po_local[j] = beta_est_local

                    for _ in range(1): 
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

                # beta_est_ini = np.mean(vec_beta_est_local)
                beta_est_ini = np.median(vec_beta_est_local)

                # exit("Test")

                ## (par / non-par) density estimation
                mat_mu_X_est = np.zeros((K, p))
                arr_sig_X_est = np.zeros((K, p, p))

                list_den_est = list()

                for j in range(K):
                    mat_X_nui = arr_X[j][idx_nui, :]
                    # mat_X_nui = arr_X[j]
                    ts_X_nui = torch.from_numpy(mat_X_nui).float()
                    # print(">> mean_X_nui: ", np.mean(mat_X_nui, axis=0).round(3))

                    model_den_est,_,_ = den_est_rkhs_rff_train(
                        train_x = ts_X_nui, 
                        D = 200, 
                        learning_rate = 1e-2, 
                        reg = 2 * n
                    )

                    list_den_est.append(model_den_est)

                    mat_mu_X_est[j, :] = mat_X_nui.mean(axis=0)
                    arr_sig_X_est[j] = np.cov(mat_X_nui, rowvar=False) + np.eye(p) * 1e-3
                    # print(
                    #     ">> mat_mu_X_truth | mat_mu_X_est: ", 
                    #     mat_X_shift[j, :].round(3), "|", mat_mu_X_est[j, :].round(3)
                    # )
                    # print(
                    #     ">> sig_X_est: \n", 
                    #     arr_sig_X_est[j].round(3)
                    # )

               


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

                    # vec_D_ora_diff = vec_D_est - np.array(list(map(lambda X: fun_mu(X, j, mu_alpha, mu_beta, mu_ip), mat_X_est))) 
                    # vec_Y_ora_diff = vec_Y_est - np.array(list(map(lambda X: fun_gamma(X, j, mat_gam[:, j]), mat_X_est)))

                    # # print("corr_D_pred|ora: ", np.corrcoef(vec_D_diff, vec_D_ora_diff)[0, 1].round(5))
                    # # print("corr_Y_pred|ora: ", np.corrcoef(vec_Y_diff, vec_Y_ora_diff)[0, 1].round(5))
                    
                    # vec_s_ora = (vec_Y_ora_diff - vec_D_est * beta_est_ini) * vec_D_ora_diff
                    # vec_S_ora_est[j] = np.mean(vec_s_ora)

                # print(">> vec_S_est: ", vec_S_est)
                # print(">> vec_S_ora_est: ", vec_S_ora_est)

                S = np.mean(vec_S_est)
                # S_ora = np.mean(vec_S_ora_est)

                ## operation in the central site
                vec_beta_est_cen = np.zeros(K)
                vec_beta_cs_est_cen = np.zeros(K)
                vec_beta_csp_est_cen = np.zeros(K)
                vec_beta_ora_est_cen = np.zeros(K)

                for j_cen in range(K):
                    vec_Y_cen = mat_Y[idx_est, j_cen]
                    vec_D_cen = mat_D[idx_est, j_cen]
                    mat_X_cen = arr_X[j_cen][idx_est,]
                    ts_X_cen = torch.from_numpy(mat_X_cen).float()

                    ## density estimation - center
                    den_est_x_cen = list_den_est[j_cen].pred(ts_X_cen, bl_shift = True).detach().numpy()
                    den_estp_x_cen = scipy.stats.multivariate_normal.pdf(
                        mat_X_cen, mean = mat_mu_X_est[j_cen, :], cov = arr_sig_X_est[j_cen]
                    )
                    den_ora_x_cen = scipy.stats.multivariate_normal.pdf(
                        mat_X_cen, mean = mat_X_shift[j_cen, :], cov = covmat_X
                    )

                    # print(">> corr_den_ora_est: ", np.corrcoef(den_ora_x_cen, den_est_x_cen)[0, 1].round(5))

                    mat_U_slope = np.zeros((n_est, K))
                    mat_U_slope_cs = np.zeros((n_est, K))
                    mat_U_slope_csp = np.zeros((n_est, K))
                    mat_U_slope_ora = np.zeros((n_est, K))

                    if den_est == "single":
                        exit(">> single density estimation is not supported in this version")

                    elif den_est == "double":
                        vec_D_cen_est = list_mu_est[j_cen].predict(mat_X_cen)
                        vec_Y_cen_diff = vec_Y_cen - list_gamma_est[j_cen].predict(mat_X_cen)

                        vec_U_cen_est = vec_Y_cen_diff - vec_D_cen * beta_est_ini
                        
                        for j in range(K):

                            vec_D_loc_est = list_mu_est[j].predict(mat_X_cen)
                            vec_D_loc_diff = vec_D_cen - vec_D_loc_est
                            vec_Y_loc_diff = vec_Y_cen - list_gamma_est[j].predict(mat_X_cen)
                            
                            vec_U_loc_est = vec_Y_loc_diff - vec_D_cen * beta_est_ini
                            
                            mat_U_slope[:, j] = np.power(vec_U_loc_est, 2) - np.power(vec_U_cen_est, 2)
                            mat_U_slope[:, j] = np.exp(-.5 * psi_u_inv_est * mat_U_slope[:, j]) ## density ratio
                            mat_U_slope[:, j] = mat_U_slope[:, j] \
                                * ((1 - vec_D_cen) + (2 * vec_D_cen - 1) * vec_D_loc_est + 0.005) \
                                    / ((1 - vec_D_cen) + (2 * vec_D_cen - 1) * vec_D_cen_est + 0.005)
                            
                            mat_U_slope[:, j] = mat_U_slope[:, j] * vec_D_cen * vec_D_loc_diff

                            ## density estimation - local
                            den_est_x_loc = list_den_est[j].pred(ts_X_cen, bl_shift = True).detach().numpy()
                            den_estp_x_loc = scipy.stats.multivariate_normal.pdf(
                                mat_X_cen, mean = mat_mu_X_est[j, :], cov = arr_sig_X_est[j]
                            )
                            den_ora_x_loc = scipy.stats.multivariate_normal.pdf(
                                mat_X_cen, mean = mat_X_shift[j, :], cov = covmat_X
                            )

                            ## kernel estimation
                            if (j == j_cen): 
                                mat_U_slope_cs[:, j] = np.ones(n_est) 
                                mat_U_slope_csp[:, j] = np.ones(n_est)
                                mat_U_slope_ora[:, j] = np.ones(n_est)
                            else:
                                if (np.max(den_ora_x_loc) < np.min(den_ora_x_cen)): 
                                    den_ora_x_loc *= 0.0

                                mat_U_slope_cs[:, j] = den_est_x_loc / den_est_x_cen
                                mat_U_slope_csp[:, j] = den_estp_x_loc / den_estp_x_cen
                                mat_U_slope_ora[:, j] = den_ora_x_loc / den_ora_x_cen
                                # print(
                                #     ">> corr_den_est | ratio_est: ", 
                                #     np.corrcoef(den_est_x_loc, den_ora_x_loc)[0, 1].round(5), " | ",
                                #     np.corrcoef(mat_U_slope_cs[:, j], mat_U_slope_ora[:, j])[0, 1].round(5)
                                # )
                            
                            mat_U_slope_cs[:, j] *= mat_U_slope[:, j]
                            mat_U_slope_csp[:, j] *= mat_U_slope[:, j]
                            mat_U_slope_ora[:, j] *= mat_U_slope[:, j]

                    elif den_est == "ora_dou":
                        exit(">> oracle double density estimation is not supported in this version")
                    
                    U_slope = np.mean(mat_U_slope)
                    U_slope_cs = np.mean(mat_U_slope_cs)
                    U_slope_csp = np.mean(mat_U_slope_csp)
                    U_slope_ora = np.mean(mat_U_slope_ora)

                    beta_est_cen = beta_est_ini + S / U_slope
                    vec_beta_est_cen[j_cen] = beta_est_cen

                    beta_csp_est_cen = beta_est_ini + S / U_slope_csp
                    vec_beta_csp_est_cen[j_cen] = beta_csp_est_cen
                    
                    beta_cs_est_cen = beta_est_ini + S / U_slope_cs
                    vec_beta_cs_est_cen[j_cen] = beta_cs_est_cen

                    beta_ora_est_cen = beta_est_ini + S / U_slope_ora
                    vec_beta_ora_est_cen[j_cen] = beta_ora_est_cen

                ## final estimation
                beta_est = np.mean(vec_beta_est_cen)
                beta_est_cs = np.mean(vec_beta_cs_est_cen)
                beta_est_csp = np.mean(vec_beta_csp_est_cen)
                beta_est_ora = np.mean(vec_beta_ora_est_cen)

                vec_beta_est_iter[i_iter] = beta_est
                vec_beta_cs_est_iter[i_iter] = beta_est_cs
                vec_beta_csp_est_iter[i_iter] = beta_est_csp
                vec_beta_ora_est_iter[i_iter] = beta_est_ora

                # i_iter += 1

                # ## iteration
                # while i_iter < n_iter:
                    
                #     ## updating estimation of nuisance parameter
                #     for j in range(K):
                #         ## training ML model
                #         mat_X_nui = arr_X[j][idx_nui, :]
                #         vec_D_nui = mat_D[idx_nui, j]
                #         vec_Y_nui = mat_Y[idx_nui, j]
                        
                #         ## updating estimation of gamma
                #         model_gamma = RandomForestRegressor(n_estimators=n_rft)
                #         model_gamma.fit(mat_X_nui, vec_Y_nui - vec_D_nui * beta_est)

                #         list_gamma_est[j] = model_gamma

                #         model_gamma_cs = RandomForestRegressor(n_estimators=n_rft)
                #         model_gamma_cs.fit(mat_X_nui, vec_Y_nui - vec_D_nui * beta_est_cs)
                        
                #         list_gamma_cs_est[j] = model_gamma_cs

                #         model_gamma_ora = RandomForestRegressor(n_estimators=n_rft)
                #         model_gamma_ora.fit(mat_X_nui, vec_Y_nui - vec_D_nui * beta_est_ora)

                #         list_gamma_ora_est[j] = model_gamma_ora

                #     beta_est_ini = beta_est
                #     beta_est_ini_cs = beta_est_cs
                #     beta_est_ini_ora = beta_est_ora

                #     ## statistics from other sites
                #     vec_s = np.zeros(n_est)
                #     vec_S_est = np.zeros(K)

                #     vec_s_cs = np.zeros(n_est)
                #     vec_S_cs_est = np.zeros(K)
                    
                #     vec_s_ora = np.zeros(n_est)
                #     vec_S_ora_est = np.zeros(K)

                #     for j in range(K): 
                #         ## estimating
                #         mat_X_est = arr_X[j][idx_est, :]
                #         vec_D_est = mat_D[idx_est, j]
                #         vec_Y_est = mat_Y[idx_est, j]

                #         vec_D_diff = vec_D_est - list_mu_est[j].predict(mat_X_est)
                #         vec_Y_diff = vec_Y_est - list_gamma_est[j].predict(mat_X_est)
                        
                #         vec_s = (vec_Y_diff - vec_D_est * beta_est_ini) * vec_D_diff
                #         vec_S_est[j] = np.mean(vec_s)

                #         vec_Y_diff_cs = vec_Y_est - list_gamma_cs_est[j].predict(mat_X_est)
                #         vec_s_cs = (vec_Y_diff_cs - vec_D_est * beta_est_ini_cs) * vec_D_diff
                #         vec_S_cs_est[j] = np.mean(vec_s_cs)
                        
                #         vec_Y_diff_ora = vec_Y_est - list_gamma_ora_est[j].predict(mat_X_est)
                #         vec_s_ora = (vec_Y_diff_ora - vec_D_est * beta_est_ini_ora) * vec_D_diff
                #         vec_S_ora_est[j] = np.mean(vec_s_ora)

                #         # vec_D_ora_diff = vec_D_est - np.array(list(map(lambda X: fun_mu(X, j, mu_alpha, mu_beta, mu_ip), mat_X_est))) 
                #         # vec_Y_ora_diff = vec_Y_est - np.array(list(map(lambda X: fun_gamma(X, j, mat_gam[:, j]), mat_X_est)))

                #         # vec_s_ora = (vec_Y_ora_diff - vec_D_est * beta_est_ini) * vec_D_ora_diff
                #         # vec_S_ora_est[j] = np.mean(vec_s_ora)

                #     S = np.mean(vec_S_est)
                #     S_cs = np.mean(vec_S_cs_est)
                #     S_ora = np.mean(vec_S_ora_est)
                #     # S_ora = np.mean(vec_S_ora_est)

                #     ## operation in the central site
                #     vec_beta_est_cen = np.zeros(K)
                #     vec_beta_cs_est_cen = np.zeros(K)
                #     vec_beta_ora_est_cen = np.zeros(K)

                #     for j_cen in range(K):
                #         vec_Y_cen = mat_Y[idx_est, j_cen]
                #         vec_D_cen = mat_D[idx_est, j_cen]
                #         mat_X_cen = arr_X[j_cen][idx_est,]
                #         ts_X_cen = torch.from_numpy(mat_X_cen).float()

                #         ## density estimation - center
                #         den_est_x_cen = list_den_est[j_cen].pred(ts_X_cen).detach().numpy()
                #         den_ora_x_cen = scipy.stats.multivariate_normal.pdf(
                #             mat_X_cen, mean = mat_X_shift[j_cen, :], cov = covmat_X
                #         )

                #         mat_U_slope = np.zeros((n_est, K))
                #         mat_U_slope_cs = np.zeros((n_est, K))
                #         mat_U_slope_ora = np.zeros((n_est, K))

                #         if den_est == "single":
                #             exit(">> single density estimation is not supported in this version")

                #         elif den_est == "double":
                #             vec_D_cen_est = list_mu_est[j_cen].predict(mat_X_cen)
                #             vec_Y_cen_diff = vec_Y_cen - list_gamma_est[j_cen].predict(mat_X_cen)

                #             vec_U_cen_est = vec_Y_cen_diff - vec_D_cen * beta_est_ini

                #             for j in range(K):

                #                 vec_D_loc_est = list_mu_est[j].predict(mat_X_cen)
                #                 vec_D_loc_diff = vec_D_cen - vec_D_loc_est
                #                 vec_Y_loc_diff = vec_Y_cen - list_gamma_est[j].predict(mat_X_cen)
                                
                #                 vec_U_loc_est = vec_Y_loc_diff - vec_D_cen * beta_est_ini
                                
                #                 mat_U_slope[:, j] = np.power(vec_U_loc_est, 2) - np.power(vec_U_cen_est, 2)
                #                 mat_U_slope[:, j] = np.exp(-.5 * psi_u_inv_est * mat_U_slope[:, j]) ## density ratio
                #                 mat_U_slope[:, j] = mat_U_slope[:, j] \
                #                     * ((1 - vec_D_cen) + (2 * vec_D_cen - 1) * vec_D_loc_est + 0.005) \
                #                         / ((1 - vec_D_cen) + (2 * vec_D_cen - 1) * vec_D_cen_est + 0.005)

                #                 mat_U_slope[:, j] = mat_U_slope[:, j] * vec_D_cen * vec_D_loc_diff

                #                 ## density estimation - local
                #                 den_est_x_loc = list_den_est[j].pred(ts_X_cen, bl_shift = True).detach().numpy()
                #                 den_ora_x_loc = scipy.stats.multivariate_normal.pdf(
                #                     mat_X_cen, mean = mat_X_shift[j, :], cov = covmat_X
                #                 )

                #                 ## kernel estimation
                #                 if (j == j_cen): 
                #                     mat_U_slope_cs[:, j] = np.ones(n_est) 
                #                     mat_U_slope_ora[:, j] = np.ones(n_est)
                #                 else:
                #                     if (np.max(den_ora_x_loc) < np.min(den_ora_x_cen)): 
                #                         # print(">> ", mat_X_shift[j, :], " | ", mat_X_shift[j_cen, :])
                #                         den_ora_x_loc *= 0.0
                #                     mat_U_slope_cs[:, j] = den_est_x_loc / den_est_x_cen
                #                     mat_U_slope_ora[:, j] = den_ora_x_loc / den_ora_x_cen

                #                     # print(
                #                     #     ">> corr_den_est | ratio_est: ", 
                #                     #     np.corrcoef(den_est_x_loc, den_ora_x_loc).round(5), " | ", 
                #                     #     np.max(den_ora_x_loc), " | ", 
                #                     #     np.min(den_ora_x_loc), " || ", 
                #                     #     np.max(den_ora_x_cen), " | ", 
                #                     #     np.min(den_ora_x_cen), " || " 
                #                     # )
                #                     # exit("test")

                #                 mat_U_slope_cs[:, j] *= mat_U_slope[:, j]
                #                 mat_U_slope_ora[:, j] *= mat_U_slope[:, j]

                #         elif den_est == "ora_dou":
                #             exit(">> oracle double density estimation is not supported in this version")

                #         U_slope = np.mean(mat_U_slope)
                #         U_slope_cs = np.mean(mat_U_slope_cs)
                #         U_slope_ora = np.mean(mat_U_slope_ora)

                #         beta_est_cen = beta_est_ini + S / U_slope
                #         vec_beta_est_cen[j_cen] = beta_est_cen

                #         beta_cs_est_cen = beta_est_ini_cs + S_cs / U_slope_cs
                #         vec_beta_cs_est_cen[j_cen] = beta_cs_est_cen

                #         beta_ora_est_cen = beta_est_ini_ora + S_ora / U_slope_ora
                #         vec_beta_ora_est_cen[j_cen] = beta_ora_est_cen
                    
                #     ## final estimation
                #     beta_est = np.mean(vec_beta_est_cen)
                #     vec_beta_est_iter[i_iter] = beta_est

                #     beta_est_cs = np.mean(vec_beta_cs_est_cen)
                #     vec_beta_cs_est_iter[i_iter] = beta_est_cs

                #     beta_est_ora = np.mean(vec_beta_ora_est_cen)
                #     vec_beta_ora_est_iter[i_iter] = beta_est_ora

                #     i_iter += 1

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
                        ','.join(str(b.round(5)) for b in vec_beta_csp_est_iter), " | ",
                        ','.join(str(b.round(5)) for b in vec_beta_cs_est_iter), " | ",
                        ','.join(str(b.round(5)) for b in vec_beta_ora_est_iter),
                        sep=",",
                    )
                else:
                    print(
                        rnd_np,
                        rnd_ds,
                        np.mean(vec_beta_est_local),
                        ','.join(str(b.round(5)) for b in vec_beta_est_iter),
                        ','.join(str(b.round(5)) for b in vec_beta_csp_est_iter),
                        ','.join(str(b.round(5)) for b in vec_beta_cs_est_iter),
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