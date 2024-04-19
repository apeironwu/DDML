import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from timeit import default_timer as timer
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

def fun_trunc_bd(vec_den_ratio):
    vec_den_ratio = np.maximum(vec_den_ratio, 0.05) ## lower bound
    vec_den_ratio = np.minimum(vec_den_ratio, 0.95) ## upper bound
    return vec_den_ratio

def main(argv): 

    ## default values
    n = 1000
    K = 5
    p = 20
    n_rnp_gen = 100
    n_rnp_ds = 10
    n_iter = 1
    n_rft = 200
    path_out = None
    sd_cs = 0.5
    
    prompt_help = [
        'sim2_org_ini_rds_args.py',
        '--n <# sample size>',
        '--K <# of sites>',
        '--p <# of confounders>', 
        '--n_rnp_gen <# replication of data generation>',
        '--n_rnp_ds <# data splitting>',
        '--n_iter <# iteration>',
        '--n_rft <# random forest trees>',
        '--path_out <output path>', 
        '--sd_cs <sd in covariate shift>' 
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
                "n_rft=",
                "path_out=", 
                "sd_cs="
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
        elif opt in ("--n_rnp_gen"):
            n_rnp_gen = int(arg)
        elif opt in ("--n_rnp_ds"):
            n_rnp_ds = int(arg)
        elif opt in ("--n_iter"):
            n_iter = int(arg)
        elif opt in ("--n_rft"):
            n_rft = int(arg)
        elif opt in ("--path_out"):
            path_out = arg
        elif opt in ("--sd_cs"):
            sd_cs = float(arg)

    print('=' * 20, "Parameter Setting", '=' * 20)
    print('>> n: ', n)
    print('>> K: ', K)
    print('>> p: ', p)
    print('>> n_rnp_gen: ', n_rnp_gen)
    print('>> n_rnp_ds: ', n_rnp_ds)
    print('>> n_iter: ', n_iter)
    print('>> n_rft: ', n_rft)
    print('>> path_out: ', path_out)
    print('>> sd_cs: ', sd_cs)

    time_start = time.time()


    ## parameter setting
    vec_beta_est_iter = np.zeros(n_iter)
    vec_beta_cs_est_iter = np.zeros(n_iter)
    # vec_beta_cs_ora_est_iter = np.zeros(n_iter)

    beta = 0.5

    # large variance
    psi_u = 1
    psi_v = 1

    psi_u_inv = 1 / psi_u
    psi_v_inv = 1 / psi_v

    covmat_X = np.fromfunction(lambda i, j: np.power(0.7, np.abs(i - j)), (p, p))

    mat_idx_ls_cs = np.zeros((K, K), dtype=int)
    idx_ls_cs = 0

    for j_cen in range(K - 1):
        j = j_cen + 1
        while j < K:
            mat_idx_ls_cs[j, j_cen] = idx_ls_cs
            j = j + 1
            idx_ls_cs = idx_ls_cs + 1

    mat_idx_ls_cs = mat_idx_ls_cs + mat_idx_ls_cs.T
    
    np.fill_diagonal(mat_idx_ls_cs, -1)

    tol_idx_ls_cs = int(K * (K - 1) / 2)

    if path_out is None: 
        print(
            "rnd_gen", "rnd_ds", "Average", 
            ",".join(["M" + str(i + 1) for i in range(n_iter)]),
            ",".join(["Ms" + str(i + 1) for i in range(n_iter)]),
            # ",".join(["Mso" + str(i + 1) for i in range(n_iter)]),
            sep=",", 
        )
    else:
        print(
            "rnd_gen", "rnd_ds", "Average", 
            ",".join(["M" + str(i + 1) for i in range(n_iter)]),
            ",".join(["Ms" + str(i + 1) for i in range(n_iter)]),
            # ",".join(["Mso" + str(i + 1) for i in range(n_iter)]),
            sep=",", 
            file = open(path_out, "w")
        )
    
    ## data generation 
    #### randomization
    for rnd_gen in (2023 + np.array(range(n_rnp_gen))): 
        for rnd_ds in (2023 + np.array(range(n_rnp_ds))):

            try:

                ## set `numpy` random seed
                np.random.seed(rnd_gen)
                
                #### covariate shift coefficient
                mat_cs_theta = np.zeros((p, K))
                for j in range(K): 
                    mat_cs_theta[0, j] = np.random.randn()
                    mat_cs_theta[j+1, j] = np.random.randn()
                mat_cs_theta = mat_cs_theta * sd_cs

                # mat_cs_theta = np.random.randn(p, K) * sd_cs
                mat_cs_mu = covmat_X @ mat_cs_theta
                vec_cs_quad = np.diag(mat_cs_theta.T @ covmat_X @ mat_cs_theta)

                #### correlated X
                arr_X = np.zeros((K, n, p))
                for j in range(K):
                    arr_X[j, :, :] = np.random.multivariate_normal(mat_cs_mu[:, j], covmat_X, n)
                
                mat_U = np.random.randn(n, K) * np.sqrt(psi_u)
                mat_V = np.random.randn(n, K) * np.sqrt(psi_v)

                mat_D = np.zeros((n, K))
                mat_Y = np.zeros((n, K))

                for j in range(K): 
                    mat_D[:, j] = np.array(list(map(lambda X: fun_mu(X, j), arr_X[j]))) + mat_V[:, j]
                    mat_Y[:, j] = mat_D[:, j] * beta \
                        + np.array(list(map(lambda X: fun_gamma(X, j), arr_X[j]))) + mat_U[:, j]

                ## randomly data splitting
                K_fold = 2

                #### set `numpy` random seed for estimation
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
                    
                    n_est = idx_est.shape[0]

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
                        vec_Y_diff = vec_Y_est - model_xi.predict(mat_X_est)
                        
                        beta_est_local = np.mean(vec_Y_diff * vec_D_diff) / np.mean(vec_D_diff * vec_D_diff)

                        model_gamma = RandomForestRegressor(n_estimators=n_rft)
                        model_gamma.fit(mat_X_nui, vec_Y_nui - vec_D_nui * beta_est_local)

                        list_mu_est.append(model_mu)
                        list_gamma_est.append(model_gamma)

                        mat_beta_est_local[j, splt] = beta_est_local

                beta_est_ini = np.mean(mat_beta_est_local)

                mat_beta_est_cen = np.zeros((K, K_fold))
                mat_beta_cs_est_cen = np.zeros((K, K_fold))
                # mat_beta_cs_ora_est_cen = np.zeros((K, K_fold))

                list_cs_clf = list()

                for splt in range(K_fold):
                    idx_nui = np.where(idx_K_fold != splt)[0]
                    n_nui = idx_nui.shape[0]
                    
                    for j_cen in range(K - 1):
                        mat_X_cen = arr_X[j_cen][idx_nui, :]
                        
                        j = j_cen + 1
                        while j < K: 
                            mat_X_loc = arr_X[j][idx_nui, :]
                            
                            mat_X_comb = np.concatenate((mat_X_loc, mat_X_cen), axis=0)
                            vec_G_comb = np.concatenate((np.zeros(n_nui), np.ones(n_nui)))

                            model_cs_clf = RandomForestClassifier(n_estimators=n_rft)
                            model_cs_clf.fit(mat_X_comb, vec_G_comb)

                            list_cs_clf.append(model_cs_clf)

                            j = j + 1

                for splt in range(K_fold):
                    idx_est = np.where(idx_K_fold == splt)[0]

                    n_est = idx_est.shape[0]

                    ## statistics from other sites
                    vec_s = np.zeros(n_est)
                    vec_S_est = np.zeros(K)

                    for j in range(K): 
                        idx_ls = j + splt * K

                        ## estimating
                        mat_X_est = arr_X[j][idx_est, :]
                        vec_D_est = mat_D[idx_est, j]
                        vec_Y_est = mat_Y[idx_est, j]

                        vec_D_diff = vec_D_est - list_mu_est[idx_ls].predict(mat_X_est)
                        vec_Y_diff = vec_Y_est - list_gamma_est[idx_ls].predict(mat_X_est)
                        
                        vec_s = (vec_Y_diff - vec_D_est * beta_est_ini) * vec_D_diff
                        
                        vec_S_est[j] = np.mean(vec_s)

                    S = np.mean(vec_S_est)
                    vec_den_ratio_cs = np.zeros(n_est)
                    vec_den_ratio_cs_ora = np.zeros(n_est)
                    
                    for j_cen in range(K):
                        idx_ls_cen = j_cen + splt * K

                        vec_Y_cen = mat_Y[idx_est, j_cen]
                        vec_D_cen = mat_D[idx_est, j_cen]
                        mat_X_cen = arr_X[j_cen][idx_est,]

                        mat_U_slope = np.zeros((n_est, K))
                        mat_U_slope_cs = np.zeros((n_est, K))
                        # mat_U_slope_cs_ora = np.zeros((n_est, K))

                        # single-equation density estimation
                        vec_Y_cen_diff = vec_Y_cen - list_gamma_est[idx_ls_cen].predict(mat_X_cen)

                        vec_U_cen_est = vec_Y_cen_diff - vec_D_cen * beta_est_ini

                        for j in range(K):
                            idx_ls = j + splt * K

                            vec_D_loc_diff = vec_D_cen - list_mu_est[idx_ls].predict(mat_X_cen)

                            vec_Y_loc_diff = vec_Y_cen - list_gamma_est[idx_ls].predict(mat_X_cen)
                            
                            vec_U_loc_est = vec_Y_loc_diff - vec_D_cen * beta_est_ini
                            
                            ## density ratio estimation
                            mat_U_slope[:, j] = np.power(vec_U_loc_est, 2) - np.power(vec_U_cen_est, 2)
                            mat_U_slope[:, j] = np.exp(-.5 * psi_u_inv * mat_U_slope[:, j])
                            ## density ratio estimation ----
                            
                            ## density ratio with covariate shift
                            if (j_cen == j): 
                                vec_den_ratio_cs = np.ones(n_est)
                            elif (j_cen < j):
                                idx_ls_cs = mat_idx_ls_cs[j, j_cen] + splt * tol_idx_ls_cs
                                # print("j_cen: ", j_cen, " j: ", j, " idx_ls_cs: ", idx_ls_cs)
                                model_cs_clf = list_cs_clf[idx_ls_cs]
                                vec_den_ratio_cs = model_cs_clf.predict_proba(mat_X_cen)[:, 1]
                                vec_den_ratio_cs = fun_trunc_bd(vec_den_ratio_cs)
                                # print(np.max(vec_den_ratio_cs), np.min(vec_den_ratio_cs))
                                vec_den_ratio_cs = (1.0 - vec_den_ratio_cs) / (vec_den_ratio_cs)
                            elif (j_cen > j):
                                idx_ls_cs = mat_idx_ls_cs[j, j_cen] + splt * tol_idx_ls_cs
                                # print("j_cen: ", j_cen, " j: ", j, " idx_ls_cs: ", idx_ls_cs)
                                model_cs_clf = list_cs_clf[idx_ls_cs]
                                vec_den_ratio_cs = model_cs_clf.predict_proba(mat_X_cen)[:, 1]
                                vec_den_ratio_cs = fun_trunc_bd(vec_den_ratio_cs)
                                # print(np.max(vec_den_ratio_cs), np.min(vec_den_ratio_cs))
                                vec_den_ratio_cs = vec_den_ratio_cs / (1.0 - vec_den_ratio_cs)

                            mat_U_slope_cs[:, j] = vec_den_ratio_cs * mat_U_slope[:, j]
                            ## density ratio with covariate shift ----

                            # ## oracle density ratio with covariate shift
                            # vec_den_ratio_cs_ora = 2.0 * mat_X_cen @ (mat_cs_theta[:, j_cen] - mat_cs_theta[:, j])
                            # vec_den_ratio_cs_ora = vec_den_ratio_cs_ora + vec_cs_quad[j] - vec_cs_quad[j_cen]
                            # vec_den_ratio_cs_ora = np.exp(-0.5 * vec_den_ratio_cs_ora)
                            # mat_U_slope_cs_ora[:, j] = vec_den_ratio_cs_ora * mat_U_slope[:, j]
                            # ## oracle density ratio with covariate shift ----

                            # if (j_cen != j):
                            #     print(np.corrcoef(vec_den_ratio_cs, vec_den_ratio_cs_ora)[0, 1])
                            
                            mat_U_slope[:, j] = mat_U_slope[:, j] * vec_D_cen * vec_D_loc_diff
                            mat_U_slope_cs[:, j] = mat_U_slope_cs[:, j] * vec_D_cen * vec_D_loc_diff
                            # mat_U_slope_cs_ora[:, j] = mat_U_slope_cs_ora[:, j] * vec_D_cen * vec_D_loc_diff

                        U_slope = np.mean(mat_U_slope)
                        U_slope_cs = np.mean(mat_U_slope_cs)
                        # U_slope_cs_ora = np.mean(mat_U_slope_cs_ora)

                        beta_est_cen = beta_est_ini + S / U_slope
                        mat_beta_est_cen[j_cen, splt] = beta_est_cen

                        beta_est_cen_cs = beta_est_ini + S / U_slope_cs
                        mat_beta_cs_est_cen[j_cen, splt] = beta_est_cen_cs

                        # beta_est_cen_cs_ora = beta_est_ini + S / U_slope_cs_ora
                        # mat_beta_cs_ora_est_cen[j_cen, splt] = beta_est_cen_cs_ora

                ## final estimation
                beta_est = np.mean(mat_beta_est_cen)
                vec_beta_est_iter[i_iter] = beta_est
                
                beta_est_cs = np.mean(mat_beta_cs_est_cen)
                vec_beta_cs_est_iter[i_iter] = beta_est_cs

                # beta_est_cs_ora = np.mean(mat_beta_cs_ora_est_cen)
                # vec_beta_cs_ora_est_iter[i_iter] = beta_est_cs_ora

                if path_out is None: 
                    print(
                        rnd_gen,
                        rnd_ds,
                        np.mean(mat_beta_est_local),
                        ','.join(str(b) for b in vec_beta_est_iter),
                        ','.join(str(b) for b in vec_beta_cs_est_iter),
                        # ','.join(str(b) for b in vec_beta_cs_ora_est_iter),
                        sep=",",
                    )
                else:
                    print(
                        rnd_gen,
                        rnd_ds,
                        np.mean(mat_beta_est_local),
                        ','.join(str(b) for b in vec_beta_est_iter),
                        ','.join(str(b) for b in vec_beta_cs_est_iter),
                        # ','.join(str(b) for b in vec_beta_cs_ora_est_iter),
                        sep=",",
                        file=open(path_out, "a")
                    )
            
            except ValueError:
                print("ValueError:  ", "rnp_", rnd_gen, " rds_", rnd_ds, sep="")
                continue

    print("running time: ", time.time() - time_start, " seconds ")

if __name__ == "__main__":
    main(sys.argv[1:])