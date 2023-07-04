import numpy as np
from sklearn.ensemble import RandomForestRegressor

def fun_sigm(X): 
    out = np.exp(X) / (1 + np.exp(X)) 
    return out

# def fun_mu(X, j):
#     p = len(X)
#     js = np.mod(j + 2, p)
#     out = fun_sigm(X[j]) + .25 * X[js]
#     out = out * 3
#     return out

# def fun_gamma(X, j): 
#     p = len(X)
#     js = np.mod(j + 2, p)
#     out = X[j] + .25 * fun_sigm(X[js])
#     out = out * 5
#     return out

def fun_mu(X, j):
    p = len(X)
    jp = np.mod(j + 1, p)
    jm = np.mod(j - 1, p)
    out = fun_sigm(X[j]) + .25 * X[jp] + .25 * X[jm]
    out = out * 2
    return out

def fun_gamma(X, j): 
    p = len(X)
    jp = np.mod(j + 1, p)
    jm = np.mod(j - 1, p)
    out = X[j] - .25 * fun_sigm(X[jp]) - .25 * fun_sigm(X[jm])
    out = out * 12
    return out

def fun_gen_data(cor_X = False): 
    if (cor_X == False): 
        arr_X = np.random.randn(K, n, p)
    else:
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
    
    return arr_X, mat_D, mat_Y

def fun_est_nui(arr_X, mat_Y, mat_D): 
    """Estimation of nuisance parameters"""

    list_xi_est = list()
    list_mu_est = list()
    
    for j in range(K):
        rf_xi = RandomForestRegressor()
        rf_xi.fit(arr_X[j], mat_Y[:, j])

        rf_mu = RandomForestRegressor()
        rf_mu.fit(arr_X[j], mat_D[:, j])

        list_xi_est.append(rf_xi)
        list_mu_est.append(rf_mu)

    return list_xi_est, list_mu_est 

def fun_est_beta_local(arr_X, mat_Y, mat_D, list_xi_est, list_mu_est): 
    """Local estimation of beta in each site"""
    vec_beta_est_local = np.zeros(K)

    for j in range(K):
        rf_xi = list_xi_est[j]
        rf_mu = list_mu_est[j]

        vec_Y_res = mat_Y[:, j] - rf_xi.predict(arr_X[j])
        vec_D_res = mat_D[:, j] - rf_mu.predict(arr_X[j])

        vec_beta_est_local[j] = 1 / (vec_D_res.T @ vec_D_res) * (vec_D_res.T @ vec_Y_res)

    return vec_beta_est_local

def fun_est_S(arr_X, mat_Y, mat_D, beta_est_ini, list_xi_est, list_mu_est): 
    """Estimation of S"""
    vec_S_est = np.zeros(K)

    for j in range(K): 
        ## estimating
        mat_X_cur = arr_X[j]
        vec_D_cur = mat_D[:, j]
        vec_Y_cur = mat_Y[:, j]

        vec_D_res = vec_D_cur - list_mu_est[j].predict(mat_X_cur)
        vec_Y_res = vec_Y_cur - list_xi_est[j].predict(mat_X_cur)
        
        vec_s_cur = vec_D_res * (vec_Y_res - vec_D_res * beta_est_ini)
        
        vec_S_est[j] = np.mean(vec_s_cur)

    S = np.mean(vec_S_est)

    return S

def fun_est_beta_cen(arr_X, mat_Y, mat_D, S, beta_est_ini, list_xi_est, list_mu_est): 
    """Estimation of beta in the central site"""
    vec_beta_est_cen = np.zeros(K)

    for j_cen in range(K):
        vec_Y_cen = mat_Y[:, j_cen]
        vec_D_cen = mat_D[:, j_cen]
        mat_X_cen = arr_X[j_cen]

        mat_G_slope = np.zeros((n_thresh, K))

        rf_xi_cen = list_xi_est[j_cen]
        rf_mu_cen = list_mu_est[j_cen]

        vec_xi_cen_pred = rf_xi_cen.predict(mat_X_cen)
        vec_mu_cen_pred = rf_mu_cen.predict(mat_X_cen)

        for j in range(K):
            rf_xi_cur = list_xi_est[j]
            rf_mu_cur = list_mu_est[j]

            vec_xi_cur_pred = rf_xi_cur.predict(mat_X_cen)
            vec_mu_cur_pred = rf_mu_cur.predict(mat_X_cen)

            vec_Y_res_cen = vec_Y_cen - vec_xi_cen_pred - \
                (vec_D_cen - vec_mu_cen_pred) * beta_est_ini
            vec_Y_res_cur = vec_Y_cen - vec_xi_cur_pred - \
                (vec_D_cen - vec_mu_cur_pred) * beta_est_ini
            
            # print(np.var(vec_Y_res_cen), np.var(vec_Y_res_cur))

            mat_G_slope[:, j] = np.power(vec_Y_res_cur, 2) - np.power(vec_Y_res_cen, 2)
            mat_G_slope[:, j] = np.exp(-.5 * psi_u_inv * mat_G_slope[:, j]) ## density ratio

            # print("density ratio: ", np.median(mat_G_slope[:, j]))
            
            mat_G_slope[:, j] = mat_G_slope[:, j] * np.power(vec_D_cen - vec_mu_cur_pred, 2) ## residual product

            # print(" " * 40, "slope: ", np.mean(mat_G_slope[:, j]))
        

        G_slope = np.mean(mat_G_slope)

        beta_est_cen = beta_est_ini + S / G_slope

        vec_beta_est_cen[j_cen] = beta_est_cen

    return vec_beta_est_cen





## csv file output
# print("rnd", "beta_est", "beta_ini", sep=", ")

n_iter = 10
vec_beta_est_iter = np.zeros(n_iter)

## parameter setting
n = 100
K = 3

p = K
beta = 0.5

psi_u = 16
psi_v = 9

psi_u_inv = 1 / psi_u


## data generation 

#### randomization
## once
# rnd = 128
# np.random.seed(rnd)

# ## loop
# for rnd in (128 + np.array(range(2))): 
# np.random.seed(rnd)

arr_X, mat_D, mat_Y = fun_gen_data(cor_X=False)


## data splitting
n_thresh = int(n / 2)

## estimation of nuisance parameter
list_xi_est, list_mu_est = fun_est_nui(arr_X[:, n_thresh:], mat_Y[n_thresh:], mat_D[n_thresh:])

## initial estimation of beta
vec_beta_est_local = fun_est_beta_local(
    arr_X[:, :n_thresh], mat_Y[:n_thresh], mat_D[:n_thresh], 
    list_xi_est, list_mu_est
)

beta_est_ini = np.mean(vec_beta_est_local)

iter = 0
vec_beta_est_iter[iter] = beta_est_ini

## iteration
while True:
    ## statistics from other sites
    S = fun_est_S(
        arr_X[:, :n_thresh], mat_Y[:n_thresh], mat_D[:n_thresh], 
        beta_est_ini, list_xi_est, list_mu_est
    )

    ## operation in the central site
    vec_beta_est_cen = fun_est_beta_cen(
        arr_X[:, :n_thresh], mat_Y[:n_thresh], mat_D[:n_thresh],
        S, beta_est_ini, list_xi_est, list_mu_est
    )

    ## final estimation
    beta_est = np.mean(vec_beta_est_cen)

    iter += 1
    vec_beta_est_iter[iter] = beta_est
    beta_est_ini = beta_est
    
    if (iter >= n_iter - 1): 
        break

## output
print(vec_beta_est_iter)
