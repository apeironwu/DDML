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



## parameter setting
n = 100
K = 4

p = K
beta = 0.5

psi_u = 16
psi_v = 9

psi_u_inv = 1 / psi_u

# print("rnd", "beta_est", "beta_ini", sep=", ")

## data generation 

#### randomization
# rnd = 128
# np.random.seed(rnd)
# for rnd in (128 + np.array(range(1))): 
# np.random.seed(rnd)

#### uncorrelated X
arr_X = np.random.randn(K, n, p)

# #### correlated X
# covmat_X = np.fromfunction(lambda i, j: np.power(0.7, np.abs(i - j)), (p, p))
# arr_X = np.random.multivariate_normal(np.zeros(p), covmat_X, K * n).reshape(K, n, p)

mat_U = np.random.randn(n, K) * np.sqrt(psi_u)
mat_V = np.random.randn(n, K) * np.sqrt(psi_v)

mat_D = np.zeros((n, K))
mat_Y = np.zeros((n, K))

for j in range(K): 
    mat_D[:, j] = np.array(list(map(lambda X: fun_mu(X, j), arr_X[j]))) + mat_V[:, j]
    mat_Y[:, j] = mat_D[:, j] * beta \
        + np.array(list(map(lambda X: fun_gamma(X, j), arr_X[j]))) + mat_U[:, j]

## data splitting
n_thresh = int(n / 2)

## initial estimation of beta and estimation of nuisance parameter
vec_beta_est_local = np.zeros(K)

list_xi_est = list()
list_mu_est = list()

for j in range(K):
    ## training ML model
    mat_X_nui = arr_X[j][n_thresh:, :]
    vec_D_nui = mat_D[n_thresh:, j]
    vec_Y_nui = mat_Y[n_thresh:, j]
    
    ## estimating
    mat_X_cau = arr_X[j][:n_thresh, :]
    vec_D_cau = mat_D[:n_thresh, j]
    vec_Y_cau = mat_Y[:n_thresh, j]

    rf_xi = RandomForestRegressor()
    rf_xi.fit(mat_X_nui, vec_Y_nui)

    rf_mu = RandomForestRegressor()
    rf_mu.fit(mat_X_nui, vec_D_nui)

    vec_Y_res = vec_Y_cau - rf_xi.predict(mat_X_cau)
    vec_D_res = vec_D_cau - rf_mu.predict(mat_X_cau)

    vec_beta_est_local[j] = 1 / (vec_D_res.T @ vec_D_res) * (vec_D_res.T @ vec_Y_res)
    
    list_xi_est.append(rf_xi)
    list_mu_est.append(rf_mu)

beta_est_ini = np.mean(vec_beta_est_local)

## statistics from other sites
vec_s_cur = np.zeros(n_thresh)
vec_S_est = np.zeros(K)

for j in range(K): 
    ## estimating
    mat_X_cau = arr_X[j][:n_thresh, :]
    vec_D_cau = mat_D[:n_thresh, j]
    vec_Y_cau = mat_Y[:n_thresh, j]

    vec_D_res = vec_D_cau - list_mu_est[j].predict(mat_X_cau)
    vec_Y_res = vec_Y_cau - list_xi_est[j].predict(mat_X_cau)
    
    vec_s_cur = vec_D_res * (vec_Y_res - vec_D_res * beta_est_ini)
    
    print("Var V: ", np.var(vec_D_res))
    print("Var U: ", np.var(vec_Y_res - vec_D_res * beta_est_ini))
    
    vec_S_est[j] = np.mean(vec_s_cur)

S = np.mean(vec_S_est)

## operation in the central site
vec_beta_est_cen = np.zeros(K)

for j_cen in range(K):
# for j_cen in [0]:
    vec_Y_cen = mat_Y[:n_thresh, j_cen]
    vec_D_cen = mat_D[:n_thresh, j_cen]
    mat_X_cen = arr_X[j_cen][:n_thresh, :]

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

## final estimation
beta_est = np.mean(vec_beta_est_cen)

## output
print("S: ", S)
print("local estimation:   ", vec_beta_est_local)
print("central estimation: ", vec_beta_est_cen)
print("initial estimation: ", beta_est_ini)
print("final estimation:   ", beta_est)

# print(rnd, beta_est, beta_est_ini, sep=", ")

# print("=" * 40)
# print("bias", beta_est - beta_est_ini)
