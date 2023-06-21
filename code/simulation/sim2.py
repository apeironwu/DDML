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
    jp = np.mod(j + 2, p)
    out = fun_sigm(X[j]) + .25 * X[jp]
    out = out * 3
    return out

def fun_gamma(X, j): 
    p = len(X)
    jp = np.mod(j + 2, p)
    out = X[jp] + .25 * fun_sigm(X[j])
    out = out * 12
    return out







## parameter setting
n = 200
K = 3

p = K + 5
beta = 2

psi_u = 36
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

    # rf_xi = RandomForestRegressor(n_estimators=578, max_features=20, max_depth=3, min_samples_leaf=6)
    rf_xi = RandomForestRegressor()
    rf_xi.fit(mat_X_nui, vec_Y_nui)

    # rf_mu = RandomForestRegressor(n_estimators=332, max_features=12, max_depth=5, min_samples_leaf=1)
    rf_mu = RandomForestRegressor()
    rf_mu.fit(mat_X_nui, vec_D_nui)

    list_xi_est.append(rf_xi)
    list_mu_est.append(rf_mu)

    vec_Y_res = vec_Y_cau - rf_xi.predict(mat_X_cau)
    vec_D_res = vec_D_cau - rf_mu.predict(mat_X_cau)

    vec_beta_est_local[j] = 1 / (vec_D_res.T @ vec_D_res) * (vec_D_res.T @ vec_Y_res)

    print("Var S org: ", np.var((vec_Y_res - vec_D_res * beta) * (vec_D_res)))
    

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
    print("Var S: ", np.var(vec_s_cur))
    
    vec_S_est[j] = np.mean(vec_s_cur)

S = np.mean(vec_S_est)

## [test] caculate oracle S estimation
vec_s_ora_cur = np.zeros(n_thresh)
vec_S_ora_est = np.zeros(K)

for j in range(K): 
    ## estimating
    mat_X_cau = arr_X[j][:n_thresh, :]
    vec_D_cau = mat_D[:n_thresh, j]
    vec_Y_cau = mat_Y[:n_thresh, j]
    
    vec_mu_ora_est = np.array(list(map(lambda X: fun_mu(X, j), mat_X_cau)))
    vec_gamma_ora_est = np.array(list(map(lambda X: fun_gamma(X, j), mat_X_cau)))

    vec_s_ora_cur = (vec_D_cau - vec_mu_ora_est) * (vec_Y_cau - vec_D_cau * beta_est_ini - vec_gamma_ora_est)
    
    vec_S_ora_est[j] = np.mean(vec_s_ora_cur)
    
S_ora = np.mean(vec_S_ora_est)

## [test] comparison of estimation and oracle estimation for S
for j in range(K):
    ## estimating
    mat_X_cau = arr_X[j][:n_thresh, :]
    vec_D_cau = mat_D[:n_thresh, j]
    vec_Y_cau = mat_Y[:n_thresh, j]
    
    vec_D_res = vec_D_cau - list_mu_est[j].predict(mat_X_cau)
    vec_Y_res = vec_Y_cau - list_xi_est[j].predict(mat_X_cau)

    vec_V_est = vec_D_res
    vec_U_est = vec_Y_res - vec_D_res * beta_est_ini
    
    vec_mu_ora_est = np.array(list(map(lambda X: fun_mu(X, j), mat_X_cau)))
    vec_xi_ora_est = np.array(list(map(lambda X: fun_gamma(X, j), mat_X_cau)))

    vec_V_ora_est = vec_D_cau - vec_mu_ora_est
    vec_U_ora_est = vec_Y_cau - vec_D_cau * beta_est_ini - vec_xi_ora_est

    
    print("Cor_V: est - ora_est: \n", np.corrcoef(
        np.column_stack((vec_V_est, vec_V_ora_est, mat_V[:n_thresh, j])).T
    ))
    print("Cor_U: est - ora_est: \n", np.corrcoef(
        np.column_stack((vec_U_est, vec_U_ora_est, mat_U[:n_thresh, j])).T
    ))

    








## operation in the central site
vec_beta_est_cen = np.zeros(K)
vec_beta_ora_est_cen = np.zeros(K)

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

        print("density ratio: ", np.median(mat_G_slope[:, j]))
        
        mat_G_slope[:, j] = mat_G_slope[:, j] * np.power(vec_D_cen - vec_mu_cur_pred, 2) ## residual product

        # print(" " * 40, "slope: ", np.mean(mat_G_slope[:, j]))
    

    G_slope = np.mean(mat_G_slope)

    beta_est_cen = beta_est_ini + S / G_slope
    vec_beta_est_cen[j_cen] = beta_est_cen

    beta_ora_est_cen = beta_est_ini + S_ora / G_slope
    vec_beta_ora_est_cen[j_cen] = beta_ora_est_cen

## final estimation
beta_est = np.mean(vec_beta_est_cen)

beta_est_ora = np.mean(vec_beta_ora_est_cen)


## output
print("|", "S estimation:       ", S)
print("|", "S oracle est:       ", S_ora)
print("|", "local estimation:   ", vec_beta_est_local)
print("|", "central estimation: ", vec_beta_est_cen)
print("|", "initial estimation: ", beta_est_ini)
print("|", "final estimation:   ", beta_est)
print("|", "oracle estimation:  ", beta_est_ora)

# print(rnd, beta_est, beta_est_ini, sep=", ")

# print("=" * 40)
# print("bias", beta_est - beta_est_ini)
