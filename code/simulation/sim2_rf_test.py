import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def fun_sigm(X): 
    out = np.exp(X) / (1 + np.exp(X)) 
    return out

def fun_mu(X, j):
    p = len(X)
    jp = np.mod(j + 2, p)
    out = X[j] + .25 * fun_sigm(X[jp])
    out = out * 2
    return out

def fun_gamma(X, j): 
    p = len(X)
    jp = np.mod(j + 2, p)

    out = X[jp] + .25 * fun_sigm(X[j])
    out = out * 2
    return out

## parameter setting
n = 1000
K = 3

p = K + 3
beta = 2

psi_u = 36
psi_v = 16

psi_u_inv = 1 / psi_u

## data generation 

#### randomization
# rnd = 128
# np.random.seed(rnd)
# for rnd in (128 + np.array(range(1))): 
# np.random.seed(rnd)

#### uncorrelated X
# arr_X = np.random.randn(K, n, p)

# #### correlated X
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

    ## Random Forest
    rf_xi = RandomForestRegressor(n_estimators=378, max_features=20, max_depth=3, min_samples_leaf=6)
    rf_xi = RandomForestRegressor()
    rf_xi.fit(mat_X_nui, vec_Y_nui)

    rf_mu = RandomForestRegressor(n_estimators=132, max_features=12, max_depth=5, min_samples_leaf=1)
    rf_mu = RandomForestRegressor()
    rf_mu.fit(mat_X_nui, vec_D_nui)

    list_xi_est.append(rf_xi)
    list_mu_est.append(rf_mu)
    
 


    vec_Y_res = vec_Y_cau - rf_xi.predict(mat_X_cau)
    vec_D_res = vec_D_cau - rf_mu.predict(mat_X_cau)
    
    print("var(Y_res) est: ", np.var(vec_Y_res))
    print("var(V) est:     ", np.var(vec_D_res))

    vec_beta_est_local[j] = 1 / (vec_D_res.T @ vec_D_res) * (vec_D_res.T @ vec_Y_res)
    
    vec_U_est = vec_Y_res - vec_D_res * vec_beta_est_local[j]

    print("mean(U) est: ", np.mean(vec_U_est), np.var(vec_U_est))


beta_est_ini = np.mean(vec_beta_est_local)




## statistics from other sites
vec_UV_est = np.zeros(n_thresh)
vec_S_est = np.zeros(K)

vec_UV_ora_est = np.zeros(n_thresh)
vec_S_ora_est = np.zeros(K)

for j in range(K): 
    ## estimating
    mat_X_cau = arr_X[j][:n_thresh, :]
    vec_D_cau = mat_D[:n_thresh, j]
    vec_Y_cau = mat_Y[:n_thresh, j]
    
    vec_mu_est = list_mu_est[j].predict(mat_X_cau)
    vec_xi_est = list_xi_est[j].predict(mat_X_cau)

    vec_V_est = vec_D_cau - vec_mu_est
    vec_U_est = vec_Y_cau - vec_V_est * beta_est_ini - vec_xi_est

    vec_V_ora_est = vec_D_cau - np.array(list(map(lambda X: fun_mu(X, j), mat_X_cau)))
    vec_U_ora_est = vec_Y_cau - vec_D_cau * beta_est_ini - np.array(list(map(lambda X: fun_gamma(X, j), mat_X_cau)))

    print("vec_V_est: ", np.var(vec_V_est))
    print("vec_U_est: ", np.var(vec_U_est))

    vec_UV_est = vec_U_est * vec_V_est
    vec_S_est[j] = np.mean(vec_UV_est)

    print("vec_V_ora_est: ", np.var(vec_V_ora_est))
    print("vec_U_ora_est: ", np.var(vec_U_ora_est))

    vec_UV_ora_est = vec_U_ora_est * vec_V_ora_est
    vec_S_ora_est[j] = np.mean(vec_UV_ora_est)
    
    # print("Cor_S: \n", np.corrcoef(vec_UV_est, vec_UV_ora_est))



S_est = np.mean(vec_S_est)
S_ora_est = np.mean(vec_S_ora_est)

print("vec_S_est:     ", vec_S_est)
print("vec_S_ora_est: ", vec_S_ora_est)
print("| S_est:       ", S_est)
print("| S_ora_est:   ", S_ora_est)
