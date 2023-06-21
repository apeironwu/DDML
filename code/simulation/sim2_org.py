import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

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

# ## linear case
# def fun_mu(X, j):
#     p = len(X)
#     jp = np.mod(j + 2, p)
#     out = X[jp]
#     out = out * 3
#     return out

# def fun_gamma(X, j): 
#     out = X[j]
#     out = out * 12
#     return out




n_iter = 10
i_iter = 0

vec_beta_est_iter = np.zeros(n_iter)

## parameter setting
n = 100
K = 3

p = K + 3
beta = 2


# large variance
psi_u = 36
psi_v = 9

# ## small variance
# psi_u = .1
# psi_v = .1


psi_u_inv = 1 / psi_u

## data generation 

#### randomization
rnd = 132
np.random.seed(rnd)
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

list_gamma_est = list()
list_mu_est = list()

for j in range(K):
    ## training ML model
    mat_X_nui = arr_X[j][n_thresh:, :]
    vec_D_nui = mat_D[n_thresh:, j]
    vec_Y_nui = mat_Y[n_thresh:, j]
    
    model_mu = RandomForestRegressor(n_estimators=n_thresh)
    model_mu.fit(mat_X_nui, vec_D_nui)
    
    # model_mu = LinearRegression()
    # model_mu.fit(mat_X_nui, vec_D_nui)
    # print("model_mu.coef_: ", model_mu.coef_)

    beta_tilde = np.mean(vec_Y_nui.T * vec_D_nui) / np.mean(vec_D_nui.T * vec_D_nui)
    # beta_tilde = 2
    # print("beta_tilde: ", beta_tilde)

    model_gamma = RandomForestRegressor(n_estimators=n_thresh)
    model_gamma.fit(mat_X_nui, vec_Y_nui - vec_D_nui * beta_tilde)

    # model_gamma = LinearRegression()
    # model_gamma.fit(mat_X_nui, vec_Y_nui - vec_D_nui * beta_tilde)
    # print("model_gamma.coef_: ", model_gamma.coef_)

    list_mu_est.append(model_mu)
    list_gamma_est.append(model_gamma)


    ## estimating
    mat_X_est = arr_X[j][:n_thresh, :]
    vec_D_est = mat_D[:n_thresh, j]
    vec_Y_est = mat_Y[:n_thresh, j]

    vec_D_diff = vec_D_est - model_mu.predict(mat_X_est)
    vec_Y_diff = vec_Y_est - model_gamma.predict(mat_X_est)
    
    beta_est_local = np.mean(vec_D_diff * vec_Y_diff) / np.mean(vec_D_est * vec_D_diff)

    # vec_D_ora_diff = vec_D_est - np.array(list(map(lambda X: fun_mu(X, j), mat_X_est)))
    # vec_Y_ora_diff = vec_Y_est - np.array(list(map(lambda X: fun_gamma(X, j), mat_X_est)))
    
    # best_ora_est_local = np.mean(vec_D_ora_diff * vec_Y_ora_diff) / np.mean(vec_D_est * vec_D_ora_diff)

    # print(">> beta_est_local: ", beta_est_local)
    # print(">> best_ora_est_local: ", best_ora_est_local)
    
    vec_beta_est_local[j] = beta_est_local

beta_est_ini = np.mean(vec_beta_est_local)


## statistics from other sites
vec_s = np.zeros(n_thresh)
vec_S_est = np.zeros(K)

vec_s_ora = np.zeros(n_thresh)
vec_S_ora_est = np.zeros(K)

for j in range(K): 
    ## estimating
    mat_X_est = arr_X[j][:n_thresh, :]
    vec_D_est = mat_D[:n_thresh, j]
    vec_Y_est = mat_Y[:n_thresh, j]

    vec_D_diff = vec_D_est - list_mu_est[j].predict(mat_X_est)
    vec_Y_diff = vec_Y_est - list_gamma_est[j].predict(mat_X_est)
    
    vec_s = (vec_Y_diff - vec_D_est * beta_est_ini) * vec_D_diff
    
    vec_S_est[j] = np.mean(vec_s)

    vec_D_ora_diff = vec_D_est - np.array(list(map(lambda X: fun_mu(X, j), mat_X_est))) 
    vec_Y_ora_diff = vec_Y_est - np.array(list(map(lambda X: fun_gamma(X, j), mat_X_est)))
    
    vec_s_ora = (vec_Y_ora_diff - vec_D_est * beta_est_ini) * vec_D_ora_diff
    vec_S_ora_est[j] = np.mean(vec_s_ora)


S = np.mean(vec_S_est)

S_ora = np.mean(vec_S_ora_est)



## operation in the central site
vec_beta_est_cen = np.zeros(K)
vec_beta_ora_est_cen = np.zeros(K)

for j_cen in range(K):
# for j_cen in [0]:
    vec_Y_cen = mat_Y[:n_thresh, j_cen]
    vec_D_cen = mat_D[:n_thresh, j_cen]
    mat_X_cen = arr_X[j_cen][:n_thresh,]

    mat_U_slope = np.zeros((n_thresh, K))

    vec_D_cen_diff = vec_D_cen - list_mu_est[j_cen].predict(mat_X_cen)
    vec_Y_cen_diff = vec_Y_cen - list_gamma_est[j_cen].predict(mat_X_cen)

    vec_U_cen_est = vec_Y_cen_diff - vec_D_cen * beta_est_ini

    for j in range(K):

        vec_D_loc_diff = vec_D_cen - list_mu_est[j].predict(mat_X_cen)
        vec_Y_loc_diff = vec_Y_cen - list_gamma_est[j].predict(mat_X_cen)
        
        vec_U_loc_est = vec_Y_loc_diff - vec_D_cen * beta_est_ini
        
        mat_U_slope[:, j] = np.power(vec_U_loc_est, 2) - np.power(vec_U_cen_est, 2)
        mat_U_slope[:, j] = np.exp(-.5 * psi_u_inv * mat_U_slope[:, j]) ## density ratio

        print("density ratio: ", np.median(mat_U_slope[:, j]))
        
        mat_U_slope[:, j] = mat_U_slope[:, j] * vec_D_cen * vec_D_loc_diff

        # print(" " * 40, "U_slope:       ", np.median(mat_U_slope[:, j]))


    U_slope = np.mean(mat_U_slope)

    beta_est_cen = beta_est_ini + S / U_slope
    vec_beta_est_cen[j_cen] = beta_est_cen

    beta_ora_est_cen = beta_est_ini + S_ora / U_slope
    vec_beta_ora_est_cen[j_cen] = beta_ora_est_cen

## final estimation
beta_est = np.mean(vec_beta_est_cen)

beta_est_ora = np.mean(vec_beta_ora_est_cen)

vec_beta_est_iter[i_iter] = beta_est
i_iter += 1

## output
# print("|", "S estimation:       ", S)
# print("|", "S oracle est:       ", S_ora)
# print("|", "local estimation:   ", vec_beta_est_local)
# print("|", "central estimation: ", vec_beta_est_cen)
# print("|", "initial estimation: ", beta_est_ini)
# print("|", "final estimation:   ", beta_est)
# print("|", "oracle estimation:  ", beta_est_ora)

# print(rnd, beta_est, beta_est_ini, sep=", ")









## iteration
while i_iter < n_iter:
    
    ## updating estimation of nuisance parameter
    for j in range(K):
        ## training ML model
        mat_X_nui = arr_X[j][n_thresh:, :]
        vec_D_nui = mat_D[n_thresh:, j]
        vec_Y_nui = mat_Y[n_thresh:, j]
        
        ## updating estimation of gamma
        model_gamma = RandomForestRegressor(n_estimators=n_thresh)
        model_gamma.fit(mat_X_nui, vec_Y_nui - vec_D_nui * beta_est)

        list_gamma_est[j] = model_gamma

    beta_est_ini = beta_est

    ## statistics from other sites
    vec_s = np.zeros(n_thresh)
    vec_S_est = np.zeros(K)

    # vec_s_ora = np.zeros(n_thresh)
    # vec_S_ora_est = np.zeros(K)

    for j in range(K): 
        ## estimating
        mat_X_est = arr_X[j][:n_thresh, :]
        vec_D_est = mat_D[:n_thresh, j]
        vec_Y_est = mat_Y[:n_thresh, j]

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
    vec_beta_ora_est_cen = np.zeros(K)

    for j_cen in range(K):
    # for j_cen in [0]:
        vec_Y_cen = mat_Y[:n_thresh, j_cen]
        vec_D_cen = mat_D[:n_thresh, j_cen]
        mat_X_cen = arr_X[j_cen][:n_thresh,]

        mat_U_slope = np.zeros((n_thresh, K))

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


print("beta local:  ", np.mean(vec_beta_est_local))
print("beta iteration: ", vec_beta_est_iter)
    
    



