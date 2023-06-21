import numpy as np

## random seed
# np.random.seed(128)

## parameter setting
n = 500
K = 3

p_gamma = 3
beta = 2

mat_gamma = np.random.randn(K, p_gamma) * 4
mat_mu = np.random.randn(K, p_gamma) * 4

psi_u = 16
psi_v = 9

psi_u_inv = 1 / psi_u


## data generation 

# #### correlated X
# covmat_X = np.fromfunction(lambda i, j: np.power(-0.7, np.abs(i - j)), (p_gamma, p_gamma))
# arr_X = np.random.multivariate_normal(np.zeros(p_gamma), covmat_X, K * n).reshape(K, n, p_gamma)

#### uncorrelated X
arr_X = np.random.randn(K, n, p_gamma)

mat_U = np.random.randn(n, K) * np.sqrt(psi_u)
mat_V = np.random.randn(n, K) * np.sqrt(psi_v)

mat_D = np.zeros((n, K))
mat_Y = np.zeros((n, K))

for j in range(K): 
    mat_D[:, j] = arr_X[j] @ mat_mu[j] + mat_V[:, j]
    mat_Y[:, j] = mat_D[:, j] * beta + arr_X[j] @ mat_gamma[j] + mat_U[:, j]


## initial estimation of beta and estimation of nuisance parameter
vec_beta_est_local = np.zeros(K)

mat_mu_est = np.zeros((K, p_gamma))
mat_gamma_est = np.zeros((K, p_gamma))

mat_D_X_cur = np.zeros((n, p_gamma + 1))
vec_theta_est = np.zeros(p_gamma + 1)

for j in range(K): 
    mat_mu_est[j] = np.linalg.solve(arr_X[j].T @ arr_X[j], arr_X[j].T @ mat_D[:, j])

    mat_D_X_cur = np.column_stack((mat_D[:, j], arr_X[j]))
    vec_theta_est = np.linalg.solve(mat_D_X_cur.T @ mat_D_X_cur, mat_D_X_cur.T @ mat_Y[:, j])

    vec_beta_est_local[j] = vec_theta_est[0]
    mat_gamma_est[j] = vec_theta_est[1:]

beta_est_ini = np.mean(vec_beta_est_local)

print( (mat_gamma_est - mat_gamma) / np.abs(mat_gamma_est) )
print( (mat_mu_est - mat_mu) / np.abs(mat_mu_est) )



## statistics from other sites
vec_s_cur = np.zeros(n)
vec_S_est = np.zeros(K)

for j in range(K): 
    vec_s_cur = (mat_D[:, j] - arr_X[j] @ mat_mu_est[j]) \
        * (mat_Y[:, j] - mat_D[:, j] * beta_est_ini - arr_X[j] @ mat_gamma_est[j])
    vec_S_est[j] = np.mean(vec_s_cur)

S = np.mean(vec_S_est)

## oracle statistics from other sites
vec_s_ora_cur = np.zeros(n)
vec_S_ora_est = np.zeros(K)

for j in range(K):
    vec_s_ora_cur = (mat_D[:, j] - arr_X[j] @ mat_mu[j]) \
        * (mat_Y[:, j] - mat_D[:, j] * beta - arr_X[j] @ mat_gamma[j])
    vec_S_ora_est[j] = np.mean(vec_s_ora_cur)

S_ora = np.mean(vec_S_ora_est)



## operations in central site
vec_beta_est_cen = np.zeros(K)

vec_beta_ora_est_cen = np.zeros(K)

for j_cen in range(K): 
    vec_Y_cen = np.zeros(n)
    vec_D_cen = np.zeros(n)
    mat_X_cen = np.zeros((n, p_gamma))

    vec_Y_res_cur = np.zeros(n)
    vec_Y_res_cen = np.zeros(n)

    mat_G_slope = np.zeros((n, K))

    vec_Y_cen = mat_Y[:, j_cen]
    vec_D_cen = mat_D[:, j_cen]
    mat_X_cen = arr_X[j_cen]

    for j in range(K): 
        vec_Y_res_cur = vec_Y_cen - vec_D_cen * beta_est_ini - mat_X_cen @ mat_gamma_est[j]
        vec_Y_res_cen = vec_Y_cen - vec_D_cen * beta_est_ini - mat_X_cen @ mat_gamma_est[j_cen]

        mat_G_slope[:, j] = np.power(vec_Y_res_cur, 2) - np.power(vec_Y_res_cen, 2)
        
        mat_G_slope[:, j] = np.exp(-.5 * psi_u_inv * mat_G_slope[:, j])

        print("density ratio: ", np.mean(mat_G_slope[:, j])) ## density ratio

        mat_G_slope[:, j] *= (vec_D_cen - mat_X_cen @ mat_mu_est[j]) * vec_D_cen

        print(" " * 40, "G slope: ", np.mean(mat_G_slope[:, j]))

    G_slope = np.mean(mat_G_slope)


    ## final estimation
    beta_est_cen = beta_est_ini + S / G_slope
    vec_beta_est_cen[j_cen] = beta_est_cen

    beta_ora_est_cen = beta_est_ini + S_ora / G_slope
    vec_beta_ora_est_cen[j_cen] = beta_ora_est_cen

beta_est = np.mean(vec_beta_est_cen)

beta_ora_est = np.mean(vec_beta_ora_est_cen)

## output
print("S estimation:       ", S)
print("S ora estimation:   ", S_ora)
print("local estimation:   ", vec_beta_est_local)
print("central estimation: ", vec_beta_est_cen)
print("initial estimation: ", beta_est_ini)
print("final estimation:   ", beta_est)
print("oracle esitmation:  ", beta_ora_est)

