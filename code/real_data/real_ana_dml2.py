
## loading package
import numpy as np
import pandas as pd
import random
import pickle
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
import time
import sys, getopt
import math
from sklearn.model_selection import GridSearchCV


## set path
import os
os.chdir('/lustre/project/Stat/s1155168529/programs/DDML/code/real_data')
os.getcwd()


## set random seed
np.random.seed(2024)
random.seed(2024)


## set parameter
n_iter = 1
K      = 3  #number of sites 
K_fold = 3
n_rft  = 50


## load data
vec_beta_est_iter = np.zeros(n_iter)

adni1_pd  = pd.read_csv('adni1.csv')
adni2_pd  = pd.read_csv('adni2.csv')
adnigo_pd = pd.read_csv('adnigo.csv')

adni1  = adni1_pd.values
adni2  = adni2_pd.values
adnigo = adnigo_pd.values

## concatenate data
df = [adni1, adni2, adnigo]


## nuisance parameter estimation
model = RandomForestRegressor()

# pickle.dump(
#    [param_d, param_y], 
#    open("rf_set.pydata", "wb")
# )

## loading
param_d, param_y = pickle.load(open("rf_set.pydata", "rb"))


# sample size
n1 = len(adni1)
n2 = len(adni2)
n3 = len(adnigo)

vec_n = np.array([n1, n2, n3])



# weight 
vec_weight = [n1, n2, n3]
vec_weight = vec_weight / np.sum(vec_weight)



## data splitting --------------------------------------------------------

# Creating empty arrays with 0 rows but defined number of columns
array1 = [0] * n1
array2 = [0] * n2
array3 = [0] * n3

# Creating a list to contain these arrays
idx_K_fold = [array1, array2, array3]
idx_K_fold[0][0]


np.random.seed(1)
idx_K_fold[0] = np.random.choice(range(K_fold), n1, replace=True)

np.random.seed(1)
idx_K_fold[1] = np.random.choice(range(K_fold), n2, replace=True)

np.random.seed(1)
idx_K_fold[2] = np.random.choice(range(K_fold), n3, replace=True)


## local DML estimation --------------------------------------------------

i_iter = 0
list_gamma_est = list() ## (Y-D*beta)~X
list_mu_est    = list() ## D~X
list_xi_est    = list() ## Y~X

mat_beta_est_local = np.zeros((K, K_fold))
for j in range(K):
    for splt in range(K_fold):
        idx_est = np.where(idx_K_fold[j] == splt)[0]
        idx_nui = np.where(idx_K_fold[j] != splt)[0]
        n_est = idx_est.shape[0]

        ## training ML model
        mat_X_nui = df[j][idx_nui, 6:20]
        vec_D_nui = df[j][idx_nui, 22]
        vec_Y_nui = df[j][idx_nui, 21]
                        
        ## estimating
        mat_X_est = df[j][idx_est, 6:20]
        vec_D_est = df[j][idx_est, 22]
        vec_Y_est = df[j][idx_est, 21]
        
        model_mu = RandomForestRegressor(**param_d[j])
        model_mu.fit(mat_X_nui, vec_D_nui)
                        
        ## estimation of beta based on partialling out score function
        model_xi = RandomForestRegressor(**param_y[j])
        model_xi.fit(mat_X_nui, vec_Y_nui)
                        
        vec_D_diff = vec_D_est - model_mu.predict(mat_X_est)
        vec_Y_diff = vec_Y_est - model_xi.predict(mat_X_est)
                        
        beta_est_local = np.mean(vec_Y_diff * vec_D_diff) / np.mean(vec_D_diff * vec_D_diff)

        model_gamma = RandomForestRegressor(n_estimators=50)
        model_gamma.fit(mat_X_nui, vec_Y_nui - vec_D_nui * beta_est_local)

        list_xi_est.append(model_xi)
        list_mu_est.append(model_mu)
        list_gamma_est.append(model_gamma)

        mat_beta_est_local[j, splt] = beta_est_local

print(
    "mat_beta_est_local: ", 
    mat_beta_est_local.mean(1)
)
print(
    "beta_est_int (weighted):   ", 
    np.sum(mat_beta_est_local.mean(1) * vec_weight)
)
print(
    "beta_est_ini (unweighted): ", 
    mat_beta_est_local.mean()
)


## estimation of variance ----

## square of Neyman orthogonal score function
mat_ns2_local = np.zeros((K, K_fold))
mat_ns_slp_local = np.zeros((K, K_fold))

## local estimation
vec_beta_est_local = mat_beta_est_local.mean(1)

for j in range(K):
    for splt in range(K_fold):
        idx_ls = splt + j * K_fold

        idx_est = np.where(idx_K_fold[j] == splt)[0]
        n_est = idx_est.shape[0]

        ## estimating
        mat_X_est = df[j][idx_est, 6:20]
        vec_D_est = df[j][idx_est, 22]
        vec_Y_est = df[j][idx_est, 21]
        
        model_mu = list_mu_est[idx_ls]
        model_xi = list_xi_est[idx_ls]
        
        vec_D_diff = vec_D_est - model_mu.predict(mat_X_est)
        vec_Y_diff = vec_Y_est - model_xi.predict(mat_X_est)

        mat_ns2_local[j, splt] = np.mean(
            np.power(vec_D_diff * (vec_Y_diff - vec_D_diff * vec_beta_est_local[j]), 2.0)
        )
        mat_ns_slp_local[j, splt] = np.mean(
            -np.power(vec_D_diff, 2.0)
        )

vec_var_local = mat_ns2_local.mean(1) / np.power(mat_ns_slp_local.mean(1), 2.0) / vec_n

print(">>>> sd: ", np.sqrt(vec_var_local))
print(">>>> sd of (weighted) average:   ", np.sqrt(np.sum(np.power(vec_weight, 2.0) * vec_var_local)))
print(">>>> sd of (unweighted) average:   ", np.sqrt(np.mean(vec_var_local) / K))



print("==================== [ DDML ] ====================")
## DDML ------------------------------------------------------------------
beta_est_ini = np.sum(mat_beta_est_local.mean(1) * vec_weight)

vec_beta_est_cen = np.zeros(K)
vec_S_split = np.zeros(K_fold)
mat_U_slope_cen_split = np.zeros((K, K_fold))


for splt in range(K_fold): 
    for j in range(K):
        idx_est = np.where(idx_K_fold[j] == splt)[0]

        n_est = idx_est.shape[0]

        ## statistics from other sites
        vec_s = np.zeros(n_est)
        vec_S_est = np.zeros(K)
        idx_ls = splt + j * K_fold

        ## estimating
        mat_X_est = df[j][idx_est, 6:20]
        vec_D_est = df[j][idx_est, 22]
        vec_Y_est = df[j][idx_est, 21]

        vec_D_diff = vec_D_est - list_mu_est[idx_ls].predict(mat_X_est)
        vec_Y_diff = vec_Y_est - list_gamma_est[idx_ls].predict(mat_X_est)
                        
        vec_s = (vec_Y_diff - vec_D_est * beta_est_ini) * vec_D_diff
                        
        vec_S_est[j] = np.mean(vec_s)
    
    # S = np.mean(vec_S_est) ## naive average
    S = np.sum(vec_S_est * vec_weight) ## weighted average
    vec_S_split[splt] = S
    
    for j_cen in range(K):
        idx_est = np.where(idx_K_fold[j] == splt)[0]

        n_est = idx_est.shape[0]
        
        idx_ls_cen = splt + j_cen * K_fold

        vec_Y_cen = df[j_cen][idx_est, 21]
        vec_D_cen = df[j_cen][idx_est, 22]
        mat_X_cen = df[j_cen][idx_est, 6:20]

        mat_U_slope = np.zeros((n_est, K))

        # single-equation density estimation
        vec_Y_cen_diff = vec_Y_cen - list_gamma_est[idx_ls_cen].predict(mat_X_cen)

        vec_U_cen_est = vec_Y_cen_diff - vec_D_cen * beta_est_ini
        for j in range(K):
            idx_ls = splt + j * K_fold

            vec_D_loc_diff = vec_D_cen - list_mu_est[idx_ls].predict(mat_X_cen)

            vec_Y_loc_diff = vec_Y_cen - list_gamma_est[idx_ls].predict(mat_X_cen)
                            
            vec_U_loc_est = vec_Y_loc_diff - vec_D_cen * beta_est_ini
                            
            ## density ratio
            varU_loc = np.mean(np.power(vec_U_loc_est, 2))
            varU_cen = np.mean(np.power(vec_U_cen_est, 2))
            mat_U_slope[:, j] = np.power(vec_U_loc_est, 2)/(2*varU_loc) - np.power(vec_U_cen_est, 2)/(2*varU_cen)
            mat_U_slope[:, j] = np.exp(-1* mat_U_slope[:, j])
            mat_U_slope[:, j] = mat_U_slope[:, j] * math.sqrt(varU_cen)/math.sqrt(varU_loc)
            mat_U_slope[:, j] = mat_U_slope[:, j] * vec_D_cen * vec_D_loc_diff
            mat_U_slope[:, j] = mat_U_slope[:, j] * vec_weight[j]
        U_slope = np.mean(mat_U_slope.sum(1))

        mat_U_slope_cen_split[j_cen, splt] = U_slope


# print("vec_S: ", vec_S_split)
# print("mat_U_slope: ", mat_U_slope_cen_split)

vec_beta_est_cen = beta_est_ini + np.mean(vec_S_split) / mat_U_slope_cen_split.mean(1)
print(">>>> (Weighted) vec_beta_est_cen average over data splitting: ", vec_beta_est_cen)
print(">>>> (Weighted) DDML estimator: ", np.sum(vec_beta_est_cen * vec_weight))


## square of Neyman orthogonal score function
beta_est = np.sum(vec_beta_est_cen * vec_weight)

mat_ns2_cen    = np.zeros((K, K_fold))
mat_ns_slp_cen = np.zeros((K, K_fold))

for j in range(K):
    for splt in range(K_fold):
        idx_ls = splt + j * K_fold

        idx_est = np.where(idx_K_fold[j] == splt)[0]
        n_est = idx_est.shape[0]

        ## estimating
        mat_X_est = df[j][idx_est, 6:20]
        vec_D_est = df[j][idx_est, 22]
        vec_Y_est = df[j][idx_est, 21]
        
        model_mu = list_mu_est[idx_ls]
        model_xi = list_xi_est[idx_ls]
        
        vec_D_diff = vec_D_est - model_mu.predict(mat_X_est)
        vec_Y_diff = vec_Y_est - model_xi.predict(mat_X_est)

        mat_ns2_cen[j, splt] = np.mean(
            np.power(vec_D_diff * (vec_Y_diff - vec_D_diff * beta_est), 2.0)
        )
        mat_ns_slp_cen[j, splt] = np.mean(
            -np.power(vec_D_diff, 2.0)
        )

# print("=============================")
# print("mat_ns2_local: ",  mat_ns2_local)
# print("=============================")
# print("mat_ns2_cen: ",    mat_ns2_cen)
# print("mat_ns_slp_cen: ", mat_ns_slp_cen)
# print("=============================")
# print("vec_s2_local: ",   mat_ns2_local.mean(1))
# print("vec_s2_cen: ",     mat_ns2_cen.mean(1))
# print("vec_js_cen: ",     mat_ns_slp_cen.mean(1))
# print("=============================")
# print("s2_cen: ", np.sum(vec_weight * mat_ns2_cen.mean(1)))
# print("js_cen: ", np.sum(vec_weight * mat_ns_slp_cen.mean(1)))
# print("=============================")


var_ddml = np.sum(vec_weight * mat_ns2_cen.mean(1)) / np.power(np.sum(vec_weight * mat_ns_slp_cen.mean(1)), 2.0) / np.sum(vec_n)

print(">>>> sd of ddml:   ", np.sqrt(var_ddml))





## unified data

vec_beta_uni_est = np.zeros(K_fold)

vec_Y_diff_comb = np.zeros(vec_n.sum())
vec_D_diff_comb = np.zeros(vec_n.sum())



for splt in range(K_fold):
    idx_est1 = np.where(idx_K_fold[0] == splt)[0]
    idx_nui1 = np.where(idx_K_fold[0] != splt)[0]
    n_est1 = idx_est1.shape[0]
    idx_est2 = np.where(idx_K_fold[1] == splt)[0]
    idx_nui2 = np.where(idx_K_fold[1] != splt)[0]
    n_est2 = idx_est2.shape[0]
    idx_est3 = np.where(idx_K_fold[2] == splt)[0]
    idx_nui3 = np.where(idx_K_fold[2] != splt)[0]
    n_est3 = idx_est3.shape[0]

    ## training ML model
    mat_X_nui1 = df[0][idx_nui1, 6:20]
    vec_D_nui1 = df[0][idx_nui1, 22]
    vec_Y_nui1 = df[0][idx_nui1, 21]
    
    mat_X_nui2 = df[1][idx_nui2, 6:20]
    vec_D_nui2 = df[1][idx_nui2, 22]
    vec_Y_nui2 = df[1][idx_nui2, 21]
    
    mat_X_nui3 = df[2][idx_nui3, 6:20]
    vec_D_nui3 = df[2][idx_nui3, 22]
    vec_Y_nui3 = df[2][idx_nui3, 21]
                    
    ## estimating
    mat_X_est1 = df[0][idx_est1, 6:20]
    vec_D_est1 = df[0][idx_est1, 22]
    vec_Y_est1 = df[0][idx_est1, 21]
    
    mat_X_est2 = df[1][idx_est2, 6:20]
    vec_D_est2 = df[1][idx_est2, 22]
    vec_Y_est2 = df[1][idx_est2, 21]
    
    mat_X_est3 = df[2][idx_est3, 6:20]
    vec_D_est3 = df[2][idx_est3, 22]
    vec_Y_est3 = df[2][idx_est3, 21]
    
    
    model_mu1 = RandomForestRegressor(**param_d[0])
    model_mu1.fit(mat_X_nui1, vec_D_nui1)
    
    model_mu2 = RandomForestRegressor(**param_d[1])
    model_mu2.fit(mat_X_nui2, vec_D_nui2)
    
    model_mu3 = RandomForestRegressor(**param_d[2])
    model_mu3.fit(mat_X_nui3, vec_D_nui3)
                    
    ## estimation of beta based on partialling out score function
    model_xi1 = RandomForestRegressor(**param_y[0])
    model_xi1.fit(mat_X_nui1, vec_Y_nui1)
    
    model_xi2 = RandomForestRegressor(**param_y[1])
    model_xi2.fit(mat_X_nui2, vec_Y_nui2)
    
    model_xi3 = RandomForestRegressor(**param_y[2])
    model_xi3.fit(mat_X_nui3, vec_Y_nui3)

    vec_D_diff1 = vec_D_est1 - model_mu1.predict(mat_X_est1)
    vec_Y_diff1 = vec_Y_est1 - model_xi1.predict(mat_X_est1)
    
    vec_D_diff2 = vec_D_est2 - model_mu2.predict(mat_X_est2)
    vec_Y_diff2 = vec_Y_est2 - model_xi2.predict(mat_X_est2)
    
    vec_D_diff3 = vec_D_est3 - model_mu3.predict(mat_X_est3)
    vec_Y_diff3 = vec_Y_est3 - model_xi3.predict(mat_X_est3)

    vec_Y_diff_comb[idx_est1] = vec_Y_diff1
    vec_Y_diff_comb[idx_est2 + n1] = vec_Y_diff2
    vec_Y_diff_comb[idx_est3 + n1 + n2] = vec_Y_diff3

    vec_D_diff_comb[idx_est1] = vec_D_diff1
    vec_D_diff_comb[idx_est2 + n1] = vec_D_diff2
    vec_D_diff_comb[idx_est3 + n1 + n2] = vec_D_diff3
    
    vec_Y_diff = np.concatenate((vec_Y_diff1, vec_Y_diff2, vec_Y_diff3))
    vec_D_diff = np.concatenate((vec_D_diff1, vec_D_diff2, vec_D_diff3))

    beta_est_split = np.mean(vec_Y_diff * vec_D_diff) / np.mean(vec_D_diff * vec_D_diff)

    vec_beta_uni_est[splt] = beta_est_split


print("")
print("==================== [ Unified Data ] ====================")

print("vec_beta_uni_est: ", vec_beta_uni_est)
print("(weighted) unified estimator: ", np.mean(vec_beta_uni_est))

beta_nui_est = np.mean(vec_beta_uni_est)

print("vec_D_diff_comb: ", vec_D_diff_comb)
print("vec_Y_diff_comb: ", vec_Y_diff_comb)

vec_ns2_uni = -np.power(vec_D_diff_comb, 2.0) * beta_nui_est 
vec_ns2_uni += vec_D_diff_comb * vec_Y_diff_comb
vec_ns2_uni = np.power(vec_ns2_uni, 2.0)

vec_ns_slp_uni = np.power(vec_D_diff_comb, 2.0)

var_uni = vec_ns2_uni.mean() / np.power(vec_ns_slp_uni.mean(), 2.0) / vec_n.sum()

print("sd_uni: ", np.sqrt(var_uni))



