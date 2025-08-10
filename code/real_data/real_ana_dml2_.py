
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

## square of Neyman orthogonal score function
mat_ns2_local = np.zeros((K, K_fold))
mat_ns_slp_local = np.zeros((K, K_fold))

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

        mat_ns2_local[j, splt] = np.mean(
            np.power(vec_D_diff * (vec_Y_diff - vec_D_diff * beta_est_local), 2.0)
        )
        mat_ns_slp_local[j, splt] = np.mean(
            -np.power(vec_D_diff, 2.0)
        )       

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

vec_var_local = mat_ns2_local.mean(1) / np.power(mat_ns_slp_local.mean(1), 2.0)

print("var of (weighted) average:   ", np.sum(np.power(vec_weight, 2.0) * vec_var_local))
print("var of (unweighted) average: ", np.mean(vec_var_local) / K)

