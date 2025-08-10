import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
import time
import sys, getopt
import math
from sklearn.model_selection import GridSearchCV


import os

os.chdir('/lustre/project/Stat/s1155168529/programs/DDML/code/real_data')
os.getcwd()

## set parameters
n_iter = 1
K = 3  #number of sites 
K_fold = 3
n_rft = 100

ls_out = list()

vec_beta_est_iter = np.zeros(n_iter)

## loading data
adni1_pd = pd.read_csv('adni1.csv')
adni2_pd = pd.read_csv('adni2.csv')
adnigo_pd = pd.read_csv('adnigo.csv')

adni1 = adni1_pd.values
adni2 = adni2_pd.values
adnigo = adnigo_pd.values

df = [adni1, adni2, adnigo]

## set random seed
random.seed(2025)
np.random.seed(2025)

vec_n = np.zeros(K)

for j in range(K):
    vec_n[j] = int(df[j].shape[0])

vec_n = vec_n.astype(int)

# print(vec_n)

## data splitting
array1 = [0] * vec_n[0]
array2 = [0] * vec_n[1]
array3 = [0] * vec_n[2]

# Creating a list to contain these arrays
idx_K_fold = [array1, array2, array3]

for j in range(K):
    idx_K_fold[j] = np.random.choice(range(K_fold), vec_n[j], replace=True)

i_iter = 0

list_gamma_est = list()
list_mu_est = list()

mat_beta_est_local = np.zeros((K, K_fold))

## local estimation ==========================================
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
        
        model_mu = RandomForestRegressor(n_estimators=n_rft)
        model_mu.fit(mat_X_nui, vec_D_nui)
                        
        ## estimation of beta based on partialling out score function
        model_xi = RandomForestRegressor(n_estimators=n_rft)
        model_xi.fit(mat_X_nui, vec_Y_nui)
                        
        vec_D_diff = vec_D_est - model_mu.predict(mat_X_est)
        vec_Y_diff = vec_Y_est - model_xi.predict(mat_X_est)
                        
        beta_est_local = np.mean(vec_Y_diff * vec_D_diff) / np.mean(vec_D_diff * vec_D_diff)

        for _ in range(2): 
            ## update nuisance parameter
            model_gamma = RandomForestRegressor(n_estimators=n_rft)
            model_gamma.fit(mat_X_nui, vec_Y_nui - vec_D_nui * beta_est_local)

            vec_Y_gam_diff = vec_Y_est - model_gamma.predict(mat_X_est)
            beta_est_local = np.mean(vec_Y_gam_diff * vec_D_diff) / np.mean(vec_D_est * vec_D_diff)

        list_mu_est.append(model_mu)
        list_gamma_est.append(model_gamma)

        mat_beta_est_local[j, splt] = beta_est_local

# print(">> mat_beta_est_local: \n", mat_beta_est_local)

vec_beta_est_local = mat_beta_est_local.mean(axis=1)

print(">> vec_beta_est_local: ", vec_beta_est_local.round(4))

vec_beta_var_local = np.zeros(K)

## variance estimation ===========================================
for j in range(K): 
    var_beta_score2 = 0.0
    var_beta_j0 = 0.0

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

        model_mu = list_mu_est[splt + j * K_fold]
        model_gamma = list_gamma_est[splt + j * K_fold]

        vec_D_diff = vec_D_est - model_mu.predict(mat_X_est)
        vec_Y_gam_diff = vec_Y_est - model_gamma.predict(mat_X_est)

        var_beta_score2 += np.sum(
            (
                (vec_Y_gam_diff - vec_D_est * vec_beta_est_local[j]) * vec_D_diff
            ) ** 2
        )
        var_beta_j0 += np.sum(vec_D_est * vec_D_diff)
    
    var_beta_score2 /= vec_n[j]
    var_beta_j0 /= vec_n[j]

    vec_beta_var_local[j] = var_beta_score2 / (var_beta_j0 ** 2) / vec_n[j]

## naive average ===========================================
vec_weight = 1.0 / K

ls_out.append(
    {
        "method": "ldml_avg", 
        "value": vec_beta_est_local.mean().round(4), 
        "sd": np.sqrt(np.sum(vec_weight ** 2 * vec_beta_var_local)).round(4)
    }
)

## inverse variance weighted average ===========================================
vec_weight_iv = 1.0 / vec_beta_var_local
vec_weight_iv = vec_weight_iv / np.sum(vec_weight_iv)

beta_est_iv = np.sum(vec_weight_iv * vec_beta_est_local)

ls_out.append(
    {
        "method": "ldml_avg_iv", 
        "value": beta_est_iv.round(4), 
        "sd": np.sqrt(np.sum(vec_weight_iv ** 2 * vec_beta_var_local)).round(4)
    }
)

vec_weight_ss = vec_n / np.sum(vec_n)
beta_est_ss = np.sum(vec_weight_ss * vec_beta_est_local)

ls_out.append(
    {
        "method": "ldml_avg_ss", 
        "value": beta_est_ss.round(4), 
        "sd": np.sqrt(np.sum(vec_weight_ss ** 2 * vec_beta_var_local)).round(4)
    }
)


## double machine learning ===========================================

### initial value
beta_est_ini = np.sum(vec_weight_ss * vec_beta_est_local)

### output
mat_beta_est_cen = np.zeros((K, K_fold))

for splt in range(K_fold): 
    vec_S_est = np.zeros(K)

    for j in range(K):
        idx_est = np.where(idx_K_fold[j] == splt)[0]

        n_est = idx_est.shape[0]

        ## statistics from other sites
        vec_s = np.zeros(n_est)
        idx_ls = j * K_fold + splt 

        ## estimating
        mat_X_est = df[j][idx_est, 6:20]
        vec_D_est = df[j][idx_est, 22]
        vec_Y_est = df[j][idx_est, 21]

        vec_D_diff = vec_D_est - list_mu_est[idx_ls].predict(mat_X_est)
        vec_Y_diff = vec_Y_est - list_gamma_est[idx_ls].predict(mat_X_est)
                        
        vec_s = (vec_Y_diff - vec_D_est * beta_est_ini) * vec_D_diff
                        
        vec_S_est[j] = np.mean(vec_s)
    
    S = np.sum(vec_S_est * vec_weight_ss)
    
    for j_cen in range(K):
        idx_est = np.where(idx_K_fold[j] == splt)[0]

        n_est = idx_est.shape[0]
        
        idx_ls_cen = j_cen * K_fold + splt

        vec_Y_cen = df[j_cen][idx_est, 21]
        vec_D_cen = df[j_cen][idx_est, 22]
        mat_X_cen = df[j_cen][idx_est, 6:20]

        mat_U_slope = np.zeros((n_est, K))

        # single-equation density estimation
        vec_Y_cen_diff = vec_Y_cen - list_gamma_est[idx_ls_cen].predict(mat_X_cen)
        vec_U_cen_est = vec_Y_cen_diff - vec_D_cen * beta_est_ini

        for j in range(K):
            idx_ls = j * K_fold + splt

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
            mat_U_slope[:, j] = mat_U_slope[:, j] * vec_weight_ss[j]

        U_slope = np.mean(mat_U_slope.sum(1))
        
        beta_est_cen = beta_est_ini + S / U_slope
        mat_beta_est_cen[j_cen, splt] = beta_est_cen

vec_fdml_cen = mat_beta_est_cen.mean(axis=1)
print(">> vec_beta_est_fdml: \n", vec_fdml_cen.round(4))


## double machine learning variance estimation ===========================================

vec_beta_var_fl = np.zeros(K)

for j_cen in range(K): 
    vec_var_beta_score2 = np.zeros(K)
    vec_var_beta_j0 = np.zeros(K)

    for splt in range(K_fold):
        idx_ls_cen = j_cen * K_fold + splt

        idx_est = np.where(idx_K_fold[j_cen] == splt)[0]
        n_est = idx_est.shape[0]

        vec_Y_cen = df[j_cen][idx_est, 21]
        vec_D_cen = df[j_cen][idx_est, 22]
        mat_X_cen = df[j_cen][idx_est, 6:20]

        vec_Y_cen_diff = vec_Y_cen - list_gamma_est[idx_ls_cen].predict(mat_X_cen)
        vec_D_cen_diff = vec_D_cen - list_mu_est[idx_ls_cen].predict(mat_X_cen)
        vec_U_cen_est = vec_Y_cen_diff - vec_D_cen * vec_fdml_cen[j_cen]

        var_U_cen = np.mean(np.power(vec_U_cen_est, 2))

        mat_score2_cur = np.zeros((n_est, K))
        mat_j0_cur = np.zeros((n_est, K))

        for j in range(K): 
            idx_ls = j * K_fold + splt
            
            vec_D_loc_diff = vec_D_cen - list_mu_est[idx_ls].predict(mat_X_cen)
            vec_Y_loc_diff = vec_Y_cen - list_gamma_est[idx_ls].predict(mat_X_cen)
            vec_U_loc_est = vec_Y_loc_diff - vec_D_cen * vec_fdml_cen[j_cen]

            var_U_loc = np.mean(np.power(vec_U_loc_est, 2))
            
            ## density ratio estimation
            if j == j_cen: 
                mat_score2_cur[:, j] = np.ones(n_est)
            else: 
                mat_score2_cur[:, j] = np.power(vec_U_loc_est, 2.0) / (2.0 * var_U_loc)
                mat_score2_cur[:, j] = mat_score2_cur[:, j] - np.power(vec_U_cen_est, 2.0) / (2.0 * var_U_cen)
                mat_score2_cur[:, j] = np.exp(-1.0 * mat_score2_cur[:, j])
                mat_score2_cur[:, j] = mat_score2_cur[:, j] * math.sqrt(var_U_cen) / math.sqrt(var_U_loc)
            
            mat_j0_cur[:, j] = mat_score2_cur[:, j]

            ## score2
            mat_score2_cur[:, j] = mat_score2_cur[:, j] * np.power(
                (vec_D_loc_diff * (vec_Y_loc_diff - vec_fdml_cen[j_cen] * vec_D_cen)), 2.0
            )

            ## j0
            mat_j0_cur[:, j] = mat_j0_cur[:, j] * vec_D_cen * vec_D_loc_diff
        
        vec_var_beta_score2 += np.sum(mat_score2_cur, axis=0)
        vec_var_beta_j0 += np.sum(mat_j0_cur, axis=0)
    
    vec_var_beta_score2 /= vec_n
    vec_var_beta_j0 /= vec_n

    vec_beta_var_fl[j_cen] = np.sum(vec_weight_ss * vec_var_beta_score2) 
    vec_beta_var_fl[j_cen] /= (np.sum(vec_weight_ss * vec_var_beta_j0) ** 2) 
    vec_beta_var_fl[j_cen] /= vec_n[j_cen]

# print(">> vec_beta_var_fl: ", vec_beta_var_fl.round(4))
            
ls_out.append(
    {
        "method": "fdml_ss", 
        "value": np.sum(vec_fdml_cen * vec_weight_ss).round(4), 
        "sd": np.sqrt(np.sum(vec_weight_ss ** 2 * vec_beta_var_fl)).round(4)
    }
)











## output results
for item in ls_out: 
    print(f">> {item['method']}: {item['value']} ({item['sd']})")










