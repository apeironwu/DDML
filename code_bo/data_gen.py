# -*- coding: utf-8 -*-
import numpy as np


def fun_expit(x):
    return 1 / (1 + np.exp(-x))

def data_gen(n, p, beta, ver = 0): 
    if ver == 0:
        mat_x = np.random.randn(n, p)
        vec_d = mat_x[:, 0] + np.random.randn(n)
        vec_y_prob = fun_expit(0.5 + vec_d * beta + 1 * mat_x[:, 1])
        vec_y = np.random.binomial(1, vec_y_prob) 

    elif ver == 1:
        mat_x = np.random.randn(n, p)
        vec_d = mat_x[:, 0] + np.random.randn(n)
        vec_y_prob = fun_expit(vec_d * beta + 1 * mat_x[:, 1])
        vec_y = np.random.binomial(1, vec_y_prob)

    elif ver == 2: 
        mat_x = np.random.randn(n, p)
        vec_d = mat_x[:, 0] + np.random.randn(n)
        vec_y_prob = fun_expit(vec_d * beta + fun_expit(mat_x[:, 1]))
        vec_y = np.random.binomial(1, vec_y_prob)
    
    else: 
        exit("Invalid version")

    return vec_y, vec_d, mat_x



## binary treatment 
def data_gen_bt(n, p, beta, ver = 0): 
    if ver == 0: 
        mat_x = np.random.randn(n, p)
        vec_d_prob = fun_expit(-1.0 + fun_expit(mat_x[:, 0]) + 0.25 * fun_expit(mat_x[:, 1]))
        vec_d = np.random.binomial(1, vec_d_prob)
        vec_y_prob = fun_expit(-0.5 + vec_d * beta + mat_x[:, 1])
        vec_y = np.random.binomial(1, vec_y_prob)

        # print(">> Pr(d=1), Pr(y=1): ", np.mean(vec_d).round(3), np.mean(vec_y).round(3))

    elif ver == 1:
        vec_d = np.random.binomial(1, 0.3, n)
        mat_x = np.random.randn(n, p)
        mat_x[:, 0] = mat_x[:, 0] + 0.5 * vec_d - 1.0
        vec_y_prob = fun_expit(-1.0 + vec_d * beta + mat_x[:, 1])
        vec_y = np.random.binomial(1, vec_y_prob) 

        # print(">> Pr(d=1), Pr(y=1): ", np.mean(vec_d).round(3), np.mean(vec_y).round(3))
    
    else: 
        exit("Invalid version")

    return vec_y, vec_d, mat_x




def data_gen_bt_het(n, p, beta, ver = 0, gam0 = 0.25, gam1 = 1.0): 
    if ver == 0: 
        gam0 = gam0 * 2 - 2
        gam1 = gam1 * 1 - 2

        mat_x = np.random.randn(n, p)
        vec_d_prob = fun_expit(-1.0 + 0.5 * mat_x[:, 0] - 1.0 * fun_expit(mat_x[:, 1]))
        vec_d = np.random.binomial(1, vec_d_prob)
        vec_y_prob = fun_expit(
            gam0 + vec_d * beta - 1.0 * fun_expit(mat_x[:, 0]) + gam1 * mat_x[:, 1]
        )
        vec_y = np.random.binomial(1, vec_y_prob)


    elif ver == 1:
        gam0 = gam0 - 1
        gam1 = gam1 * 2 - 4

        vec_d = np.random.binomial(1, 0.1, n)
        mat_x = np.random.randn(n, p)
        mat_x[:, 0] = mat_x[:, 0] + vec_d - 0.3
        vec_y_prob = fun_expit(
            gam0 + vec_d * beta + gam1 * fun_expit(mat_x[:, 0])
        )
        vec_y = np.random.binomial(1, vec_y_prob) 

    
    else: 
        exit("Invalid version")

    return vec_y, vec_d, mat_x
