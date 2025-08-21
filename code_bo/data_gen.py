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
    
    # elif ver == 3:
    #     mat_x = np.random.randn(n, p)
    #     vec_d_prob = fun_expit(mat_x[:, 0])
    #     vec_d = np.random.binomial(1, vec_d_prob)
    #     vec_y_prob = fun_expit(0.5 + vec_d * beta + fun_expit(mat_x[:, 1]))
    #     vec_y = np.random.binomial(1, vec_y_prob)

    # elif ver == 4:
    #     mat_x = np.random.randn(n, p)
    #     vec_d_prob = fun_expit(0.5 + mat_x[:, 0])
    #     vec_d = np.random.binomial(1, vec_d_prob)
    #     vec_y_prob = fun_expit(0.5 + vec_d * beta + fun_expit(mat_x[:, 1]))
    #     vec_y = np.random.binomial(1, vec_y_prob)
    
    else: 
        exit("Invalid version")

    return vec_y, vec_d, mat_x
