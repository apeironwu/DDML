# -*- coding: utf-8 -*-
from sympy import false
import torch
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

## for testing
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

device = torch.device("cpu")

def fun_expit(x):
    return 1 / (1 + np.exp(-x))

def fun_logit(x, eps = 1e-2):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))

def fun_logit_inflat(x, eps = 1e-2, inflat = 1.0):
    x = np.clip(x, eps, 1-eps)
    x_logit = np.log(x / (1 - x))

    x_logit[x_logit == np.min(x_logit)] = np.min(x_logit[x_logit != np.min(x_logit)]) * inflat
    x_logit[x_logit == np.max(x_logit)] = np.max(x_logit[x_logit != np.max(x_logit)]) * inflat

    return x_logit



class DataSite(torch.utils.data.Dataset):
    def __init__(self, vec_y, vec_d, mat_x, K_fold):
        self.vec_y = np.array(vec_y, dtype=np.int8)     ## input outcome (binary, int)
        self.vec_d = np.array(vec_d, dtype=np.float32)  ## input treatment
        self.mat_x = np.array(mat_x, dtype=np.float32)  ## input confounders

        ## data splitting
        self.K_fold = K_fold
        self.lab_ds = np.random.randint(0, K_fold, size=len(vec_y))

        self.ls_model_m = [None] * K_fold
        self.vec_m_est = np.zeros(len(vec_y), dtype=np.float32)

        ## double data splitting
        self.lab_dds = np.random.randint(0, K_fold, size = (len(vec_y), K_fold))

        self.ls_model_a = [None] * (K_fold ** 2)
        self.ls_model_M = [None] * (K_fold ** 2)
        self.vec_beta_check = np.zeros(K_fold, dtype=np.float32)

        self.mat_w = np.zeros((len(vec_y), K_fold), dtype=np.float32)
        self.mat_a_est = np.zeros((len(vec_y), K_fold), dtype=np.float32)
        self.ls_model_t = [None] * K_fold

        self.vec_gam_est = np.zeros(len(vec_y), dtype=np.float32)

    def __len__(self):
        return len(self.vec_y)

    def model_train_est(self):
        self.model_train_m()
        self.model_est_m()
        self.model_train_gam()
        self.model_est_gam()

    def get_fold_test(self, fold):
        mask = self.lab_ds == fold
        return self.vec_y[mask], self.vec_d[mask], self.mat_x[mask, :]
    
    def get_fold_train(self, fold):
        mask = self.lab_ds != fold
        return self.vec_y[mask], self.vec_d[mask], self.mat_x[mask, :]
    
    def model_train_m(self): 
        for k in range(self.K_fold):
            vec_y_cur, vec_d_cur, mat_x_cur = self.get_fold_train(k)
            mask_cur = vec_y_cur == 0

            model_m_cur = RandomForestRegressor(n_estimators=100)
            model_m_cur.fit(mat_x_cur[mask_cur, :], vec_d_cur[mask_cur])

            self.ls_model_m[k] = model_m_cur
    
    def model_est_m(self):
        for k in range(self.K_fold):
            _, _, mat_x_cur = self.get_fold_test(k)
            mask_cur = self.lab_ds == k

            self.vec_m_est[mask_cur] = self.ls_model_m[k].predict(mat_x_cur)

    def get_fold2_train_nn(self, fold1, fold2): 
        mask = (self.lab_ds != fold1) & (self.lab_dds[:, fold1] != fold2)
        return self.vec_y[mask], self.vec_d[mask], self.mat_x[mask, :]

    def get_fold2_train_np(self, fold1, fold2): 
        mask = (self.lab_ds != fold1) & (self.lab_dds[:, fold1] == fold2)
        return self.vec_y[mask], self.vec_d[mask], self.mat_x[mask, :], mask
    
    def get_idx_dds(self, fold1, fold2): 
        idx = fold1 * self.K_fold + fold2
        return idx

    def model_train_gam(self):
        for k in range(self.K_fold):
            for j in range(self.K_fold):
                vec_y_nn, vec_d_nn, mat_x_nn = self.get_fold2_train_nn(k, j)

                model_a_curr  = RandomForestRegressor(n_estimators=100)
                model_a_curr.fit(mat_x_nn, vec_d_nn)

                model_M_curr = RandomForestClassifier(n_estimators=100)
                model_M_curr.fit(
                    np.column_stack((vec_d_nn, mat_x_nn)), vec_y_nn
                )

                self.ls_model_a[self.get_idx_dds(k, j)] = model_a_curr
                self.ls_model_M[self.get_idx_dds(k, j)] = model_M_curr

                _, vec_d_np, mat_x_np, mask_np = self.get_fold2_train_np(k, j)

                self.mat_w[mask_np, k] = model_M_curr.predict_proba(
                    np.column_stack((vec_d_np, mat_x_np))
                )[:, 1]
                self.mat_a_est[mask_np, k] = model_a_curr.predict(mat_x_np)

            _, vec_d_train, mat_x_train = self.get_fold_train(k)
            # self.mat_w[:, k] = fun_logit(self.mat_w[:, k])

            mask_train = self.lab_ds != k

            model_t_cur = RandomForestRegressor(n_estimators=100)
            model_t_cur.fit(
                mat_x_train, 
                fun_logit(self.mat_w[mask_train, k], eps=1e-2)
            )

            self.ls_model_t[k] = model_t_cur

            # fig_lm_coef(
            #     vec_d_train - self.mat_a_est[mask_train, k], 
            #     fun_logit_inflat(self.mat_w[mask_train, k])
            # )

            self.vec_beta_check[k] = cal_lm_coef(
                vec_d_train - self.mat_a_est[mask_train, k], 
                fun_logit_inflat(self.mat_w[mask_train, k], inflat=1.0)
            )

    def model_est_gam(self):
        for k in range(self.K_fold):
            _, _, mat_x_cur = self.get_fold_test(k)
            mask_cur = self.lab_ds == k

            mat_a_est = np.zeros((mat_x_cur.shape[0], self.K_fold), dtype=np.float32)

            for j in range(self.K_fold):
                idx = self.get_idx_dds(k, j)
                mat_a_est[:, j] = self.ls_model_a[idx].predict(mat_x_cur)
            
            self.vec_gam_est[mask_cur] = self.ls_model_t[k].predict(mat_x_cur)
            self.vec_gam_est[mask_cur] -= self.vec_beta_check[k] * mat_a_est.mean(axis=1)
        

class ModelLogisticDML(torch.nn.Module):
    def __init__(self, vec_y, vec_d, vec_gam_est, vec_m_est):
        super().__init__()

        self.vec_y = torch.tensor(vec_y, dtype=torch.float32)  ## input outcome (binary, float)
        self.vec_d = torch.tensor(vec_d, dtype=torch.float32)  ## input treatment
        self.vec_gam_est = torch.tensor(vec_gam_est, dtype=torch.float32)  ## input confounders
        self.vec_m_est = torch.tensor(vec_m_est, dtype=torch.float32)      ## input confounders

        self.beta = torch.nn.Parameter(torch.tensor(0.0))
    
    def score(self): 
        score = 0.0
        score += torch.mean(
            (
                self.vec_y * torch.exp(
                    -self.beta * self.vec_d - self.vec_gam_est
                ) - (1.0 - self.vec_y) 
            ) * (self.vec_d - self.vec_m_est)
        )
        return score

    def score2(self): 
        score2 = self.score() ** 2
        return score2
    
    def var(self):
        I02 = torch.pow(
            torch.mean(
                self.vec_y * torch.exp(
                    -self.beta * self.vec_d - self.vec_gam_est
                ) * self.vec_d * (self.vec_d - self.vec_m_est)
            ), 2.0
        )

        Eh2 = torch.mean(
            torch.pow(
                (
                    self.vec_y * torch.exp(
                        -self.beta * self.vec_d - self.vec_gam_est
                    ) - (1.0 - self.vec_y)
                ) * (self.vec_d - self.vec_m_est),
                2.0
            )
        )

        return Eh2 / I02 / self.vec_y.__len__()

def train_logistic_dml(data, learning_rate):
    model = ModelLogisticDML(
        data.vec_y, data.vec_d, data.vec_gam_est, data.vec_m_est
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    eps = 1e-6
    bl_conv = False

    for epoch in range(10000):  # number of epochs

        optimizer.zero_grad()
        score2 = model.score2()
        score2.backward()
        optimizer.step()

        if score2 < eps:
            bl_conv = True
            # print(">> Converged at epoch {}".format(epoch))
            break

    return model, optimizer, bl_conv



class OperaSiteCen(torch.utils.data.Dataset):
    def __init__(self, idx_cen, ls_site):
        self.idx_cen = idx_cen
        self.S = ls_site.__len__()

        self.mat_m_est = np.zeros((ls_site[idx_cen].__len__(), self.S))
        self.mat_gam_est = np.zeros((ls_site[idx_cen].__len__(), self.S))
        self.mat_den_est = np.ones((ls_site[idx_cen].__len__(), self.S))

        self.operate_cen(ls_site)

    def operate_cen(self, ls_site): 

        site_cen = ls_site[self.idx_cen]

        for k in range(site_cen.K_fold):
            _, _, mat_x_cur = site_cen.get_fold_test(k)
            mask_cur = site_cen.lab_ds == k

            for s in range(self.S):
                self.mat_m_est[mask_cur, s] = ls_site[s].ls_model_m[k].predict(mat_x_cur)

                mat_a_cur = np.zeros((mat_x_cur.shape[0], site_cen.K_fold), dtype=np.float32)
                for j in range(site_cen.K_fold): 
                    idx = ls_site[s].get_idx_dds(k, j)
                    mat_a_cur[:, j] = ls_site[s].ls_model_a[idx].predict(mat_x_cur)
                
                self.mat_gam_est[mask_cur, s] = ls_site[s].ls_model_t[k].predict(mat_x_cur)
                self.mat_gam_est[mask_cur, s] -= ls_site[s].vec_beta_check[k] * mat_a_cur.mean(axis=1)


class ModelLogisticFDML(torch.nn.Module):
    def __init__(self, site_cen, op_site_cen, score_loc, beta_ini):
        super().__init__()

        self.vec_y = torch.tensor(site_cen.vec_y, dtype=torch.float32) 
        self.vec_d = torch.tensor(site_cen.vec_d, dtype=torch.float32)
        self.mat_x = torch.tensor(site_cen.mat_x, dtype=torch.float32)

        self.mat_m_est = torch.tensor(op_site_cen.mat_m_est, dtype=torch.float32)
        self.mat_gam_est = torch.tensor(op_site_cen.mat_gam_est, dtype=torch.float32)
        self.mat_den_est = torch.tensor(op_site_cen.mat_den_est, dtype=torch.float32)

        self.beta_ini = torch.tensor(beta_ini, dtype=torch.float32)
        self.score_loc = torch.tensor(score_loc, dtype=torch.float32)

        self.beta = torch.nn.Parameter(torch.tensor(beta_ini))
    
    ## surrogate score
    def score(self): 
        mat_score = self.vec_y[:, None] * (
            torch.exp(
                -self.beta * self.vec_d[:, None] - self.mat_gam_est
            ) - torch.exp(
                -self.beta_ini * self.vec_d[:, None] - self.mat_gam_est
            )
        )

        mat_score = self.mat_den_est * mat_score * (self.vec_d[:, None] - self.mat_m_est)
        mat_score = mat_score + self.score_loc

        score = torch.mean(mat_score) 

        return score
    
    def score2(self):
        return self.score() ** 2

def train_logistic_fdml(site_cen, op_site_cen, score_loc, beta_ini, learning_rate):

    model = ModelLogisticFDML(
        site_cen, op_site_cen, score_loc, beta_ini
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    eps = 1e-6
    bl_conv = False

    for epoch in range(2000):  # number of epochs

        optimizer.zero_grad()
        score2 = model.score2()
        score2.backward()
        optimizer.step()

        if score2 < eps:
            bl_conv = True
            # print(">> Converged at epoch {}".format(epoch))
            break

        # if epoch % 100 == 0:
        #     print(
        #         ">> Epoch {} - beta {} - score2 {}".format(
        #             epoch, model.beta.detach().numpy().round(4), score2.detach().numpy().round(4)
        #         )
        #     )

    return model, optimizer, bl_conv


def fig_check(vec1, vec2): 
    plt.scatter(vec1, vec2)
    plt.show()
    exit()

def cal_lm_coef(vec1, vec2): 
    reg = LinearRegression()
    reg.fit(vec1.reshape(-1, 1), vec2)
    return reg.coef_


def fig_lm_coef(vec1, vec2): 
    reg = LinearRegression()
    reg.fit(vec1.reshape(-1, 1), vec2)
    print(">> coef: ", reg.coef_)
    fig_check(vec1, vec2)

def fig_lm_coef_trunc(vec1, vec2, mask): 
    reg = LinearRegression()
    reg.fit(vec1[~mask].reshape(-1, 1), vec2[~mask])
    print(">> coef: ", reg.coef_)
    fig_check(vec1[~mask], vec2[~mask])
    