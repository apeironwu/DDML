# -*- coding: utf-8 -*-
import torch
import numpy as np

class Den_est_rkhs_rff(torch.nn.Module):

    def __init__(self, x, omega, D, reg = 1e-3):
        super().__init__()
        
        self.x = x  ## input data (n, p_x)
        self.omega = omega  ## RFF frequencies (p_x, D)
        self.D = D  ## dim of RFF

        self.reg = reg

        # Parameters
        self.beta = torch.nn.Parameter(torch.rand(2 * self.D))

        # Parameter for SE kernel
        # self.ell_log = torch.nn.Parameter(torch.tensor(0.0))
        self.ell_log = torch.tensor(0.0)

    ## return phi(x)
    def phi_x(self, x): 
        return torch.cat(
            [torch.cos(x.mm(torch.exp(self.ell_log) * self.omega)), 
             torch.sin(x.mm(torch.exp(self.ell_log) * self.omega))], 
            axis=1
        ) / np.sqrt(self.D)
    
    ## return density estimation
    def forward(self, x_Fourier):
        return (x_Fourier.matmul(self.beta)).reshape(-1) 

    ## return shift parameter (scale / loc / max)
    def shift_par3(self): 
        shift_scale = self.beta.mean() / self.phi_x(self.x).mean()

        vec_self_pred = self.pred(self.x, bl_shift = False)
        shift_loc = vec_self_pred.quantile(0.05)

        shift_max = (vec_self_pred.max() - shift_loc) / shift_scale * 0.5

        return shift_scale, shift_loc, shift_max

    ## return density estimation for prediction
    def pred(self, x, bl_shift = True): 

        f_preds = self.forward(self.phi_x(x)) 

        if bl_shift: 
            shift_scale, shift_loc, shift_max = self.shift_par3()

            f_preds = f_preds - shift_loc
            f_preds[f_preds < 0.0] = 0.0

            f_preds = f_preds / shift_scale * 0.5

            f_preds = 0.95 * f_preds + 0.05 * shift_max

        return f_preds

    def loss(self):
        loss = - 2.0 * torch.sum(
            torch.log(
                torch.maximum(
                    torch.tensor(1e-6),
                    self.forward(self.phi_x(self.x))
                )
            )
        ) 
        regularizer = self.reg * torch.sum(self.beta ** 2)

        return loss + regularizer

