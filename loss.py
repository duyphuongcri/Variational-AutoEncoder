"""
Author: Duy-Phuong Dao
Email: phuongdd.1997@gmail.com (or duyphuongcri@gmail.com)
"""

import torch
import torch.nn as nn

class KLDivergence(nn.Module):
    "KL divergence between the estimated normal distribution and a prior distribution"
    def __init__(self):
        super(KLDivergence, self).__init__()
        """
        N :  the index N spans all dimensions of input 
        N = H x W x D
        """
        self.N = 80*96*80
    def forward(self, z_mean, z_log_sigma):
        z_log_var = z_log_sigma * 2
        #return (1/self.N) * ( (z_mean**2 + z_var**2 - z_log_var**2 - 1).sum() )
        return 0.5 * ((z_mean**2 + z_log_var.exp() - z_log_var - 1).sum())

class L2Loss(nn.Module): 
    "Measuring the `Euclidian distance` between prediction and ground truh using `L2 Norm`"
    def __init__(self):
        super(L2Loss, self).__init__()
        
    def forward(self, x, y): 
        N = y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3]*y.shape[4]
        return  ( (x - y)**2 ).sum() / N

class L1Loss(nn.Module): 
    "Measuring the `Euclidian distance` between prediction and ground truh using `L1 Norm`"
    def __init__(self):
        super(L1Loss, self).__init__()
        
    def forward(self, x, y): 
        N = y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3]*y.shape[4]
        return  ( (x - y).abs()).sum() / N