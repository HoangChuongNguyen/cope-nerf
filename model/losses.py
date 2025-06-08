import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class SmoothnessLoss(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.loss = lambda x: torch.mean(torch.abs(x))
    
    def forward(self, inputs):
        L1 = self.loss(inputs[:,:,:-1] - inputs[:,:,1:])
        L2 = self.loss(inputs[:,:-1,:] - inputs[:,1:,:])
        L3 = self.loss(inputs[:,:-1,:-1] - inputs[:,1:,1:])
        L4 = self.loss(inputs[:,1:,:-1] - inputs[:,:-1,1:])
        return (L1 + L2 + L3 + L4) / 4               

class EdgePreservingSmoothnessLoss(nn.Module):
    def __init__(self, patch_size, bilateral_gamma=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.gamma = bilateral_gamma
        self.loss = lambda x: torch.mean(torch.abs(x))
        self.bilateral_filter = lambda x: torch.exp(-torch.abs(x).sum(-1) / self.gamma).unsqueeze(-1)
    
    def forward(self, inputs, weights):
        w1 = self.bilateral_filter(weights[:,:,:-1] - weights[:,:,1:])
        w2 = self.bilateral_filter(weights[:,:-1,:] - weights[:,1:,:])
        w3 = self.bilateral_filter(weights[:,:-1,:-1] - weights[:,1:,1:])
        w4 = self.bilateral_filter(weights[:,1:,:-1] - weights[:,:-1,1:])

        L1 = self.loss(w1 * (inputs[:,:,:-1] - inputs[:,:,1:]))
        L2 = self.loss(w2 * (inputs[:,:-1,:] - inputs[:,1:,:]))
        L3 = self.loss(w3 * (inputs[:,:-1,:-1] - inputs[:,1:,1:]))
        L4 = self.loss(w4 * (inputs[:,1:,:-1] - inputs[:,:-1,1:]))
        return (L1 + L2 + L3 + L4) / 4

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
compute_ssim_loss = SSIM().to('cuda')