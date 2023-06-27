import torch
import torch.nn as nn
import torch.nn.functional as F


class vae_lf(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon_x, x, mean, logvar):
        BCE = F.mse_loss(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        return BCE + KLD
