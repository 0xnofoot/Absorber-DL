import torch
import torch.nn as nn
from . import block


class Encoder_0(nn.Module):
    def __init__(self, img_channels, img_size, hidden_dim, img_latent_dim, sp_latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            block.BN_Conv2d(1, 32, 4, 2, 1),  # 32
            block.BN_Conv2d(32, 64, 4, 2, 1),  # 16
            block.BN_Conv2d(64, 128, 4, 2, 1),  # 8
            block.BN_Conv2d(128, hidden_dim, 4, 2, 1),  # 4
            nn.AvgPool2d(4),
            nn.Flatten()
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(hidden_dim, img_latent_dim),
            nn.ReLU(),
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(img_latent_dim + sp_latent_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mean = nn.Linear(hidden_dim, img_latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, img_latent_dim)

    def forward(self, img, sp_z):
        img_out = self.conv(img)
        img_out = self.fc_1(img_out)
        out = torch.cat([img_out, sp_z], dim=1)
        out = self.fc_2(out)
        img_mu = self.fc_mean(out)
        img_logvar = self.fc_logvar(out)
        return img_mu, img_logvar


class Encoder_1(nn.Module):
    def __init__(self, img_channels, img_size, hidden_dim, img_latent_dim, sp_latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            block.BN_Conv2d(3, 64, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1),
            block.BN_Conv2d(64, 192, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),  # 28
            block.BN_Conv2d(192, hidden_dim, 4, 2, 1),  # 14
            block.BN_Conv2d(hidden_dim, hidden_dim, 4, 2, 1),  # 7
            nn.AvgPool2d(7),
            nn.Flatten()
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(hidden_dim, img_latent_dim),
            nn.ReLU(),
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(img_latent_dim + sp_latent_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mean = nn.Linear(hidden_dim, img_latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, img_latent_dim)

    def forward(self, img, sp_z):
        img = nn.functional.interpolate(img, size=(256, 256))
        img = torch.cat([img, img, img], dim=1)
        img_out = self.conv(img)
        img_out = self.fc_1(img_out)
        out = torch.cat([img_out, sp_z], dim=1)
        out = self.fc_2(out)
        img_mu = self.fc_mean(out)
        img_logvar = self.fc_logvar(out)
        return img_mu, img_logvar
