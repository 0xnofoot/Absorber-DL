import torch
import torch.nn as nn
from . import block


class Decoder_0(nn.Module):
    def __init__(self, img_channels, img_size, hidden_dim, img_latent_dim, sp_latent_dim):
        super().__init__()

        self.img_channels = img_channels
        self.img_size = img_size
        self.hidden_dim = hidden_dim

        self.fc_decode = nn.Sequential(
            nn.Linear(img_latent_dim + sp_latent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.deconv_decode = nn.Sequential(
            block.BN_DeConv2d(hidden_dim, 512, 4, 1, 0),
            block.BN_DeConv2d(512, 480, 4, 2, 1),
            block.BN_DeConv2d(480, 256, 4, 2, 1),
            block.BN_DeConv2d(256, 64, 4, 2, 1),
            block.BN_DeConv2d(64, 32, 1, 1, 0),
            nn.ConvTranspose2d(32, img_channels, 1, 1, 0),  # img_channels
            nn.Sigmoid(),
        )

    @staticmethod
    def pic_montage(image):
        image = torch.cat([image, image.flip(image.dim() - 1)], dim=3)
        image = torch.cat([image, image.flip(image.dim() - 2)], dim=2)
        return image

    def forward(self, img_z, sp_z):
        z = torch.cat([img_z, sp_z], dim=1)
        hidden_out = self.fc_decode(z).unsqueeze(2).unsqueeze(3)
        out = self.deconv_decode(hidden_out)
        out = self.pic_montage(out)
        return out


class Decoder_1(nn.Module):
    def __init__(self, img_channels, img_size, hidden_dim, img_latent_dim, sp_latent_dim):
        super().__init__()

        self.img_channels = img_channels
        self.img_size = img_size
        self.hidden_dim = hidden_dim

        self.fc_decode = nn.Sequential(
            nn.Linear(img_latent_dim + sp_latent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.deconv_decode = nn.Sequential(
            block.BN_DeConv2d(hidden_dim, 480, 4, 1, 0),  # 4
            block.BN_DeConv2d(480, 512, 4, 2, 1),  # 8
            block.BN_DeConv2d(512, 480, 4, 2, 1),  # 16
            block.BN_DeConv2d(480, 256, 4, 2, 1),  # 32
            block.BN_DeConv2d(256, 192, 4, 2, 1),  # 64
            block.BN_DeConv2d(192, 128, 4, 2, 1),  # 128
            block.BN_Conv2d(128, 64, 4, 2, 1),  # 64
            block.BN_Conv2d(64, 32, 4, 2, 1),  # 32
            block.BN_Conv2d(32, 16, 1, 1, 0),  # 16
            nn.Conv2d(16, img_channels, 1, 1, 0),  # img_channels
            nn.Sigmoid(),
        )

    @staticmethod
    def pic_montage(image):
        image = torch.cat([image, image.flip(image.dim() - 1)], dim=3)
        image = torch.cat([image, image.flip(image.dim() - 2)], dim=2)
        return image

    def forward(self, img_z, sp_z):
        z = torch.cat([img_z, sp_z], dim=1)
        hidden_out = self.fc_decode(z).unsqueeze(2).unsqueeze(3)
        out = self.deconv_decode(hidden_out)
        out = self.pic_montage(out)
        return out
