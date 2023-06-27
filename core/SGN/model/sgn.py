import os

import torch
import torch.nn as nn
from . import encoder as ed, decoder as dd
from core.SFEN.model import sfen


def get_sgn(img_channels=1, img_size=64, hidden_dim=256, img_latent_dim=64):
    return SGN_0(img_channels, img_size, hidden_dim, img_latent_dim), img_latent_dim
    # return VAE_1(img_channels, img_size, hidden_dim, img_latent_dim), img_latent_dim


def SGN_0(img_channels, img_size, hidden_dim, img_latent_dim):
    return SGN_builder(ed.Encoder_0, dd.Decoder_0, img_channels, img_size, hidden_dim, img_latent_dim)


def SGN_1(img_channels, img_size, hidden_dim, img_latent_dim):
    return SGN_builder(ed.Encoder_1, dd.Decoder_0, img_channels, img_size, hidden_dim, img_latent_dim)


class SGN_builder(nn.Module):
    def __init__(self, encoder, decoder, img_channels, img_size, hidden_dim, img_latent_dim):
        super().__init__()

        self.img_latent_dim = img_latent_dim

        SFEN, sp_latent_dim = sfen.get_sfen()
        SFEN.load_state_dict(torch.load(os.path.join("ext", "SFEN.pth")))
        for param in SFEN.parameters():
            param.requires_grad = False
        self.sp_vae = SFEN

        self.encode = encoder(img_channels, img_size, hidden_dim, img_latent_dim, sp_latent_dim)
        self.decode = decoder(img_channels, img_size, hidden_dim, img_latent_dim, sp_latent_dim)

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def fetch_spz(self, sp):
        sp_mu, sp_logvar = self.sp_vae.encode(sp)
        sp_z = sp_mu + torch.exp(0.5 * sp_logvar)
        return sp_z

    def only_decode(self, sp, noise=None):
        sp_z = self.fetch_spz(sp)
        if noise is None:
            noise = torch.randn(sp.size(0), self.img_latent_dim).to(next(self.parameters()).device)
        g_img = self.decode(noise, sp_z)
        return g_img

    def forward(self, img, sp):
        sp_z = self.fetch_spz(sp)
        img_mu, img_logvar = self.encode(img, sp_z)
        img_z = self.reparameterize(img_mu, img_logvar)

        g_img = self.decode(img_z, sp_z)
        return g_img, img_mu, img_logvar
