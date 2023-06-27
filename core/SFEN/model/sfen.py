import torch
import torch.nn as nn

from . import encoder as ed
from . import decoder as dd


def get_sfen(input_dim=1000, hidden_dim=256, latent_dim=64):
    return SFEN_0(input_dim, hidden_dim, latent_dim), latent_dim


def SFEN_0(input_dim, hidden_dim, latent_dim):
    return SFEN_builder(ed.Encoder_0, dd.Decoder_0, input_dim, hidden_dim, latent_dim)


class SFEN_builder(nn.Module):
    def __init__(self, encoder, decoder, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encode = encoder(input_dim, hidden_dim, latent_dim)
        self.decode = decoder(latent_dim, hidden_dim, input_dim)

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decode(z)
        return recon_x, mean, logvar
