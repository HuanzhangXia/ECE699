import torch.nn as nn
import torch
import  torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Model (deeper residual Conv-VAE)
# ──────────────────────────────────────────────────────────────────────────────
class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.GroupNorm(8, c),
        )
    def forward(self, x):
        return F.silu(self.net(x) + x)

def enc_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, 2, 1, bias=False),
        nn.GroupNorm(8, c_out), nn.SiLU(),
        Residual(c_out)
    )

def dec_block(c_in, c_out):
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False),
        nn.GroupNorm(8, c_out), nn.SiLU(),
        Residual(c_out)
    )

class ConvVAE_L(nn.Module):
    """(1,62,200) ↔ 128-D latent ↔ (1,62,200)"""
    def __init__(self, input_shape=(62, 200), latent_dim=128):
        super().__init__()
        H, W = input_shape
        self.in_shape = input_shape

        # Encoder 4 levels
        self.encoder = nn.Sequential(
            enc_block(1, 32),    # 62×200 → 31×100
            enc_block(32, 64),   # 31×100 → 16×50
            enc_block(64, 128),  # 16×50 →  8×25
            enc_block(128,256),  # 8×25  →  4×13
        )
        with torch.no_grad():
            feat = self.encoder(torch.zeros(1,1,H,W))
        self.feat_shape = feat.shape[1:]      # (256,4,13)
        flat = math.prod(self.feat_shape)

        self.fc_mu, self.fc_logvar = nn.Linear(flat, latent_dim), nn.Linear(flat, latent_dim)
        self.fc_decode             = nn.Linear(latent_dim, flat)

        # Decoder
        self.decoder = nn.Sequential(
            dec_block(256,128), dec_block(128,64), dec_block(64,32),
            nn.ConvTranspose2d(32, 1, kernel_size=(2,4), stride=2, padding=1, bias=False),
            # nn.Sigmoid()
        )

        # Kaiming init
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def encode(self,x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu)*torch.exp(0.5*logvar)
    def decode(self,z):
        h = self.fc_decode(z).view(z.size(0), *self.feat_shape)
        xh = self.decoder(h)
        if xh.shape[-2:] != self.in_shape:           # rare guard
            xh = F.interpolate(xh, size=self.in_shape,
                               mode="bilinear", align_corners=False)
        return xh
    def forward(self,x):
        mu,logvar = self.encode(x)
        z         = self.reparameterize(mu,logvar)
        xh        = self.decode(z)

        return xh, mu, logvar



    
class ConvVAE(nn.Module):
    def __init__(self, input_shape=(62, 200), latent_dim=128):
        super().__init__()
        self.input_shape = input_shape          # (H=62, W=200)

        # ───── Encoder (unchanged) ─────
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)      # [1,1,62,200]
            feat   = self.encoder(dummy)                 # → [1,64,8,25]
        self.feature_shape = feat.shape[1:]              # (64, 8, 25)
        self.flatten_dim   = int(torch.prod(torch.tensor(self.feature_shape)))

        # ───── Latent layers ─────
        self.fc_mu     = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        # ───── Decoder (kernel_size=4 always doubles size cleanly) ─────
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 8×25  → 16×50
            nn.BatchNorm2d(32), nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 16×50 → 32×100
            nn.BatchNorm2d(16), nn.LeakyReLU(),

            nn.ConvTranspose2d(16,  1, 4, 2, 1),  # 32×100 → 64×200
            nn.Tanh()
        )

    # ───── VAE forward paths ─────
    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.fc_decode(z).view(-1, *self.feature_shape)  # [B,64,8,25]
        return self.decoder(h)                               # [B,1,64,200]

    def forward(self, x):
        mu, logvar = self.encode(x)
        x_hat = self.decode(self.reparameterize(mu, logvar))

        # ── Crop height (64→62) if needed so shapes match exactly ──
        H_in, W_in = self.input_shape
        x_hat = x_hat[:, :, :H_in, :W_in]                    # [B,1,62,200]

        return x_hat, mu, logvar



class VAEEncoderClassifier(nn.Module):
    """
    Wraps a pre-trained ConvVAE encoder and adds an MLP classifier.
    Set `freeze_encoder=True` for linear-probe; False for finetuning.
    """
    def __init__(self, vae: "ConvVAE", num_classes: int = 4, freeze_encoder: bool = True):
        super().__init__()

        # copy encoder from the trained VAE
        self.encoder = vae.encoder
        self.flatten_dim = vae.fc_mu.in_features
        self.mu_layer    = vae.fc_mu          # use μ as representation

        # optionally freeze
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.mu_layer.parameters():
                p.requires_grad = False

        # classification head (simple MLP)
        self.classifier = nn.Sequential(
            nn.Linear(self.mu_layer.out_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """x: [B, 1, 62, 200]  → logits: [B, num_classes]"""
        h = self.encoder(x)                 # [B, 64, 8, 25]
        mu = self.mu_layer(h.view(x.size(0), -1))
        logits = self.classifier(mu)
        return logits

