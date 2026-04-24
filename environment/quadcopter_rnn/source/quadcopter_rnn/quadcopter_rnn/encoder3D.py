"""
uses Conv3d / ConvTranspose3d
avoids fragile hardcoded flatten sizes (uses adaptive pooling)
replaces BatchNorm with InstanceNorm3d
reduces channels to be memory-safe
keeps your residual design
keeps VAE logic unchanged
"""

import torch
import torch.nn as nn


# -------------------------
# Residual Block 3D
# -------------------------
class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


# -------------------------
# Encoder 3D
# -------------------------
class ImgEncoder3D(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.relu = nn.ReLU()

        # Safer channel sizes for 3D
        self.conv0 = nn.Sequential(
            nn.Conv3d(input_dim, 16, kernel_size=5, stride=(1,2,2), padding=2, bias=False),
            nn.InstanceNorm3d(16),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
        )

        # Residual at deeper level
        self.res = ResidualBlock3D(64)

        # Force fixed shape (prevents shape headaches)
        self.pool = nn.AdaptiveAvgPool3d((4, 4, 4))

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim),
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


# -------------------------
# Decoder 3D
# -------------------------
class ImgDecoder3D(nn.Module):
    def __init__(self, output_dim=1, latent_dim=64, with_logits=False):
        super().__init__()
        self.with_logits = with_logits

        self.elu = nn.ELU()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 64 * 4 * 4 * 4),
            nn.ELU(),
        )

        # Bottleneck reshape: (B, 64, 4, 4, 4)
        self.res = ResidualBlock3D(64)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(64),
            nn.ELU(),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(32),
            nn.ELU(),
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=(1,2,2), padding=1, bias=False),
            nn.InstanceNorm3d(16),
            nn.ELU(),
        )

        self.final = nn.ConvTranspose3d(
            16, output_dim, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 64, 4, 4, 4)

        x = self.res(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        x = self.final(x)

        if self.with_logits:
            return x
        return torch.sigmoid(x)


# -------------------------
# Lambda helper
# -------------------------
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


# -------------------------
# VAE 3D
# -------------------------
class VAE3D(nn.Module):
    def __init__(self, input_dim=1, latent_dim=64, with_logits=False, inference_mode=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.inference_mode = inference_mode
        self.with_logits = with_logits

        self.encoder = ImgEncoder3D(input_dim, latent_dim)
        self.decoder = ImgDecoder3D(input_dim, latent_dim, with_logits)

        self.mean = Lambda(lambda x: x[:, :latent_dim])
        self.logvar = Lambda(lambda x: x[:, latent_dim:])

    def reparameterize(self, mean, logvar):
        logvar = torch.clamp(logvar, -4, 4)
        std = torch.exp(0.5 * logvar)

        if self.inference_mode:
            eps = torch.zeros_like(std)
        else:
            eps = torch.randn_like(std)

        return mean + eps * std, std, logvar

    def forward(self, x):
        z = self.encoder(x)

        mean = self.mean(z)
        logvar = self.logvar(z)

        z_sampled, std, logvar = self.reparameterize(mean, logvar)

        recon = self.decoder(z_sampled)

        return recon, mean, logvar, z_sampled

    def encode(self, x):
        z = self.encoder(x)
        mean = self.mean(z)
        logvar = self.logvar(z)
        z_sampled, std, _ = self.reparameterize(mean, logvar)
        return z_sampled, mean, std

    def decode(self, z):
        x = self.decoder(z)
        if self.with_logits:
            return torch.sigmoid(x)
        return x

    def set_inference_mode(self, mode=True):
        self.inference_mode = mode