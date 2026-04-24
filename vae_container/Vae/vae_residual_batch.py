import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block that refines features WITHOUT changing shape.
    Input and output are identical in (channels, H, W).
    Uses: Conv → BN → ELU → Conv → BN → add input → ELU
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            # NO activation here — applied AFTER addition
        )

    def forward(self, x):
        return x + self.block(x)


class ImgEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.relu = nn.ReLU()
        self._build()

    def _build(self):
        # Block 0 — no residual, features are simple at this scale
        self.conv0   = nn.Conv2d(self.input_dim, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn0     = nn.BatchNorm2d(32)
        self.conv0_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0_1   = nn.BatchNorm2d(32)

        # Block 1 — no residual
        self.conv1_0      = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1_0        = nn.BatchNorm2d(32)
        self.conv1_1      = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1        = nn.BatchNorm2d(64)
        self.conv0_jump_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn0_jump_2   = nn.BatchNorm2d(64)

        # Block 2 — first residual block added here (128ch, abstract features)
        self.conv2_0      = nn.Conv2d(64,  64,  kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2_0        = nn.BatchNorm2d(64)
        self.conv2_1      = nn.Conv2d(64,  128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2_1        = nn.BatchNorm2d(128)
        self.conv1_jump_3 = nn.Conv2d(64,  128, kernel_size=5, stride=4, padding=(2,1), bias=False)
        self.bn1_jump_3   = nn.BatchNorm2d(128)
        self.res2         = ResidualBlock(128)  # ← first residual block

        # Block 3 — second residual block (deepest, most abstract)
        self.conv3_0 = nn.Conv2d(128, 128, kernel_size=5, stride=2, bias=False)
        self.bn3_0   = nn.BatchNorm2d(128)
        self.res3    = ResidualBlock(128)       # ← second residual block

        # Dense — no BN, approaching latent space
        self.dense0 = nn.Linear(3 * 6 * 128, 512)
        self.dense1 = nn.Linear(512, 2 * self.latent_dim)

    def forward(self, img):
        # Block 0
        x0_0 = self.relu(self.bn0(self.conv0(img)))
        x0_1 = self.relu(self.bn0_1(self.conv0_1(x0_0)))

        # Block 1
        x1_0 = self.relu(self.bn1_0(self.conv1_0(x0_1)))
        x1_1 = self.bn1_1(self.conv1_1(x1_0))
        x0_j = self.bn0_jump_2(self.conv0_jump_2(x0_1))
        x1_1 = self.relu(x1_1 + x0_j)

        # Block 2 + first residual
        x2_0 = self.relu(self.bn2_0(self.conv2_0(x1_1)))
        x2_1 = self.bn2_1(self.conv2_1(x2_0))
        x1_j = self.bn1_jump_3(self.conv1_jump_3(x1_1))
        x2_1 = self.relu(x2_1 + x1_j)
        x2_1 = self.res2(x2_1)         # refine deep 128ch features

        # Block 3 + second residual
        x3_0 = self.relu(self.bn3_0(self.conv3_0(x2_1)))
        x3_0 = self.res3(x3_0)         # refine deepest features

        # Flatten + dense
        x = x3_0.view(x3_0.size(0), -1)
        x = self.relu(self.dense0(x))
        x = self.dense1(x)              # output: [mu | logvar], no BN
        return x


class ImgDecoder(nn.Module):
    def __init__(self, input_dim=1, latent_dim=64, with_logits=False):
        super().__init__()
        self.with_logits = with_logits
        self.n_channels  = input_dim
        self.elu = nn.ELU()

        # Dense — no BN, input is sampled latent z
        self.dense0 = nn.Linear(latent_dim, 512)
        self.dense1 = nn.Linear(512, 1024)
        self.dense2 = nn.Linear(1024, 9 * 15 * 128)

        # One residual block right at the bottleneck
        # This is where spatial structure first appears
        # and needs the most refinement
        self.res_bottleneck = ResidualBlock(128)  # ← only one

        # Upsampling layers with BN
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2,
                               padding=(2,2), output_padding=(0,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=4,
                               padding=(2,2), output_padding=(0,0), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2,
                               padding=(0,0), output_padding=(0,1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )
        # Final output — no BN, sigmoid handles range
        self.deconv7 = nn.ConvTranspose2d(
            16, self.n_channels, kernel_size=4, stride=2, padding=2
        )

    def forward(self, z):
        # Dense expansion — no BN
        x = self.elu(self.dense0(z))
        x = self.elu(self.dense1(x))
        x = self.elu(self.dense2(x))
        x = x.view(x.size(0), 128, 9, 15)

        # Residual refinement at bottleneck — hardest transition
        x = self.res_bottleneck(x)

        # Upsample to full resolution
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv4(x)
        x = self.deconv6(x)

        # Final output — no BN, no activation before sigmoid
        x = self.deconv7(x)
        if self.with_logits:
            return x
        return torch.sigmoid(x)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class VAE(nn.Module):
    def __init__(self, input_dim=1, latent_dim=64, with_logits=False, inference_mode=False):
        super().__init__()
        self.with_logits     = with_logits
        self.input_dim       = input_dim
        self.latent_dim      = latent_dim
        self.inference_mode  = inference_mode

        self.encoder     = ImgEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.img_decoder = ImgDecoder(input_dim=1, latent_dim=latent_dim, with_logits=with_logits)

        self.mean_params   = Lambda(lambda x: x[:, :latent_dim])
        self.logvar_params = Lambda(lambda x: x[:, latent_dim:])

    def forward(self, img):
        z = self.encoder(img)

        mean   = self.mean_params(z)
        # clamp logvar — prevents std from exploding or vanishing
        logvar = torch.clamp(self.logvar_params(z), min=-4.0, max=4.0)

        std = torch.exp(0.5 * logvar)
        eps = torch.zeros_like(std) if self.inference_mode else torch.randn_like(std)
        z_sampled = mean + eps * std

        img_recon = self.img_decoder(z_sampled)
        return img_recon, mean, logvar, z_sampled

    def encode(self, img):
        z      = self.encoder(img)
        mean   = self.mean_params(z)
        logvar = torch.clamp(self.logvar_params(z), min=-4.0, max=4.0)
        std    = torch.exp(0.5 * logvar)
        eps    = torch.zeros_like(std) if self.inference_mode else torch.randn_like(std)
        return mean + eps * std, mean, std

    def decode(self, z):
        img_recon = self.img_decoder(z)
        if self.with_logits:
            return torch.sigmoid(img_recon)
        return img_recon

    def set_inference_mode(self, mode):
        self.inference_mode = mode


# # LIKE THE FIGURE

# class ImgEncoder(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super().__init__()
#         self.latent_dim = latent_dim

#         self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.pool  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.res1  = ResidualBlock(64)
#         self.res2  = ResidualBlock(64)

#         self.flatten = nn.Flatten()
#         # Adjust input size to match your spatial dims after pooling
#         self.dense0 = nn.Linear(64 * H * W, 512)  # replace H, W accordingly
#         self.dense1 = nn.Linear(512, 2 * latent_dim)
#         self.relu   = nn.ReLU()

#     def forward(self, img):
#         x = self.relu(self.conv1(img))
#         x = self.pool(x)
#         x = self.res1(x)
#         x = self.res2(x)
#         x = self.flatten(x)
#         x = self.relu(self.dense0(x))
#         return self.dense1(x)