import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    """
    Residual block that refines features WITHOUT changing shape.
    Input and output are identical in (channels, D, H, W).
    Uses: Activation → Conv → IN → Activation → Conv → IN → add input
    """
    def __init__(self, channels, activation=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            activation(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            activation(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ImgEncoder3D(nn.Module):
    def __init__(self, input_dim=2,
                 latent_dim=64,
                 num_conv_layers=4,
                 use_residual=True,
                 residual_every=2,
                 use_skip=True):
        super().__init__()
        self.input_dim       = input_dim
        self.latent_dim      = latent_dim
        self.num_conv_layers = num_conv_layers
        self.use_residual    = use_residual
        self.residual_every  = residual_every
        self.use_skip        = use_skip
        self.relu            = nn.ReLU()

        self.channel_plan = [
            (input_dim, 32),
            (32,        32),
            (32,        64),
            (64,       128),
            (128,      128),
            (128,      128),
        ]

        # Exactly 2 downsample steps: layer 0 and layer num_conv_layers // 2
        # Input (2, 8, 16, 16) → stride 2 → (ch, 4, 8, 8) → stride 2 → (ch, 2, 4, 4)
        self.downsample_at = {0, num_conv_layers // 2}

        # ── Residual block placement ──────────────────────────────────────
        # Same logic as 2D: only in the deep half, every N layers
        #
        # Example with num_conv_layers=4, residual_every=2:
        #   deep half starts at layer 2
        #   candidates: 2, 3
        #   every 2: place at 2 only
        #
        # Example with num_conv_layers=6, residual_every=2:
        #   deep half starts at layer 3
        #   candidates: 3, 4, 5
        #   every 2: place at 3, 5

        self.residual_at = set()
        if use_residual and num_conv_layers >= 3:
            deep_start = num_conv_layers // 2
            deep_layers = list(range(deep_start, num_conv_layers))
            self.residual_at = set(
                deep_layers[i]
                for i in range(0, len(deep_layers), residual_every)
            )

        print(f"  Encoder3D: {num_conv_layers} conv layers | "
              f"residual every {residual_every} deep layers | "
              f"residual at: {sorted(self.residual_at) or 'none'}")

        # ── Build layers ──────────────────────────────────────────────────
        self.conv_layers = nn.ModuleList()
        self.bn_layers   = nn.ModuleList()
        self.res_blocks  = nn.ModuleDict()
        self.skip_layers = nn.ModuleDict()
        self.skip_bn     = nn.ModuleDict()

        for i in range(num_conv_layers):
            in_ch, out_ch = self.channel_plan[i]
            stride = 2 if i in self.downsample_at else 1

            self.conv_layers.append(
                nn.Conv3d(in_ch, out_ch, kernel_size=3,
                          stride=stride, padding=1, bias=False)
            )
            self.bn_layers.append(nn.InstanceNorm3d(out_ch))

            if i in self.residual_at:
                self.res_blocks[str(i)] = ResidualBlock3D(out_ch)

            if use_skip and i > 0 and stride == 2 and in_ch != out_ch:
                self.skip_layers[str(i)] = nn.Conv3d(
                    in_ch, out_ch, kernel_size=1, stride=2, bias=False
                )
                self.skip_bn[str(i)] = nn.InstanceNorm3d(out_ch)

        self.flat_size = self._compute_flat_size()
        print(f"  Encoder3D flat size: {self.flat_size}")

        self.dense0 = nn.Linear(self.flat_size, 4 * self.latent_dim)
        self.dense1 = nn.Linear(4 * self.latent_dim, 2 * self.latent_dim)

    def _compute_flat_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_dim, 8, 16, 16)
            x = dummy
            for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
                identity = x
                x = self.relu(bn(conv(x)))
                if str(i) in self.skip_layers:
                    skip = self.skip_bn[str(i)](self.skip_layers[str(i)](identity))
                    x = self.relu(x + skip)
                if str(i) in self.res_blocks:
                    x = self.res_blocks[str(i)](x)
            return x.view(1, -1).shape[1]

    def forward(self, img):
        x = img
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            identity = x
            out = bn(conv(x))
            if str(i) in self.skip_layers:
                skip = self.skip_bn[str(i)](self.skip_layers[str(i)](identity))
                x = self.relu(out + skip)
            else:
                x = self.relu(out)

            if str(i) in self.res_blocks:
                x = self.res_blocks[str(i)](x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.dense0(x))
        return self.dense1(x)


class ImgDecoder3D(nn.Module):
    def __init__(self, output_dim=2, latent_dim=64, with_logits=False,
                 num_deconv_layers=4,
                 use_residual=True,
                 residual_every=2,
                 use_skip=True):
        super().__init__()
        self.with_logits       = with_logits
        self.output_dim        = output_dim
        self.num_deconv_layers = num_deconv_layers
        self.use_residual      = use_residual
        self.residual_every    = residual_every
        self.elu = nn.ELU()

        # ── Dense layers ──────────────────────────────────────────────────
        self.dense0 = nn.Linear(latent_dim, latent_dim * 2)
        self.dense1 = nn.Linear(latent_dim * 2, latent_dim * 4)
        self.dense2 = nn.Linear(latent_dim * 4, 128 * 2 * 4 * 4)

        # ── Deconv layer pool ─────────────────────────────────────────────
        # Seed shape: (128, 2, 4, 4)
        # Target:     (output_dim, 8, 16, 16)
        # Need: 2→8 (×4), 4→16 (×4) → exactly 2 stride-2 deconv layers
        #
        # ConvTranspose3d k=4, s=2, p=1, op=0 → doubles all spatial dims
        # ConvTranspose3d k=3, s=1, p=1, op=0 → same dims (refinement)
        #
        # Spatial progression (4-layer version):
        # seed:     (128, 2,  4,  4)
        # deconv0:  (128, 2,  4,  4)  stride=1  refinement
        # deconv1:  ( 64, 4,  8,  8)  stride=2  ×2
        # deconv2:  ( 16, 8, 16, 16)  stride=2  ×2
        # deconv3:  ( od, 8, 16, 16)  stride=1  final output

        od = output_dim
        self.ALL_DECONV_CFGS = {
            # (in_ch, out_ch, kernel, stride, padding, output_padding)
            3: [
                (128,  64, 4, 2, 1, 0),   # (64,  4, 8, 8)     ×2
                ( 64,  16, 4, 2, 1, 0),   # (16,  8, 16, 16)   ×2
                ( 16,  od, 3, 1, 1, 0),   # (od,  8, 16, 16)   final
            ],
            4: [
                (128, 128, 3, 1, 1, 0),   # (128, 2, 4, 4)     refinement
                (128,  64, 4, 2, 1, 0),   # (64,  4, 8, 8)     ×2
                ( 64,  16, 4, 2, 1, 0),   # (16,  8, 16, 16)   ×2
                ( 16,  od, 3, 1, 1, 0),   # (od,  8, 16, 16)   final
            ],
            5: [
                (128, 128, 3, 1, 1, 0),   # (128, 2, 4, 4)     refinement
                (128,  64, 4, 2, 1, 0),   # (64,  4, 8, 8)     ×2
                ( 64,  64, 3, 1, 1, 0),   # (64,  4, 8, 8)     refinement
                ( 64,  16, 4, 2, 1, 0),   # (16,  8, 16, 16)   ×2
                ( 16,  od, 3, 1, 1, 0),   # (od,  8, 16, 16)   final
            ],
            6: [
                (128, 128, 3, 1, 1, 0),   # (128, 2, 4, 4)     refinement
                (128,  64, 4, 2, 1, 0),   # (64,  4, 8, 8)     ×2
                ( 64,  64, 3, 1, 1, 0),   # (64,  4, 8, 8)     refinement
                ( 64,  32, 4, 2, 1, 0),   # (32,  8, 16, 16)   ×2
                ( 32,  16, 3, 1, 1, 0),   # (16,  8, 16, 16)   refinement
                ( 16,  od, 3, 1, 1, 0),   # (od,  8, 16, 16)   final
            ],
        }

        if num_deconv_layers not in self.ALL_DECONV_CFGS:
            raise ValueError(
                f"num_deconv_layers={num_deconv_layers} not supported. "
                f"Choose from {list(self.ALL_DECONV_CFGS.keys())}"
            )

        deconv_cfgs = self.ALL_DECONV_CFGS[num_deconv_layers]

        # ── Residual placement (same logic as 2D) ────────────────────────
        self.residual_at = set()
        if use_residual and num_deconv_layers >= 3:
            deep_start = num_deconv_layers // 2
            deep_layers = list(range(deep_start, num_deconv_layers))
            self.residual_at = set(
                deep_layers[i]
                for i in range(0, len(deep_layers), residual_every)
            )

        print(f"  Decoder3D: {num_deconv_layers} deconv layers | "
              f"residual at: {sorted(self.residual_at) or 'none'}")

        # ── Build layers ──────────────────────────────────────────────────
        self.deconv_layers = nn.ModuleList()
        self.deconv_bn     = nn.ModuleList()
        self.res_blocks    = nn.ModuleDict()
        self.skip_layers   = nn.ModuleDict()
        self.skip_bn       = nn.ModuleDict()

        for i, (in_ch, out_ch, k, s, p, op) in enumerate(deconv_cfgs):
            is_final = (i == len(deconv_cfgs) - 1)

            self.deconv_layers.append(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=k,
                                   stride=s, padding=p,
                                   output_padding=op, bias=False)
            )
            # no norm on final layer — sigmoid handles normalization
            self.deconv_bn.append(
                nn.InstanceNorm3d(out_ch) if not is_final else nn.Identity()
            )

            if i in self.residual_at:
                self.res_blocks[str(i)] = ResidualBlock3D(out_ch, activation=nn.ELU)

            # skip connection when channels change, stride > 1, and not final
            if use_skip and in_ch != out_ch and s > 1 and not is_final:
                skip_op = op + (k - 1) - 2 * p
                if 0 <= skip_op < s:
                    skip_k, skip_p, skip_op_final = 1, 0, skip_op
                else:
                    skip_k, skip_p, skip_op_final = k, p, op
                self.skip_layers[str(i)] = nn.ConvTranspose3d(
                    in_ch, out_ch, kernel_size=skip_k,
                    stride=s, padding=skip_p,
                    output_padding=skip_op_final, bias=False
                )
                self.skip_bn[str(i)] = nn.InstanceNorm3d(out_ch)

        # ── Verify output size ────────────────────────────────────────────
        self._verify_output_size()

    def _verify_output_size(self):
        """Sanity check — confirm we reach exactly (output_dim, 8, 16, 16)."""
        with torch.no_grad():
            dummy = torch.zeros(1, 128, 2, 4, 4)
            x = dummy
            for i, (deconv, bn) in enumerate(
                zip(self.deconv_layers, self.deconv_bn)
            ):
                identity = x
                out = deconv(x)
                out = bn(out)

                if str(i) in self.skip_layers:
                    skip = self.skip_bn[str(i)](
                        self.skip_layers[str(i)](identity)
                    )
                    x = out + skip
                else:
                    x = out

                if i < len(self.deconv_layers) - 1:
                    x = torch.nn.functional.elu(x)

                if str(i) in self.res_blocks:
                    x = self.res_blocks[str(i)](x)

        expected = (1, self.output_dim, 8, 16, 16)
        if tuple(x.shape) != expected:
            raise RuntimeError(
                f"Decoder3D output shape {tuple(x.shape)} != {expected}. "
                f"Fix the deconv configs for "
                f"num_deconv_layers={self.num_deconv_layers}."
            )
        print(f"  Decoder3D output verified: {tuple(x.shape)} ✓")

    def forward(self, z):
        # fixed dense expansion
        x = self.elu(self.dense0(z))
        x = self.elu(self.dense1(x))
        x = self.elu(self.dense2(x))
        x = x.view(x.size(0), 128, 2, 4, 4)

        # variable deconv
        for i, (deconv, bn) in enumerate(
            zip(self.deconv_layers, self.deconv_bn)
        ):
            is_final = (i == len(self.deconv_layers) - 1)
            identity = x
            out = deconv(x)
            out = bn(out)

            if str(i) in self.skip_layers:
                skip = self.skip_bn[str(i)](
                    self.skip_layers[str(i)](identity)
                )
                x = out + skip
            else:
                x = out

            if not is_final:
                x = self.elu(x)

            if str(i) in self.res_blocks:
                x = self.res_blocks[str(i)](x)

        if self.with_logits:
            return x
        return torch.sigmoid(x)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class VAE3D(nn.Module):
    def __init__(self, input_dim=2, latent_dim=64, with_logits=False,
                 inference_mode=False,
                 num_conv_layers=4,
                 use_residual=True,
                 residual_every=2,
                 num_deconv_layers=4,
                 use_skip=True):
        super().__init__()
        self.with_logits    = with_logits
        self.input_dim      = input_dim
        self.latent_dim     = latent_dim
        self.inference_mode = inference_mode

        self.encoder = ImgEncoder3D(
            input_dim       = input_dim,
            latent_dim      = latent_dim,
            num_conv_layers = num_conv_layers,
            use_residual    = use_residual,
            residual_every  = residual_every,
            use_skip        = use_skip,
        )
        self.decoder = ImgDecoder3D(
            output_dim        = input_dim,
            latent_dim        = latent_dim,
            with_logits       = with_logits,
            num_deconv_layers = num_deconv_layers,
            use_residual      = use_residual,
            residual_every    = residual_every,
            use_skip          = use_skip,
        )

        self.mean_params   = Lambda(lambda x: x[:, :latent_dim])
        self.logvar_params = Lambda(lambda x: x[:, latent_dim:])

    def forward(self, x):
        z = self.encoder(x)

        mean   = self.mean_params(z)
        logvar = torch.clamp(self.logvar_params(z), min=-4.0, max=4.0)

        std = torch.exp(0.5 * logvar)
        eps = torch.zeros_like(std) if self.inference_mode else torch.randn_like(std)
        z_sampled = mean + eps * std

        recon = self.decoder(z_sampled)
        return recon, mean, logvar, z_sampled

    def encode(self, x):
        z      = self.encoder(x)
        mean   = self.mean_params(z)
        logvar = torch.clamp(self.logvar_params(z), min=-4.0, max=4.0)
        std    = torch.exp(0.5 * logvar)
        eps    = torch.zeros_like(std) if self.inference_mode else torch.randn_like(std)
        return mean + eps * std, mean, std

    def decode(self, z):
        # Same convention as forward()/the decoder:
        #   with_logits=True  → raw logits (caller applies sigmoid)
        #   with_logits=False → probabilities (sigmoid already applied)
        return self.decoder(z)


    def set_inference_mode(self, mode):
        self.inference_mode = mode
