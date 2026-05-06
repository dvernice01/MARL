import torch
import torch.nn as nn


"""
Summary of changes from the 2D version:
                                                                                          
  Layer-level swaps                                                                                                                                                    
  - Conv2d → Conv3d, ConvTranspose2d → ConvTranspose3d, BatchNorm2d → BatchNorm3d                                                                                      
  - output_padding is now a 3-tuple (d, h, w) instead of (h, w)                                                                                                        
  - The skip-connection output_padding fix applies per-axis (D, H, W) instead of (H, W)                                                                                
                                                                                                                                                                       
  Sizing changes (defaults targeted at local-map input (2, 8, 16, 16))                                                                                                 
  - input_shape is a parameter (D, H, W) — the 2D version hard-coded (270, 480)                                                                                        
  - channel_plan uses much smaller channels (16/32/64) — 3D activations cost ~D× more memory                                                                           
  - Encoder dense path: flat → 256 → 2*latent_dim (vs 2D's flat → 512 → 2*latent_dim)                                                                                  
  - Decoder dense path: latent → 256 → flat (vs 2D's latent → 512 → 1024 → flat)                                                                                       
  - Bottleneck shape is configurable (default (2, 4, 4) with 64 channels) and decoupled from the encoder,
    just like in the 2D version                                  
                                                                                                                                                                       
  Deconv configs rewritten for the smaller volumetric problem: bottleneck (2, 4, 4) → output (8, 16, 16).
  Each upsample is a clean k=4, s=2, p=1 doubling, with refinement layers (k=3, s=1) interleaved for higher
  num_deconv_layers. Sweepable values are 2, 3, 4, 5.                                                              
                                                                                                                                                                       
  Same patterns kept                                                                                                                                                   
  - ResidualBlock3D with configurable activation (uses ELU in decoder, ReLU in encoder, matching the 2D fix)
  - Single-activation skip path (no double-activation bug)                                                                                                             
  - _verify_output_size runs at construction, raises if shapes don't match
  - VAE3D wrapper preserves the same API: forward / encode / decode / set_inference_mode                                                                               
                                                                                                                                                                       
  Default VAE3D() instantiates the network sized for your (2, 8, 16, 16) local maps.                                                                                   
                                                                                       
"""

class ResidualBlock3D(nn.Module):
    """
    3D residual block — refines features WITHOUT changing shape.
    Pre-activation pattern: act → Conv3d → BN → act → Conv3d → BN → add input.
    """
    def __init__(self, channels, activation=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            activation(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            activation(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class VolumeEncoder(nn.Module):
    """
    3D encoder mirroring the 2D ImgEncoder.
    Input volume: (B, input_dim, D, H, W). Defaults sized for local maps (2, 8, 16, 16).
    """
    def __init__(self, input_dim, latent_dim,
                 input_shape=(8, 16, 16),     # (D, H, W)
                 num_conv_layers=3,           # ← swept: 2, 3, 4
                 use_residual=True,
                 residual_every=2):
        super().__init__()
        self.input_dim       = input_dim
        self.latent_dim      = latent_dim
        self.input_shape     = tuple(input_shape)
        self.num_conv_layers = num_conv_layers
        self.use_residual    = use_residual
        self.residual_every  = residual_every
        self.relu            = nn.ReLU()

        # 3D activations are heavier than 2D — keep channel growth modest
        self.channel_plan = [
            (input_dim, 16),
            (16,        32),
            (32,        64),
            (64,        64),
            (64,        64),
        ]

        # ── Residual block placement (deep half only) ─────────────────────────
        self.residual_at = set()
        if use_residual and num_conv_layers >= 3:
            deep_start = num_conv_layers // 2
            deep_layers = list(range(deep_start, num_conv_layers))
            self.residual_at = set(
                deep_layers[i]
                for i in range(0, len(deep_layers), residual_every)
            )

        print(f"  VolumeEncoder: {num_conv_layers} conv layers | "
              f"residual every {residual_every} deep layers | "
              f"residual at: {sorted(self.residual_at) or 'none'}")

        # ── Build layers ──────────────────────────────────────────────────────
        self.conv_layers = nn.ModuleList()
        self.bn_layers   = nn.ModuleList()
        self.res_blocks  = nn.ModuleDict()
        self.skip_layers = nn.ModuleDict()
        self.skip_bn     = nn.ModuleDict()

        for i in range(num_conv_layers):
            in_ch, out_ch = self.channel_plan[i]
            stride = 2 if i < 2 else (2 if i % 2 == 0 else 1)

            self.conv_layers.append(
                nn.Conv3d(in_ch, out_ch, kernel_size=3,
                          stride=stride, padding=1, bias=False)
            )
            self.bn_layers.append(nn.BatchNorm3d(out_ch))

            if i in self.residual_at:
                self.res_blocks[str(i)] = ResidualBlock3D(out_ch)

            if i > 0 and i % 2 == 0:
                prev_ch = self.channel_plan[i - 1][1]
                if prev_ch != out_ch:
                    self.skip_layers[str(i)] = nn.Conv3d(
                        prev_ch, out_ch, kernel_size=1, stride=2, bias=False
                    )
                    self.skip_bn[str(i)] = nn.BatchNorm3d(out_ch)

        self.flat_size, self.bottleneck_shape = self._compute_flat_size()
        print(f"  VolumeEncoder bottleneck: {self.bottleneck_shape}, "
              f"flat: {self.flat_size}")

        self.dense0 = nn.Linear(self.flat_size, 256)
        self.dense1 = nn.Linear(256, 2 * self.latent_dim)

    def _compute_flat_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_dim, *self.input_shape)
            x = dummy
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
            return x.view(1, -1).shape[1], tuple(x.shape[1:])

    def forward(self, vol):
        x = vol
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


class VolumeDecoder(nn.Module):
    """
    3D decoder mirroring the 2D ImgDecoder.
    Default sizing: bottleneck (64, 2, 4, 4) → output (output_dim, 8, 16, 16).
    The deconv configs are decoupled from the encoder — connection is via the
    latent space, exactly like the 2D version.
    """
    def __init__(self, output_dim=2, latent_dim=64, with_logits=False,
                 output_shape=(8, 16, 16),       # (D, H, W) of reconstruction
                 bottleneck_shape=(2, 4, 4),     # (D, H, W) input to first deconv
                 bottleneck_channels=64,
                 num_deconv_layers=3,            # ← swept: 2, 3, 4, 5
                 use_residual=True,
                 residual_every=2):
        super().__init__()
        self.with_logits         = with_logits
        self.n_channels          = output_dim
        self.output_shape        = tuple(output_shape)
        self.bottleneck_shape    = tuple(bottleneck_shape)
        self.bottleneck_channels = bottleneck_channels
        self.num_deconv_layers   = num_deconv_layers
        self.use_residual        = use_residual
        self.residual_every      = residual_every
        self.elu = nn.ELU()

        # ── Dense layers — expand latent → flat bottleneck volume ─────────────
        bD, bH, bW = self.bottleneck_shape
        flat = bottleneck_channels * bD * bH * bW

        self.dense0 = nn.Linear(latent_dim, 256)
        self.dense1 = nn.Linear(256, flat)

        # ── Deconv layer pool ─────────────────────────────────────────────────
        # Each entry: (in_ch, out_ch, kernel, stride, padding, output_padding 3-tuple)
        # Designed for bottleneck (2, 4, 4) → output (8, 16, 16).
        # Final layer maps to output_dim.
        c = bottleneck_channels
        self.ALL_DECONV_CFGS = {
            2: [
                (c,          c // 2, 4, 2, 1, (0, 0, 0)),  # x2 spatial
                (c // 2, output_dim, 4, 2, 1, (0, 0, 0)),  # x2 — final
            ],
            3: [
                (c,             c, 3, 1, 1, (0, 0, 0)),  # refinement
                (c,        c // 2, 4, 2, 1, (0, 0, 0)),  # x2
                (c // 2, output_dim, 4, 2, 1, (0, 0, 0)),  # x2 — final
            ],
            4: [
                (c,             c, 3, 1, 1, (0, 0, 0)),  # refinement
                (c,        c // 2, 4, 2, 1, (0, 0, 0)),  # x2
                (c // 2,   c // 2, 3, 1, 1, (0, 0, 0)),  # refinement
                (c // 2, output_dim, 4, 2, 1, (0, 0, 0)),  # x2 — final
            ],
            5: [
                (c,             c, 3, 1, 1, (0, 0, 0)),  # refinement
                (c,        c // 2, 4, 2, 1, (0, 0, 0)),  # x2
                (c // 2,   c // 2, 3, 1, 1, (0, 0, 0)),  # refinement
                (c // 2,   c // 4, 4, 2, 1, (0, 0, 0)),  # x2
                (c // 4, output_dim, 3, 1, 1, (0, 0, 0)),  # final refinement
            ],
        }

        if num_deconv_layers not in self.ALL_DECONV_CFGS:
            raise ValueError(
                f"num_deconv_layers={num_deconv_layers} not supported. "
                f"Choose from {list(self.ALL_DECONV_CFGS.keys())}"
            )

        deconv_cfgs = self.ALL_DECONV_CFGS[num_deconv_layers]

        # ── Residual placement — only on non-final upsample layers ────────────
        upsample_idxs = [
            i for i, (_, _, _, s, _, _) in enumerate(deconv_cfgs)
            if s > 1 and i < len(deconv_cfgs) - 1
        ]
        self.residual_at = set()
        if use_residual and len(upsample_idxs) > 0:
            self.residual_at = set(
                upsample_idxs[i]
                for i in range(0, len(upsample_idxs), residual_every)
            )

        print(f"  VolumeDecoder: {num_deconv_layers} deconv layers | "
              f"residual at: {sorted(self.residual_at) or 'none'}")

        # ── Build layers ──────────────────────────────────────────────────────
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
            self.deconv_bn.append(
                nn.BatchNorm3d(out_ch) if not is_final else nn.Identity()
            )

            if i in self.residual_at:
                self.res_blocks[str(i)] = ResidualBlock3D(out_ch, activation=nn.ELU)

            # ── Skip connection — generalised 2D rule to 3D output_padding ────
            if in_ch != out_ch and s > 1 and not is_final:
                op_d = op[0] + (k - 1) - 2 * p
                op_h = op[1] + (k - 1) - 2 * p
                op_w = op[2] + (k - 1) - 2 * p
                if 0 <= op_d < s and 0 <= op_h < s and 0 <= op_w < s:
                    skip_k, skip_p, skip_op = 1, 0, (op_d, op_h, op_w)
                else:
                    skip_k, skip_p, skip_op = k, p, op  # fallback: same kernel as main
                self.skip_layers[str(i)] = nn.ConvTranspose3d(
                    in_ch, out_ch, kernel_size=skip_k,
                    stride=s, padding=skip_p, output_padding=skip_op, bias=False
                )
                self.skip_bn[str(i)] = nn.BatchNorm3d(out_ch)

        # ── Verify output size ────────────────────────────────────────────────
        self._verify_output_size()

    def _verify_output_size(self):
        """Sanity check — confirm we reach exactly (1, output_dim, *output_shape)."""
        with torch.no_grad():
            dummy = torch.zeros(1, self.bottleneck_channels, *self.bottleneck_shape)
            x = dummy
            for i, (deconv, bn) in enumerate(
                zip(self.deconv_layers, self.deconv_bn)
            ):
                identity = x
                out = bn(deconv(x))

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

        expected = (1, self.n_channels, *self.output_shape)
        if tuple(x.shape) != expected:
            raise RuntimeError(
                f"Decoder output shape {tuple(x.shape)} != {expected}. "
                f"Fix the deconv configs for num_deconv_layers={self.num_deconv_layers}."
            )
        print(f"  VolumeDecoder output verified: {tuple(x.shape)} ✅")

    def forward(self, z):
        x = self.elu(self.dense0(z))
        x = self.elu(self.dense1(x))
        x = x.view(x.size(0), self.bottleneck_channels, *self.bottleneck_shape)

        for i, (deconv, bn) in enumerate(
            zip(self.deconv_layers, self.deconv_bn)
        ):
            is_final = (i == len(self.deconv_layers) - 1)
            identity = x
            out = bn(deconv(x))

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
    """
    3D variational autoencoder mirroring the 2D VAE.
    Default sizing matches local map data: input (input_dim=2, 8, 16, 16).
    """
    def __init__(self, input_dim=2, latent_dim=64, with_logits=False,
                 inference_mode=False,
                 input_shape=(8, 16, 16),
                 num_conv_layers=3,            # ← swept
                 use_residual=True,
                 residual_every=2,
                 num_deconv_layers=3,          # ← swept
                 bottleneck_channels=64,
                 bottleneck_shape=(2, 4, 4)):
        super().__init__()
        self.with_logits    = with_logits
        self.input_dim      = input_dim
        self.latent_dim     = latent_dim
        self.inference_mode = inference_mode

        self.encoder = VolumeEncoder(
            input_dim       = input_dim,
            latent_dim      = latent_dim,
            input_shape     = input_shape,
            num_conv_layers = num_conv_layers,
            use_residual    = use_residual,
            residual_every  = residual_every,
        )
        self.vol_decoder = VolumeDecoder(
            output_dim          = input_dim,
            latent_dim          = latent_dim,
            with_logits         = with_logits,
            output_shape        = input_shape,
            bottleneck_shape    = bottleneck_shape,
            bottleneck_channels = bottleneck_channels,
            num_deconv_layers   = num_deconv_layers,
            use_residual        = use_residual,
            residual_every      = residual_every,
        )

        self.mean_params   = Lambda(lambda x: x[:, :latent_dim])
        self.logvar_params = Lambda(lambda x: x[:, latent_dim:])

    def forward(self, vol):
        z = self.encoder(vol)

        mean   = self.mean_params(z)
        # clamp logvar — prevents std from exploding or vanishing
        logvar = torch.clamp(self.logvar_params(z), min=-4.0, max=4.0)

        std = torch.exp(0.5 * logvar)
        eps = torch.zeros_like(std) if self.inference_mode else torch.randn_like(std)
        z_sampled = mean + eps * std

        vol_recon = self.vol_decoder(z_sampled)
        return vol_recon, mean, logvar, z_sampled

    def encode(self, vol):
        z      = self.encoder(vol)
        mean   = self.mean_params(z)
        logvar = torch.clamp(self.logvar_params(z), min=-4.0, max=4.0)
        std    = torch.exp(0.5 * logvar)
        eps    = torch.zeros_like(std) if self.inference_mode else torch.randn_like(std)
        return mean + eps * std, mean, std

    def decode(self, z):
        vol_recon = self.vol_decoder(z)
        if self.with_logits:
            return torch.sigmoid(vol_recon)
        return vol_recon

    def set_inference_mode(self, mode):
        self.inference_mode = mode
