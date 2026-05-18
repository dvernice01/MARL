import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block that refines features WITHOUT changing shape.
    Input and output are identical in (channels, H, W).
    Uses: Relu -> Conv → BN → ELU → Conv → BN → add input → ELU
    """
    def __init__(self, channels, activation=nn.ReLU):
        super().__init__()                                                                                                                                           
        act = activation()
        self.block = nn.Sequential(                                                                                                                                  
            activation(), 
            # 103: channels, channels va bene? 3x3 come kernel va bene?
            # 103: viene chiamato come: self.res_blocks[str(i)] = ResidualBlock(out_ch)   
            # 103: conviene usare la "Dilated convolution"? Consigliato da Claude ma per adesso ignorato   
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),                               
            activation(),                                                                                                                                            
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),                                                                                                                                
        )   

    def forward(self, x):
        return x + self.block(x)


class ImgEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim,
                 num_conv_layers=4,
                 use_residual=True,
                 residual_every=2,
                 use_skip=True):    # ← swept: 1, 2, 3 (every N deep layers)
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
            (64,        128),
            (128,       128),
            (128,       128),
        ]

        # ── Residual block placement ───────────────────────────────────────────
        # Residuals are only placed in the DEEP half of the network
        # (shallow layers detect simple features, don't benefit from refinement)
        # Within the deep half, place one every `residual_every` layers
        #
        # Example with num_conv_layers=6, residual_every=2:
        #   deep half starts at layer 3
        #   candidates: 3, 4, 5
        #   every 2: place at 3, 5  → skip layer 4
        #
        # Example with num_conv_layers=6, residual_every=3:
        #   candidates: 3, 4, 5
        #   every 3: place at 3 only
        #
        # Example with num_conv_layers=4, residual_every=1:
        #   deep half starts at layer 2
        #   candidates: 2, 3
        #   every 1: place at 2, 3  → original behavior

        self.residual_at = set()
        if use_residual and num_conv_layers >= 3:
            # 103: residual dalla metà in poi è ok? ogni 1 o 2 layers?
            deep_start = num_conv_layers // 2          # first deep layer
            deep_layers = list(range(deep_start, num_conv_layers))
            # pick every Nth layer from the deep half
            self.residual_at = set(
                deep_layers[i]
                for i in range(0, len(deep_layers), residual_every)
            )

        print(f"  Encoder: {num_conv_layers} conv layers | "
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
            # 103: ha senso usare stride in questo modo? 
            # 103: 2 stride nei primi 2 layer per dimezzare subito, poi alternanza di stride 2 e 1 per bilanciare downsampling e raffinamento?

            self.conv_layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=3,
                          stride=stride, padding=1, bias=False)
            )
            self.bn_layers.append(nn.BatchNorm2d(out_ch))

            if i in self.residual_at:
                self.res_blocks[str(i)] = ResidualBlock(out_ch)

            if use_skip and i > 0 and i % 2 == 0:
                prev_ch = self.channel_plan[i - 1][1]
                if prev_ch != out_ch:
                    # 103: skip layers con kernel 1 e stride 2?
                    self.skip_layers[str(i)] = nn.Conv2d(
                        prev_ch, out_ch, kernel_size=1, stride=2, bias=False
                    )
                    self.skip_bn[str(i)] = nn.BatchNorm2d(out_ch)

        self.flat_size = self._compute_flat_size()
        print(f"  Encoder flat size: {self.flat_size}")

        self.dense0 = nn.Linear(self.flat_size, 4 * self.latent_dim)
        self.dense1 = nn.Linear(4 * self.latent_dim, 2 * self.latent_dim)

    def _compute_flat_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_dim, 270, 480)
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

class ImgDecoder(nn.Module):
    def __init__(self, input_dim=1, latent_dim=64, with_logits=False,
                 num_deconv_layers=5,    # ← swept: 3, 4, 5, 6, 7
                 use_residual=True,
                 residual_every=2,
                 use_skip=True):   # ← swept
        super().__init__()
        self.with_logits       = with_logits
        self.n_channels        = input_dim
        self.num_deconv_layers = num_deconv_layers
        self.use_residual      = use_residual
        self.residual_every    = residual_every
        self.elu = nn.ELU()

        # ── Dense layers — FIXED, identical to original ────────────────────────
        self.dense0 = nn.Linear(latent_dim, latent_dim * 2)
        self.dense1 = nn.Linear(latent_dim * 2, latent_dim * 4)
        self.dense2 = nn.Linear(latent_dim * 4, 9 * 15 * 128) # 103: 9x15x128 deriva dal codice originale, va bene anche nel mio caso?

        # ── Deconv layer pool — all possible layers in order ───────────────────
        # Each entry: (in_ch, out_ch, kernel, stride, padding, output_padding)
        # The full sequence goes from (128, 9, 15) → (1, 270, 480)
        # We define MORE layers than needed and select a subset based on depth.
        #
        # Spatial progression (full 7-layer version):
        # start:    (128,  9,  15)
        # deconv0:  (128,  9,  15)  stride=1  refinement
        # deconv1:  (128, 17,  30)  stride=2  x2
        # deconv2:  ( 64, 17,  30)  stride=1  refinement + channel reduction
        # deconv3:  ( 64, 34,  60)  stride=2  x2
        # deconv4:  ( 32, 34,  60)  stride=1  refinement + channel reduction
        # deconv5:  ( 16,135, 241)  stride=4  x4 (big jump to near-final res)
        # deconv6:  (  1,270, 480)  stride=2  final output
        #
        # For fewer layers we skip refinement layers and use larger strides
        # to still reach (270, 480) from (9, 15).

        self.ALL_DECONV_CFGS = {
            # key = num_deconv_layers → list of (in_ch, out_ch, k, s, pad, out_pad)
            3: [
                # aggressive — must cover 9→270 (x30) and 15→480 (x32) in 2 steps + final
                (128, 64, 6, 4, 1, (0, 0)),    # (64,  34,  60)   x4
                ( 64, 16, 6, 4, 0, (0, 1)),    # (16, 135, 241)   x4
                ( 16,  1, 4, 2, 2, (0, 0)),    # (1,  270, 480)   x2  final
            ],
            4: [
                (128, 64, 5, 2, 2, (0, 1)),    # (64,  17,  30)   x2
                ( 64, 32, 6, 4, 2, (0, 0)),    # (32,  68, 120)   x4
                ( 32, 16, 6, 2, 0, (0, 1)),    # (16, 135, 241)   x2
                ( 16,  1, 4, 2, 2, (0, 0)),    # (1,  270, 480)   x2  final
            ],
            5: [
                # original architecture
                (128, 128, 3, 1, 1, (0, 0)),   # (128,  9,  15)   refinement
                (128,  64, 5, 2, 2, (0, 1)),   # (64,  17,  30)   x2
                ( 64,  32, 6, 4, 2, (0, 0)),   # (32,  68, 120)   x4
                ( 32,  16, 6, 2, 0, (0, 1)),   # (16, 135, 241)   x2
                ( 16,   1, 4, 2, 2, (0, 0)),   # (1,  270, 480)   x2  final
            ],
            6: [
                (128, 128, 3, 1, 1, (0, 0)),   # (128,  9,  15)   refinement
                (128,  64, 5, 2, 2, (0, 1)),   # (64,  17,  30)   x2
                ( 64,  64, 3, 1, 1, (0, 0)),   # (64,  17,  30)   refinement
                ( 64,  32, 6, 4, 2, (0, 0)),   # (32,  68, 120)   x4
                ( 32,  16, 6, 2, 0, (0, 1)),   # (16, 135, 241)   x2
                ( 16,   1, 4, 2, 2, (0, 0)),   # (1,  270, 480)   x2  final
            ],
            7: [
                (128, 128, 3, 1, 1, (0, 0)),   # (128,  9,  15)   refinement
                (128,  64, 5, 2, 2, (0, 1)),   # (64,  17,  30)   x2
                ( 64,  64, 3, 1, 1, (0, 0)),   # (64,  17,  30)   refinement
                ( 64,  32, 6, 4, 2, (0, 0)),   # (32,  68, 120)   x4
                ( 32,  32, 3, 1, 1, (0, 0)),   # (32,  68, 120)   refinement
                ( 32,  16, 6, 2, 0, (0, 1)),   # (16, 135, 241)   x2
                ( 16,   1, 4, 2, 2, (0, 0)),   # (1,  270, 480)   x2  final
            ],
        }

        if num_deconv_layers not in self.ALL_DECONV_CFGS:
            raise ValueError(
                f"num_deconv_layers={num_deconv_layers} not supported. "
                f"Choose from {list(self.ALL_DECONV_CFGS.keys())}"
            )

        deconv_cfgs = self.ALL_DECONV_CFGS[num_deconv_layers]

        # ── Residual placement — only on non-final upsample layers ────────────
        # skip refinement layers (stride=1) and final output layer
        # upsample_idxs = [
        #     i for i, (_, _, _, s, _, _) in enumerate(deconv_cfgs)
        #     if s > 1 and i < len(deconv_cfgs) - 1   # not final
        # ]
        self.residual_at = set()
        if use_residual and num_deconv_layers >= 3:
            deep_start = num_deconv_layers // 2          # first deep layer
            deep_layers = list(range(deep_start, num_deconv_layers))
            # pick every Nth layer from the deep half
            self.residual_at = set(
                deep_layers[i]
                for i in range(0, len(deep_layers), residual_every)
            )
        # if use_residual and len(upsample_idxs) > 0:
        #     self.residual_at = set(
        #         upsample_idxs[i]
        #         for i in range(0, len(upsample_idxs), residual_every)
        #     )

        print(f"  Decoder: {num_deconv_layers} deconv layers | "
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
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k,
                                   stride=s, padding=p,
                                   output_padding=op, bias=False)
            )
            # no BN on final layer — sigmoid handles normalization
            self.deconv_bn.append(
                nn.BatchNorm2d(out_ch) if not is_final else nn.Identity()
            )

            if i in self.residual_at:
                self.res_blocks[str(i)] = ResidualBlock(out_ch, activation=nn.ELU)

            # skip connection when channels change and not final
            if use_skip and in_ch != out_ch and s > 1 and not is_final:                                                                                                                       
                op_h = op[0] + (k - 1) - 2 * p                                                                                                                                   
                op_w = op[1] + (k - 1) - 2 * p
                if 0 <= op_h < s and 0 <= op_w < s:                                                                                                                              
                    skip_k, skip_p, skip_op = 1, 0, (op_h, op_w)
                else:                                                                                                                                                            
                    skip_k, skip_p, skip_op = k, p, op  # fallback: same kernel as main
                self.skip_layers[str(i)] = nn.ConvTranspose2d(                                                                                                                   
                    in_ch, out_ch, kernel_size=skip_k,                                                                                                                           
                    stride=s, padding=skip_p, output_padding=skip_op, bias=False
                )                                                                                                                                                                
                self.skip_bn[str(i)] = nn.BatchNorm2d(out_ch) 

        # ── Verify output size ────────────────────────────────────────────────
        self._verify_output_size()

    def _verify_output_size(self):
        """Sanity check — confirm we reach exactly (1, 270, 480)."""
        with torch.no_grad():
            dummy = torch.zeros(1, 128, 9, 15)
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

        expected = (1, 1, 270, 480)
        if tuple(x.shape) != expected:
            raise RuntimeError(
                f"Decoder output shape {tuple(x.shape)} != {expected}. "
                f"Fix the deconv configs for num_deconv_layers={self.num_deconv_layers}."
            )
        print(f"  Decoder output verified: {tuple(x.shape)} ✅")

    def forward(self, z):
        # fixed dense expansion — identical to original
        x = self.elu(self.dense0(z))
        x = self.elu(self.dense1(x))
        x = self.elu(self.dense2(x))
        x = x.view(x.size(0), 128, 9, 15)
        # 103: in questo caso uso nn.ELU perchè lo faceva il codice originale

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
    

class VAE(nn.Module):
    def __init__(self, input_dim=1, latent_dim=64, with_logits=False,
                 inference_mode=False,
                 num_conv_layers=4,      # ← swept
                 use_residual=True,
                 residual_every=2,
                 num_deconv_layers=5,
                 use_skip=True):   # ← swept
        
        super().__init__()
        self.with_logits    = with_logits
        self.input_dim      = input_dim
        self.latent_dim     = latent_dim
        self.inference_mode = inference_mode

        self.encoder = ImgEncoder(
            input_dim       = input_dim,
            latent_dim      = latent_dim,
            num_conv_layers = num_conv_layers,
            use_residual    = use_residual,
            residual_every   = residual_every,
            use_skip        = use_skip,
        )
        self.img_decoder = ImgDecoder(
            input_dim        = 1,
            latent_dim       = latent_dim,
            with_logits      = with_logits,
            num_deconv_layers = num_deconv_layers,  
            use_residual     = use_residual,
            residual_every   = residual_every,
            use_skip         = use_skip,
        )

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

