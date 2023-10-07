import torch, torchvision
from torch import nn
import torch.nn.functional as F
import sys
import os
import numpy as np
from einops import rearrange, repeat

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers import *
from utils import *
from PixelNeRF.pixelnerf_helpers import *
from functools import partial


class PixelNeRFEncoder(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        # time embeddings
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        # layers
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_downsample = Downsample(mid_dim, mid_dim)
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )
        self.out = nn.Sequential(nn.Conv2d(512, 512, 1),)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        x = self.init_conv(x)
        t = self.time_mlp(time)
        h = []
        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = downsample(x)
            if not i == len(self.downs) - 1:
                h.append(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x = self.mid_downsample(x)
        h.append(x)
        latents = h
        align_corners = True
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i], latent_sz, mode="bilinear", align_corners=align_corners,
            )
        self.latent = torch.cat(latents, dim=1)
        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
        out = self.out(self.latent)
        return out


class PixelNeRFEncoderOriginal(nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
        in_ch=3,
    ):
        super().__init__()
        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        print(f"use_first_pool: {use_first_pool}")
        norm_layer = get_norm_layer(norm_type)
        print("Using torchvision", backbone, "encoder")
        self.model = getattr(torchvision.models, backbone)(
            pretrained=pretrained, norm_layer=norm_layer
        )
        if in_ch != 3:
            self.model.conv1 = nn.Conv2d(
                in_ch,
                self.model.conv1.weight.shape[0],
                self.model.conv1.kernel_size,
                self.model.conv1.stride,
                self.model.conv1.padding,
                padding_mode=self.model.conv1.padding_mode,
            )
        # Following 2 lines need to be uncommented for older configs
        self.model.fc = nn.Sequential()
        self.model.avgpool = nn.Sequential()
        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )
        self.out = nn.Sequential(nn.Conv2d(512, 512, 1),)

    def forward(
        self, x, custom_size=None, t=0,
    ):
        # print(f"input shape: {x.shape}, scale: {self.feature_scale}")
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)

        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            # print(f"latents shape: {latents[-1].shape}")
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
                # print(f"latents shape: {latents[-1].shape}")
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
                # print(f"latents shape: {latents[-1].shape}")
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
                # print(f"latents shape: {latents[-1].shape}")
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)
                # print(f"latents shape: {latents[-1].shape}")

            self.latents = latents
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]
            for i in range(len(latents)):
                # print(latent_sz, latents[i].shape, custom_size)
                latents[i] = F.interpolate(
                    latents[i],
                    tuple(latent_sz) if custom_size is None else custom_size,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )
            self.latent = torch.cat(latents, dim=1)

        # print(f"latent shape: {self.latent.shape}", flush=True)
        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
        out = self.out(self.latent)
        return out
