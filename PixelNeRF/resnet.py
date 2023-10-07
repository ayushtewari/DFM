import torch
import torch.nn as nn

# from .utils import load_state_dict_from_url
import torch.nn.functional as F
from utils import *
from layers import RandomOrLearnedSinusoidalPosEmb, SinusoidalPosEmb
from einops import rearrange, reduce


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockTimeEmbed(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        time_emb_dim=None,
    ):
        super(BasicBlockTimeEmbed, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, planes * 2))
            if exists(time_emb_dim)
            else None
        )

    # def forward(self, x, time_emb=None):
    def forward(self, inp):
        x = inp[0]
        time_emb = inp[1]
        identity = x
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            # print(f"time_emb.shape: {time_emb.shape}")
            time_emb = self.mlp(time_emb)
            # print(f"time_emb.shape: {time_emb.shape}")
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        out = self.conv1(x)
        out = self.bn1(out)
        if exists(scale_shift):
            scale, shift = scale_shift
            out = out * (scale + 1) + shift
        # print(f"out.shape: {out.shape}, scale_shift: {scale.shape}, {shift.shape}")

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class PixelNeRFTimeEmbed(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        # time_emb_dim=None,
        use_first_pool=False,
        upsample_interp="bilinear",
        in_channels=3,
        cond_feats_dim=4,
    ):
        super(PixelNeRFTimeEmbed, self).__init__()
        self.model = ResNetTimeEmbed(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
            # time_emb_dim,
            use_first_pool,
            upsample_interp,
            in_channels=in_channels,
            cond_feats_dim=cond_feats_dim,
        )
        self.out = nn.Sequential(nn.Conv2d(512, 512, 1),)

    def forward(self, inp, time_emb=None, custom_size=None):
        latent = self.model(inp, time_emb=time_emb, custom_size=custom_size)
        out = self.out(latent)
        return out


class ResNetTimeEmbed(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        # time_emb_dim=None,
        use_first_pool=False,
        upsample_interp="bilinear",
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        in_channels=3,
        cond_feats_dim=4,
    ):
        super(ResNetTimeEmbed, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.use_first_pool = use_first_pool
        self.upsample_interp = upsample_interp

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.cond_feats_dim = cond_feats_dim
        self.conv1_ch7 = nn.Conv2d(
            3 + self.cond_feats_dim,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        time_emb_dim = self.inplanes * 4
        self.layer1 = self._make_layer(block, 64, layers[0], time_emb_dim=time_emb_dim)

        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            time_emb_dim=time_emb_dim,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            time_emb_dim=time_emb_dim,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            time_emb_dim=time_emb_dim,
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlockTimeEmbed):
                    nn.init.constant_(m.bn2.weight, 0)
        self.index_interp = ("bilinear",)
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(self.inplanes)
            fourier_dim = self.inplanes
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def _make_layer(
        self, block, planes, blocks, stride=1, dilate=False, time_emb_dim=None,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                time_emb_dim=time_emb_dim,
            ).cuda()
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    time_emb_dim=time_emb_dim,
                ).cuda()
            )

        return nn.Sequential(*layers)
        # return layers

    def forward(self, x, time_emb=None, custom_size=None):
        # See note [TorchScript super()]
        b, c, h, w = x.shape
        if c == 3 + self.cond_feats_dim:
            x = self.conv1_ch7(x)
        elif c == 3:
            x = self.conv1(x)
        else:
            raise ValueError("Input image should have 3 or 7 channels")
        x = self.bn1(x)
        x = self.relu(x)
        latents = [x]
        # print("x.shape", x.shape)
        if self.use_first_pool:
            x = self.maxpool(x)

        if time_emb is not None:
            time_emb = self.time_mlp(time_emb)

        for layer in self.layer1:
            inp = (x, time_emb)
            x = layer(inp)
        latents.append(x)
        # print("x.shape", x.shape)

        for layer in self.layer2:
            inp = (x, time_emb)
            x = layer(inp)
        latents.append(x)
        # print("x.shape", x.shape)

        for layer in self.layer3:
            inp = (x, time_emb)
            x = layer(inp)
        latents.append(x)
        # print("x.shape", x.shape)

        # for layer in self.layer4:
        #     inp = (x, time_emb)
        #     x = layer(inp)
        # latents.append(x)
        # print("x.shape", x.shape)

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
        return self.latent
