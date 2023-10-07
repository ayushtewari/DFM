from torch import nn
import torch

#  import torch_scatter
import torch.autograd.profiler as profiler
from einops import rearrange, repeat

from utils import *
from layers import RandomOrLearnedSinusoidalPosEmb, SinusoidalPosEmb


def combine_interleaved(t, inner_dims=(1,), agg_type="average"):
    if len(inner_dims) == 1 and inner_dims[0] == 1:
        return t
    t = t.reshape(-1, *inner_dims, *t.shape[1:])
    if agg_type == "average":
        t = torch.mean(t, dim=1)
    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t


# Resnet Blocks
class ResnetBlockFCTimeEmbed(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(
        self, size_in, size_out=None, size_h=None, beta=0.0, time_emb_dim=None,
    ):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, size_h * 2))
            if exists(time_emb_dim)
            else None
        )

    def forward(self, x, time_emb=None):
        with profiler.record_function("resblock"):
            net = self.fc_0(self.activation(x))
            if exists(self.mlp) and exists(time_emb):
                time_emb = self.mlp(time_emb)
                # print(f"time_emb: {time_emb.shape}, net: {net.shape}, x: {x.shape}")
                time_emb = rearrange(time_emb, "b c -> b 1 c")
                scale_shift = time_emb.chunk(2, dim=-1)
                scale, shift = scale_shift
                # print(f"scale: {scale.shape}, shift: {shift.shape}, net: {net.shape}")
                net = net * (scale + 1) + shift

            dx = self.fc_1(self.activation(net))
            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            return x_s + dx


class ResnetFCTimeEmbed(nn.Module):
    def __init__(
        self,
        d_in,
        d_out=4,
        n_blocks=5,
        d_latent=0,
        d_hidden=128,
        beta=0.0,
        combine_layer=1000,
        combine_type="average",
        use_spade=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
    ):
        """
        :param d_in input size
        :param d_out output size
        :param n_blocks number of Resnet blocks
        :param d_latent latent size, added in each resnet block (0 = disable)
        :param d_hidden hiddent dimension throughout network
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        """
        super().__init__()
        if d_in > 0:
            self.lin_in = nn.Linear(d_in, d_hidden)
            nn.init.constant_(self.lin_in.bias, 0.0)
            nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")

        self.lin_out = nn.Linear(d_hidden, 4)
        self.lin_out_feats = nn.Linear(d_hidden, d_out)

        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.n_blocks = n_blocks
        self.d_latent = d_latent
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden

        self.combine_layer = combine_layer
        self.combine_type = combine_type
        self.use_spade = use_spade
        time_emb_dim = d_hidden * 4
        self.blocks = nn.ModuleList(
            [
                ResnetBlockFCTimeEmbed(d_hidden, beta=beta, time_emb_dim=time_emb_dim)
                for i in range(n_blocks)
            ]
        )

        if d_latent != 0:
            n_lin_z = min(combine_layer, n_blocks)
            self.lin_z = nn.ModuleList(
                [nn.Linear(d_latent, d_hidden) for i in range(n_lin_z)]
            )
            for i in range(n_lin_z):
                nn.init.constant_(self.lin_z[i].bias, 0.0)
                nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")

            if self.use_spade:
                self.scale_z = nn.ModuleList(
                    [nn.Linear(d_latent, d_hidden) for _ in range(n_lin_z)]
                )
                for i in range(n_lin_z):
                    nn.init.constant_(self.scale_z[i].bias, 0.0)
                    nn.init.kaiming_normal_(self.scale_z[i].weight, a=0, mode="fan_in")

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(self.d_hidden)
            fourier_dim = self.d_hidden
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(
        self,
        zx,
        ns=-1,
        combine_inner_dims=(1,),
        combine_index=None,
        dim_size=None,
        time_emb=None,
        return_mlp_input=False,
    ):
        """
        :param zx (..., d_latent + d_in)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """
        with profiler.record_function("resnetfc_infer"):
            assert zx.size(-1) == self.d_latent + self.d_in
            if self.d_latent > 0:
                z = zx[..., : self.d_latent]
                x = zx[..., self.d_latent :]
            else:
                x = zx
            if self.d_in > 0:
                x = self.lin_in(x)
            else:
                x = torch.zeros(self.d_hidden, device=zx.device)

            if time_emb is not None:
                t = time_emb.clone()
                time_emb = self.time_mlp(time_emb)

            for blkid in range(self.n_blocks):
                if blkid == self.combine_layer:
                    x = rearrange(x, "(b ns) n  ch -> b ns n ch", ns=ns)
                    x = x.mean(dim=1)

                    #     first average the context, then averate the noisy target
                    # print("averaging context feats separately", x.shape)
                    # x_ctxt = x[:, :-1].mean(dim=1)
                    # x_ctxt_trgt = torch.cat([x_ctxt[:, None], x[:, -1:]], dim=1)
                    # x = x_ctxt_trgt.mean(dim=1)
                    #
                    if time_emb is not None:
                        time_emb = rearrange(time_emb, "(b ns) c -> b ns c", ns=ns)
                        time_emb = time_emb[:, -1, :]  # only noisy time relevant now
                    if return_mlp_input:
                        return x
                if self.d_latent > 0 and blkid < self.combine_layer:
                    tz = self.lin_z[blkid](z)
                    if self.use_spade:
                        sz = self.scale_z[blkid](z)
                        x = sz * x + tz
                    else:
                        x = x + tz

                x = self.blocks[blkid](x, time_emb=time_emb)
            if self.d_out == 4:
                out = self.lin_out(self.activation(x))
            else:
                out = self.lin_out_feats(self.activation(x))
            return out

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            n_blocks=conf.get_int("n_blocks", 5),
            d_hidden=conf.get_int("d_hidden", 128),
            beta=conf.get_float("beta", 0.0),
            combine_layer=conf.get_int("combine_layer", 1000),
            combine_type=conf.get_string("combine_type", "average"),  # average | max
            use_spade=conf.get_bool("use_spade", False),
            **kwargs,
        )
