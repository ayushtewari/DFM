import torch
from torch import nn
import torch.nn.functional as F
import sys
import os
from einops import rearrange, repeat

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PixelNeRF.renderer import *
from PixelNeRF.resnetfc import *
import numpy as np
from PixelNeRF.resnetfc_time_embed import *


def pixel_aligned_features_cond(
    coords_3d_world,
    cam2world,
    intrinsics,
    img_features,
    viewdirs=None,
    interp="bilinear",
):
    # Args:
    #     coords_3d_world: shape (b, n, 3)
    #     cam2world: camera pose of shape (..., 4, 4)
    # project 3d points to 2D
    c3d_world_hom = homogenize_points(coords_3d_world)
    c3d_cam_hom = transform_world2cam(c3d_world_hom, cam2world)
    c2d_cam, depth = project(c3d_cam_hom, intrinsics.unsqueeze(1))

    if viewdirs is not None:
        viewdirs_cam = transform_world2cam(viewdirs, cam2world[..., :3, :3])
    else:
        viewdirs_cam = None

    # now between 0 and 1. Map to -1 and 1
    c2d_norm = (c2d_cam - 0.5) * 2
    c2d_norm = rearrange(c2d_norm, "b n ch -> b n () ch")
    c2d_norm = c2d_norm[..., :2]

    # grid_sample
    dtype = img_features.dtype
    feats = F.grid_sample(
        img_features.float(),
        c2d_norm,
        align_corners=True,
        padding_mode="border",
        mode=interp,
    )
    feats = feats.to(dtype)

    feats = feats.squeeze(-1)  # b ch n

    feats = rearrange(feats, "b ch n -> b n ch")
    return feats, c3d_cam_hom[..., :3], c2d_cam, viewdirs_cam


class PixelAlignedRadianceField(nn.Module):
    def __init__(self, n_feats=64):
        super().__init__()
        self.pos_enc = PositionalEncoding(freq_factor=1.5)

        n_blocks = 5
        d_hidden = 512
        self.mlp = ResnetFCTimeEmbed(
            d_in=self.pos_enc.d_out + n_feats,
            n_blocks=n_blocks,
            d_hidden=d_hidden,
            combine_layer=3,
            combine_type="average",
        )

    def forward(
        self,
        xyz,
        clean_ctxt_img_feats,
        noisy_trgt_img_feats,
        ctxt_c2w,
        trgt_c2w,
        intrinsics,
        t=None,
    ):
        # xyz: shape (b, n, 3)
        # clean_ctxt_img_feats: shape (b, ch, h, w)
        # noisy_trgt_img_feats: shape (b, ch, h, w) or None
        # ctxt_c2w: shape (b, 4, 4)
        # trgt_c2w: shape (b, 4, 4)
        # intrinsics: shape (b, 3, 3)
        # return: sigma, rgb, alpha
        b = clean_ctxt_img_feats.shape[0]
        num_target = xyz.shape[0] // b
        xyz = xyz.view(b, num_target, -1, 3)
        xyz = rearrange(xyz, "b ns n ch -> b (ns n) ch")

        if noisy_trgt_img_feats is not None:
            num_context = 2
            clean_ctxt_img_feats = repeat(
                clean_ctxt_img_feats, "b ch h w -> b () ch h w"
            )
            noisy_trgt_img_feats = repeat(
                noisy_trgt_img_feats, "b ch h w -> b () ch h w"
            )
            all_img_feats = torch.cat(
                (clean_ctxt_img_feats, noisy_trgt_img_feats), dim=1
            )
            all_img_feats = rearrange(all_img_feats, "b ns ch h w -> (b ns) ch h w")

            ctxt_c2w = rearrange(ctxt_c2w, "b h w -> b () h w")
            trgt_c2w = rearrange(trgt_c2w, "b h w -> b () h w")
            all_c2w = torch.cat((ctxt_c2w, trgt_c2w), dim=1)
            all_c2w = rearrange(all_c2w, "b ns h w -> (b ns) h w")

            intrinsics = repeat(
                intrinsics, "b h w -> (b ns) h w", ns=num_context
            )  # TODO: check this
            xyz = repeat(xyz, "b n ch -> (b ns) n ch", ns=num_context)

        else:
            num_context = 1
            all_img_feats = clean_ctxt_img_feats
            all_c2w = ctxt_c2w

        pa_feats, cam_coords, _ = pixel_aligned_features_cond(
            xyz, all_c2w, intrinsics, all_img_feats
        )
        pos_enc = self.pos_enc(cam_coords)


        mlp_in = torch.cat((pos_enc, pa_feats), dim=-1)

        """ custom MLP 
        feats = self.mlp(mlp_in)
        sigma = self.sigma(feats)
        rad = self.radiance(feats)
        """
        mlp_output = self.mlp(mlp_in, ns=num_context, t=t)
        rad = torch.sigmoid(mlp_output[..., :3])
        sigma = torch.relu(mlp_output[..., 3:4])

        rad = rearrange(rad, "b (ns n) ch -> (b ns)  n ch", ns=num_context)
        sigma = rearrange(sigma, "b (ns n) ch -> (b ns) n ch", ns=num_context)

        return sigma, rad, None


class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        embed = x.unsqueeze(-2)
        embed = repeat(embed, "... j n -> ... (k j) n", k=2 * self.num_freqs)
        embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
        embed = rearrange(embed, "... j n -> ... (j n)")

        if self.include_input:
            embed = torch.cat((x, embed), dim=-1)
        return embed


class PixelAlignedRadianceFieldTimeEmbed(nn.Module):
    def __init__(self, n_feats=64, n_feats_out=64, use_viewdir=False):
        super().__init__()
        self.pos_enc = PositionalEncoding(freq_factor=1.5)

        if use_viewdir:
            d_in = self.pos_enc.d_out * 2 + n_feats
        else:
            d_in = self.pos_enc.d_out + n_feats
        n_blocks = 5
        d_hidden = 512
        self.mlp = ResnetFCTimeEmbed(
            d_in=d_in,
            d_out=n_feats_out + 4,
            n_blocks=n_blocks,
            d_hidden=d_hidden,
            combine_layer=3,
            combine_type="average",
        )
        self.mlp_fine = ResnetFCTimeEmbed(
            d_in=d_in,
            d_out=n_feats_out + 4,
            n_blocks=n_blocks,
            d_hidden=d_hidden,
            combine_layer=3,
            combine_type="average",
        )

    def forward(
        self,
        xyz,
        feats,
        c2w,
        intrinsics,
        viewdir=None,
        t=None,
        return_mlp_input=False,
        fine=False,
    ):
        # xyz shape (b, n, 3)   # world coordinates, same for all contexts
        # feats shape (b, n_ctxt, ch, h, w) for diffusion, n_ctxt = n_ctxt + 1 (the noisy target is also included)
        # c2w shape (b, n_ctxt, h, w)
        # intrinsics shape (b, h, w)   #same for all contexts
        # t shape (b, , n_ctxt, 1)   # different for each context (0 for clean, t for noisy)

        b = feats.shape[0]
        # xyzshape = xyz.shape
        num_target = xyz.shape[0] // b
        num_context = feats.shape[1]

        assert len(c2w.shape) == 4
        assert len(intrinsics.shape) == 3
        assert len(xyz.shape) == 3
        assert len(feats.shape) == 5

        xyz = xyz.view(b, -1, 3)
        feats = rearrange(feats, "b ns ch h w -> (b ns) ch h w")
        c2w = rearrange(c2w, "b ns h w -> (b ns) h w")
        if t is not None:
            t = rearrange(t, "b ns  -> (b ns)")

        intrinsics = repeat(
            intrinsics, "b h w -> (b ns) h w", ns=num_context
        )  # TODO: check this
        xyz = repeat(xyz, "b n ch -> (b ns) n ch", ns=num_context)

        if viewdir is not None:
            viewdir = viewdir.view(b, -1, 3)
            viewdir = repeat(viewdir, "b n ch -> (b ns) n ch", ns=num_context)

        pa_feats, cam_coords, _, viewdirs_cam = pixel_aligned_features_cond(
            coords_3d_world=xyz,
            viewdirs=viewdir,
            cam2world=c2w,
            intrinsics=intrinsics,
            img_features=feats,
        )

        pos_enc = self.pos_enc(cam_coords)
        if viewdir is not None:
            pos_enc_dir = self.pos_enc(viewdirs_cam)
            pos_enc = torch.cat((pos_enc, pos_enc_dir), dim=-1)

        mlp_in = torch.cat((pos_enc, pa_feats), dim=-1)
        if return_mlp_input:  # for feature inputs for diffusion
            mlp_in = self.mlp(mlp_in, ns=num_context, time_emb=t, return_mlp_input=True)
            return mlp_in

        if not fine:
            mlp_output = self.mlp(mlp_in, ns=num_context, time_emb=t)
        else:
            mlp_output = self.mlp_fine(mlp_in, ns=num_context, time_emb=t)

        rad = torch.sigmoid(mlp_output[..., :3])
        sigma = torch.relu(mlp_output[..., 3:4])
        feats = mlp_output[..., 4:]

        rad = rearrange(rad, "b (ns n) ch -> (b ns)  n ch", ns=num_target)
        sigma = rearrange(sigma, "b (ns n) ch -> (b ns) n ch", ns=num_target)
        feats = rearrange(feats, "b (ns n) ch -> (b ns) n ch", ns=num_target)
        feats = torch.cat((rad, feats), dim=-1)

        return sigma, feats, None
