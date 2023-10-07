import torch
from torch_efficient_distloss import (
    eff_distloss,
    eff_distloss_native,
    flatten_eff_distloss,
)
from einops import rearrange, repeat


def mse_loss(img, gt):
    img = img.view(gt.shape)
    return ((img - gt) ** 2).mean()


def distortion_loss(weights, z_vals, near, far):
    # loss from mip-nerf 360; efficient implementation from DVGOv2 (https://github.com/sunset1995/torch_efficient_distloss) with some modifications

    # weights: [B, N, n_samples, 1]
    # z_vals: [B, N, n_samples, 1]

    assert weights.shape == z_vals.shape
    assert len(weights.shape) == 4
    weights = rearrange(weights, "b n s 1 -> (b n) s")
    z_vals = rearrange(z_vals, "b n s 1 -> (b n) s")

    # go from z space to s space (for linear sampling; INVERSE SAMPLING NOT IMPLEMENTED)
    s = (z_vals - near) / (far - near)

    # distance between samples
    interval = s[:, 1:] - s[:, :-1]

    loss = eff_distloss(weights[:, :-1], s[:, :-1], interval)
    return loss


def occupancy_loss(weights):
    # loss from lolnerf (prior on weights to be distributed as a mixture of Laplacian distributions around mode 0 or 1)
    # weights: [B, N, n_samples, 1]
    assert len(weights.shape) == 4

    pw = torch.exp(-torch.abs(weights)) + torch.exp(
        -torch.abs(torch.ones_like(weights) - weights)
    )
    loss = -1.0 * torch.log(pw).mean()
    return loss
