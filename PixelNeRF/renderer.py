"""Volume rendering code."""
# parts are adapted from https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py

from typing import Callable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor, device
from typeguard import typechecked
from torchtyping import TensorType
from einops import rearrange
from geometry import *

def sample_points_along_rays(
    z_near: TensorType["camera_batch", torch.float32],
    z_far: TensorType["camera_batch", torch.float32],
    num_samples: int,
    ray_origins: TensorType["camera_batch", "ray_batch", 3, torch.float32],
    ray_directions: TensorType["camera_batch", "ray_batch", 3, torch.float32],
    lindisp: bool,
    device: torch.device,
) -> Tuple[
    TensorType["camera_batch", "ray_batch", "z_batch", 3, torch.float32],
    TensorType["camera_batch", "ray_batch", "z_batch", 1, torch.float32],
    TensorType["camera_batch", "ray_batch", "z_batch", 3, torch.float32],
]:
    # Define num_samples linearly spaced depth values between z_near and z_far.
    z_vals = torch.linspace(0, 1, num_samples, device=device)
    z_vals = rearrange(z_vals, "z -> () () z ()")
    z_near = rearrange(z_near, "c -> c () () ()")
    z_far = rearrange(z_far, "c -> c () () ()")

    if lindisp:
        z_vals = 1.0 / (1.0 / z_near * (1.0 - z_vals) + 1.0 / z_far * (z_vals))
    else:
        z_vals = z_vals * (z_far - z_near) + z_near

    # Generate 3D points along the rays according to the z_vals.
    ray_origins = rearrange(ray_origins, "c r xyz -> c r () xyz")
    ray_directions = rearrange(ray_directions, "c r xyz -> c r () xyz")
    points = ray_origins + ray_directions * z_vals

    dirs = torch.zeros_like(ray_origins) + ray_directions * torch.ones_like(
        z_vals
    )  # to make the shapes same as points:)
    # Return points and depth values.
    _, num_rays, _, _ = points.shape
    return points, repeat(z_vals, "c () z () -> c r z ()", r=num_rays), dirs


def pdf_z_values(
    bins: Tensor, weights: Tensor, samples: int, d: device, perturb: bool,
) -> Tensor:
    """Generate z-values from pdf

    Arguments:
        bins (Tensor): z-value bins (B, N - 2)
        weights (Tensor): bin weights gathered from first pass (B, N - 1)
        samples (int): number of samples N
        d (device): torch device
        perturb (bool): peturb ray query segment

    Returns:
        t (Tensor): z-values sampled from pdf (B, N)
    """
    EPS = 1e-5
    B, N = weights.size()

    weights = weights + EPS
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat((torch.zeros_like(cdf[:, :1]), cdf), dim=-1)

    if perturb:
        u = torch.rand((B, samples), device=d)
        u = u.contiguous()
    else:
        u = torch.linspace(0, 1, samples, device=d)
        u = u.expand(B, samples)
        u = u.contiguous()

    idxs = torch.searchsorted(cdf, u, right=True)
    idxs_below = torch.clamp_min(idxs - 1, 0)
    idxs_above = torch.clamp_max(idxs, N)
    idxs = torch.stack((idxs_below, idxs_above), dim=-1).view(B, 2 * samples)

    cdf = torch.gather(cdf, dim=1, index=idxs).view(B, samples, 2)
    bins = torch.gather(bins, dim=1, index=idxs).view(B, samples, 2)

    den = cdf[:, :, 1] - cdf[:, :, 0]
    den[den < EPS] = 1.0

    t = (u - cdf[:, :, 0]) / den
    t = bins[:, :, 0] + t * (bins[:, :, 1] - bins[:, :, 0])

    return t


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[
        denom < eps
    ] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
        bins_g[..., 1] - bins_g[..., 0]
    )
    return samples


def pdf_rays(
    ro: Tensor, rd: Tensor, t: Tensor, weights: Tensor, samples: int, perturb: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Sample pdf along rays given computed weights

    Arguments:
        ro (Tensor): rays origin (B, 3)
        rd (Tensor): rays direction (B, 3)
        t (Tensor): coarse z-value (B, N)
        weights (Tensor): weights gathered from first pass (B, N)
        samples (int): number of samples along the ray
        perturb (bool): peturb ray query segment

    Returns:
        rx (Tensor): rays position queries (B, Nc + Nf, 3)
        rd (Tensor): rays direction (B, Nc + Nf, 3)
        t (Tensor): z-values from near to far (B, Nc + Nf)
        delta (Tensor): rays segment lengths (B, Nc + Nf)
    """
    B, S, N_coarse, _ = weights.shape
    weights = rearrange(weights, "b n s 1 ->  (b n) s")
    t = rearrange(t, "b n s 1 -> (b n) s")

    Nf = samples
    tm = 0.5 * (t[:, :-1] + t[:, 1:])
    t_pdf = sample_pdf(tm, weights[..., 1:-1], Nf, det=False).detach().view(B, S, Nf)
    rx = ro[..., None, :] + rd[..., None, :] * t_pdf[..., None]
    rd = rd[..., None, :].expand(B, S, Nf, 3)
    return rx, t_pdf, rd


def volume_integral(
    z_vals: torch.tensor, sigmas: torch.tensor, radiances: torch.tensor
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    # Compute the deltas in depth between the points.
    dists = torch.cat(
        [
            z_vals[..., 1:, :] - z_vals[..., :-1, :],
            (z_vals[..., 1:, :] - z_vals[..., :-1, :])[..., -1:, :],
        ],
        -2,
    )

    # Compute the alpha values from the densities and the dists.
    # Tip: use torch.einsum for a convenient way of multiplying the correct
    # dimensions of the sigmas and the dists.
    # alpha = 1.0 - torch.exp(-torch.einsum("brzs, z -> brzs", F.relu(sigmas), dists))

    alpha = 1.0 - torch.exp(-F.relu(sigmas) * dists)

    alpha_shifted = torch.cat(
        [torch.ones_like(alpha[:, :, :1]), 1.0 - alpha + 1e-10], -2
    )

    # Compute the Ts from the alpha values. Use torch.cumprod.
    Ts = torch.cumprod(alpha_shifted, -2)

    # Compute the weights from the Ts and the alphas.
    weights = alpha * Ts[..., :-1, :]

    # Compute the pixel color as the weighted sum of the radiance values.
    rgb = torch.einsum("brzs, brzs -> brs", weights, radiances)
    # print(f'weights: {weights.shape}, radiances: {radiances.shape}, rgb: {rgb.shape}')

    # Compute the depths as the weighted sum of z_vals.
    # Tip: use torch.einsum for a convenient way of computing the weighted sum,
    # without the need to reshape the z_vals.
    depth_map = torch.einsum("brzs, brzs -> brs", weights, z_vals)

    return rgb, depth_map, weights


class VolumeRenderer(nn.Module):
    def __init__(
        self,
        near,
        far,
        n_samples=64,
        n_fine_samples=16,
        use_viewdir=False,
        backgrd_color=None,
        lindisp=False,
    ):
        super().__init__()
        self.near = near
        self.far = far
        self.n_samples = n_samples
        self.n_fine_samples = n_fine_samples
        self.lindisp = lindisp
        self.use_viewdir = use_viewdir
        if backgrd_color is not None:
            self.register_buffer("backgrd_color", backgrd_color)
        else:
            self.backgrd_color = None

    def forward(
        self,
        cam2world: TensorType["camera_batch", 4, 4, torch.float32],
        intrinsics: TensorType["camera_batch", 3, 3, torch.float32],
        x_pix: TensorType["camera_batch", "ray_batch", 2, torch.float32],
        radiance_field: Callable,
        # add_noise: bool = False,
        z_near: Optional[TensorType["camera_batch", torch.float32]] = None,
        z_far: Optional[TensorType["camera_batch", torch.float32]] = None,
        top_down: bool = False,
    ):
        """
        Takes as inputs ray origins and directions - samples points along the
        rays and then calculates the volume rendering integral.

        Returns:
            Tuple of rgb, depth_map
            rgb: for each pixel coordinate x_pix, the color of the respective ray.
            depth_map: for each pixel coordinate x_pix, the depth of the respective ray.

        """
        batch_size, num_rays = x_pix.shape[0], x_pix.shape[1]
        device = cam2world.device

        # Compute the ray directions in world coordinates.
        if not top_down:
            ros, rds = get_world_rays(x_pix, intrinsics, cam2world)
            if z_far is None:
                z_far = torch.tensor(self.far, device=device).broadcast_to(
                    (batch_size,)
                )
            if z_near is None:
                z_near = torch.tensor(self.near, device=device).broadcast_to(
                    (batch_size,)
                )
        else:
            ros, rds = get_world_rays_top_down(x_pix, intrinsics, cam2world)
            z_far = torch.tensor(2.0, device=device).broadcast_to((batch_size,))
            z_near = torch.tensor(0.0, device=device).broadcast_to((batch_size,))

        # Generate the points along rays and their depth values.
        pts, z_vals, viewdirs = sample_points_along_rays(
            z_near,
            z_far,
            self.n_samples,
            ros,
            rds,
            device=x_pix.device,
            lindisp=self.lindisp,
        )

        pts = pts.reshape(batch_size, -1, 3)
        if self.use_viewdir:
            viewdirs = viewdirs.reshape(batch_size, -1, 3)
        else:
            viewdirs = None
        sigma, feats, _ = radiance_field(pts, viewdirs, fine=False)

        sigma = sigma.view(batch_size, num_rays, self.n_samples, 1)
        feats = feats.view(batch_size, num_rays, self.n_samples, -1)

        if self.n_fine_samples > 0:
            with torch.no_grad():
                # Compute coarse pixel colors, depths, and weights via the volume integral.
                rendering, depth_map, weights = volume_integral(z_vals, sigma, feats)
                pts_fine, z_vals_fine, viewdirs_fine = pdf_rays(
                    ros, rds, z_vals, weights, self.n_fine_samples, perturb=False
                )
            pts_fine = pts_fine.reshape(batch_size, -1, 3)
            if self.use_viewdir:
                viewdirs_fine = viewdirs_fine.reshape(batch_size, -1, 3)
            else:
                viewdirs_fine = None

            # using the same network to compute the fine features
            # sigma_fine, feats_fine, _ = radiance_field(
            #     pts_fine, viewdirs_fine, fine=True
            # )
            # sigma_fine = sigma_fine.view(batch_size, num_rays, self.n_fine_samples, 1)
            # feats_fine = feats_fine.view(batch_size, num_rays, self.n_fine_samples, -1)
            # sigma_all = torch.cat([sigma, sigma_fine], dim=-2)
            # feats_all = torch.cat([feats, feats_fine], dim=-2)

            # use separate network to compute the coarse and fine features
            # print(pts.shape, pts_fine.shape, viewdirs.shape, viewdirs_fine.shape)
            pts_all = torch.cat(
                [
                    pts.view(batch_size, -1, self.n_samples, 3),
                    pts_fine.view(batch_size, -1, self.n_fine_samples, 3),
                ],
                dim=-2,
            ).view(batch_size, -1, 3)
            if self.use_viewdir:
                viewdirs_all = torch.cat(
                    [
                        viewdirs.view(batch_size, -1, self.n_samples, 3),
                        viewdirs_fine.view(batch_size, -1, self.n_fine_samples, 3),
                    ],
                    dim=-2,
                ).view(batch_size, -1, 3)
            else:
                viewdirs_all = None
            sigma_all, feats_all, _ = radiance_field(pts_all, viewdirs_all, fine=True)
            sigma_all = sigma_all.view(
                batch_size, num_rays, self.n_samples + self.n_fine_samples, 1
            )
            feats_all = feats_all.view(
                batch_size, num_rays, self.n_samples + self.n_fine_samples, -1
            )

            z_vals_fine = rearrange(z_vals_fine, "b n s -> b n s 1")
            z_vals_all = torch.cat([z_vals, z_vals_fine], dim=-2)

            # sort the coarse and fine samples
            _, indices = torch.sort(z_vals_all, dim=-2)
            z_vals_all = torch.gather(z_vals_all, dim=-2, index=indices)
            sigma_all = torch.gather(sigma_all, dim=-2, index=indices)
            feats_all = torch.gather(
                feats_all, dim=-2, index=indices.expand(-1, -1, -1, feats_all.shape[-1])
            )
        else:
            # only use the coarse samples
            sigma_all = sigma
            feats_all = feats
            z_vals_all = z_vals

        rendering, depth_map, weights = volume_integral(
            z_vals_all, sigma_all, feats_all
        )

        if self.backgrd_color is not None:
            accum = weights.sum(dim=-2)
            backgrd_color = self.backgrd_color.broadcast_to(rendering[..., :3].shape)
            rendering[..., :3] = rendering[..., :3] + (1.0 - accum) * backgrd_color
        misc = {"z_vals": z_vals_all, "weights": weights}
        return rendering, depth_map, misc


class NoVolumeRenderer(nn.Module):
    def __init__(
        self,
        near,
        far,
        n_samples=64,
        n_fine_samples=0,
        n_feats=512,
        backgrd_color=None,
        lindisp=False,
    ):
        super().__init__()
        self.near = near
        self.far = far
        self.n_samples = n_samples
        self.n_fine_samples = n_fine_samples
        assert n_fine_samples == 0
        self.lindisp = lindisp
        if backgrd_color is not None:
            self.register_buffer("backgrd_color", backgrd_color)
        else:
            self.backgrd_color = None
        self.n_feats = n_feats
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * n_samples, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

    @typechecked
    def forward(
        self,
        cam2world: TensorType["camera_batch", 4, 4, torch.float32],
        intrinsics: TensorType["camera_batch", 3, 3, torch.float32],
        x_pix: TensorType["camera_batch", "ray_batch", 2, torch.float32],
        radiance_field: Callable,
        # add_noise: bool = False,
        z_near: Optional[TensorType["camera_batch", torch.float32]] = None,
        z_far: Optional[TensorType["camera_batch", torch.float32]] = None,
    ):
        """
        Takes as inputs ray origins and directions - samples points along the
        rays and then calculates the volume rendering integral.

        Returns:
            Tuple of rgb, depth_map
            rgb: for each pixel coordinate x_pix, the color of the respective ray.
            depth_map: for each pixel coordinate x_pix, the depth of the respective ray.

        """
        batch_size, num_rays = x_pix.shape[0], x_pix.shape[1]
        device = cam2world.device

        # Compute the ray directions in world coordinates.
        ros, rds = get_world_rays(x_pix, intrinsics, cam2world)

        # Generate the points along rays and their depth values.
        if z_far is None:
            z_far = torch.tensor(self.far, device=device).broadcast_to((batch_size,))
        if z_near is None:
            z_near = torch.tensor(self.near, device=device).broadcast_to((batch_size,))
        pts, z_vals = sample_points_along_rays(
            z_near,
            z_far,
            self.n_samples,
            ros,
            rds,
            device=x_pix.device,
            lindisp=self.lindisp,
        )

        pts = pts.reshape(batch_size, -1, 3)
        feats = radiance_field(pts)
        feats = feats.view(batch_size, num_rays, self.n_samples * self.n_feats)
        feats = self.mlp(feats)
        return feats
