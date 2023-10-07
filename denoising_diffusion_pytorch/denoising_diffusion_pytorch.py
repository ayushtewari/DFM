# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch

import math
from pathlib import Path
from random import random
from collections import OrderedDict
import numpy as np
from functools import partial
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import Adam, AdamW
from torchvision import utils

from einops import rearrange, reduce, repeat

from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
from torchvision.utils import make_grid

from denoising_diffusion_pytorch.version import __version__
import wandb
import sys
import os
import imageio
from accelerate import DistributedDataParallelKwargs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from utils import *
from layers import *
from losses import *
import lpips

from scipy import interpolate

# constants
ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "pred_x_start_high_res"]
)


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type="l1",
        objective="pred_noise",
        beta_schedule="sigmoid",
        schedule_fn_kwargs=dict(),
        p2_loss_weight_gamma=0.0,  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1,
        ddim_sampling_eta=0.0,
        auto_normalize=True,
        use_guidance=False,
        guidance_scale=1.0,
        temperature=1.0,
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        # assert not model.enc.model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.temperature = temperature

        self.image_size = image_size
        self.use_guidance = use_guidance
        self.guidance_scale = guidance_scale
        self.objective = objective

        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate p2 reweighting
        use_constant_p2_weight = True
        if use_constant_p2_weight:
            register_buffer(
                "p2_loss_weight",
                (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
                ** -p2_loss_weight_gamma,
            )
        else:
            snr = alphas_cumprod / (1 - alphas_cumprod)
            register_buffer(
                "p2_loss_weight", torch.minimum(snr, torch.ones_like(snr) * 5.0)
            )  # https://arxiv.org/pdf/2303.09556.pdf

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
        self.perceptual_loss = lpips.LPIPS(net="vgg")

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        inp,
        t,
        x_self_cond=None,
        clip_x_start=False,
        render_coarse=False,
        guidance_scale=1.0,
        render_high_res=False,
    ):
        x = inp["noisy_trgt_rgb"]
        model_output, depth, uncond_model_output, _ = self.model.render_full_image(
            inp["clean_ctxt_feats"],
            inp["trgt_rgbd"],
            inp["x_pix"],
            inp["intrinsics"],
            inp["ctxt_c2w"],
            inp["trgt_c2w"][:, :1, ...],
            inp["noisy_trgt_rgb"],
            t,
            x_self_cond,
            render_coarse=render_coarse,
            guidance_scale=guidance_scale,
            uncond_trgt_rgbd=inp["uncond_trgt_rgbd"],
            uncond_clean_ctxt_feats=inp["uncond_clean_ctxt_feats"],
            render_high_res=render_high_res,
            xy_pix_high_res=None,  # inp["x_pix_128"],
            trgt_abs_camera_poses=inp["trgt_abs_camera_poses"],
        )
        # print(f"model_outpus range: {model_output.min()} {model_output.max()}")
        dynamic_threshold = True
        dynamic_thresholding_percentile = 0.95
        if dynamic_threshold:
            # following pseudocode in appendix
            # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
            # print("using dynamic thresholding")

            def maybe_clip(x_start):
                s = torch.quantile(
                    rearrange(x_start, "b ... -> b (...)").abs(),
                    dynamic_thresholding_percentile,
                    dim=-1,
                )

                s.clamp_(min=1.0)
                s = right_pad_dims_to(x_start, s)
                x_start = x_start.clamp(-s, s) / s
                return x_start

        else:
            maybe_clip = (
                partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
            )

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == "pred_x0":
            x_start = model_output

            if self.use_guidance and guidance_scale > 1.0:
                uncond_x_start = uncond_model_output
                x_start = uncond_x_start + guidance_scale * (x_start - uncond_x_start)

            x_start = maybe_clip(x_start)
            num_targets = x_start.shape[0] // x.shape[0]
            x_start = rearrange(x_start, "(b nt) c h w -> b nt c h w", nt=num_targets)[
                :, 0, ...
            ]
            x_start_high_res = None
            if render_high_res:
                x_start_high_res = x_start
                x_start = F.interpolate(
                    x_start, size=(64, 64), mode="bilinear", antialias=True,
                )
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start, x_start_high_res)

    def p_mean_variance(self, inp, t, x_self_cond=None, clip_denoised=True):
        x = inp["noisy_trgt_rgb"]
        preds = self.model_predictions(inp, t, x_self_cond)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, inp, t: int, x_self_cond=None):
        x = inp["noisy_trgt_rgb"]
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            inp=inp, t=batched_times, x_self_cond=x_self_cond, clip_denoised=True
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps=False, inp=None):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        print("p sample loop")

        x_start = None
        with torch.no_grad():
            ctxt_rgbd, trgt_rgbd, ctxt_feats = self.model.render_ctxt_from_trgt_cam(
                ctxt_rgb=inp["ctxt_rgb"],
                intrinsics=inp["intrinsics"],
                xy_pix=inp["x_pix"],
                ctxt_c2w=inp["ctxt_c2w"],
                trgt_c2w=inp["trgt_c2w"],
            )
            inp["trgt_rgbd"] = trgt_rgbd
            clean_ctxt_feats = ctxt_feats
            inp["clean_ctxt_feats"] = clean_ctxt_feats

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            self_cond = x_start if self.self_condition else None
            inp["noisy_trgt_rgb"] = img

            img, x_start = self.p_sample(inp, t, self_cond)
            imgs.append(img)

        inp["noisy_trgt_rgb"] = img

        # ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        # ret = self.unnormalize(ret)
        # return ret
        time_embed = torch.full((1,), t, device=device, dtype=torch.long)
        frames, depth_frames, *_ = self.model.render_video(
            inp, time_embed, 20, x_self_cond=False,
        )
        td_frames, td_depth_frames = None, None
        # if "render_top_down" in inp.keys():
        #     td_frames, td_depth_frames, *_ = self.model.render_video(
        #         inp, time_embed, 20, x_self_cond=False, top_down=True,
        #     )

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.unnormalize(ret)
        print("ret shape", ret.shape)
        rgb = None if trgt_rgbd is None else self.unnormalize(trgt_rgbd[:, :3, :, :])
        depth = None if trgt_rgbd is None else trgt_rgbd[:, 3:, :, :]
        out_dict = {
            "images": ret,
            "videos": frames,
            "rgb": rgb,
            "depth": depth,
            "depth_videos": depth_frames,
            "td_videos": td_frames,
            "td_depth_videos": td_depth_frames,
        }
        return out_dict

    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps=False, inp=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )
        """Normalize input images"""
        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        temperature = self.temperature  # 0.85
        img = torch.randn(shape, device=device) * temperature
        imgs = [img]
        x_start = None
        # """
        if "num_frames_render" in inp.keys():
            num_frames_render = inp["num_frames_render"]
        else:
            num_frames_render = 20
        with torch.no_grad():
            # print(f"ctxt_rgb shape {inp['ctxt_rgb'].shape}, img shape {img.shape}")
            ctxt_rgbd, trgt_rgbd, ctxt_feats = self.model.render_ctxt_from_trgt_cam(
                ctxt_rgb=inp["ctxt_rgb"],
                intrinsics=inp["intrinsics"],
                xy_pix=inp["x_pix"],
                ctxt_c2w=inp["ctxt_c2w"],
                trgt_c2w=inp["trgt_c2w"],
                render_cond=True,
                ctxt_abs_camera_poses=inp["ctxt_abs_camera_poses"]
            )
            inp["trgt_rgbd"] = trgt_rgbd
            clean_ctxt_feats = ctxt_feats
            inp["clean_ctxt_feats"] = clean_ctxt_feats

            if self.use_guidance:
                (
                    uncond_ctxt_rgbd,
                    uncond_trgt_rgbd,
                    uncond_ctxt_feats,
                ) = self.model.render_ctxt_from_trgt_cam(
                    ctxt_rgb=inp["ctxt_rgb"] * 0.0,
                    intrinsics=inp["intrinsics"],
                    xy_pix=inp["x_pix"],
                    ctxt_c2w=inp["ctxt_c2w"],
                    trgt_c2w=inp["trgt_c2w"],
                    render_cond=True,
                    ctxt_abs_camera_poses=inp["ctxt_abs_camera_poses"]
                )
                inp["uncond_trgt_rgbd"] = uncond_trgt_rgbd
                inp["uncond_clean_ctxt_feats"] = uncond_ctxt_feats
            else:
                inp["uncond_trgt_rgbd"] = None
                inp["uncond_clean_ctxt_feats"] = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            inp["noisy_trgt_rgb"] = img

            if time_next < 0:
                render_high_res = True
            else:
                render_high_res = False
            render_high_res = False
            pred_noise, x_start, x_start_high_res = self.model_predictions(
                inp,
                time_cond,
                self_cond,
                clip_x_start=True,
                guidance_scale=self.guidance_scale,
                render_high_res=render_high_res,
            )
            if time_next < 0:
                img = x_start
                imgs.append(img)

                # render the video
                frames, depth_frames, render_poses = self.model.render_video(
                    inp,
                    time_cond,
                    n=num_frames_render,
                    x_self_cond=False,
                    render_high_res=False,
                )
                td_frames, td_depth_frames = None, None
                # if "render_top_down" in inp.keys():
                #     td_frames, td_depth_frames, *_ = self.model.render_video(
                #         inp, time_cond, 20, x_self_cond=False, top_down=True,
                #     )
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img) * temperature
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        # ret = x_start_high_res if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.unnormalize(ret)
        print("ret shape", ret.shape)
        rgb = None if trgt_rgbd is None else self.unnormalize(trgt_rgbd[:, :3, :, :])
        depth = None if trgt_rgbd is None else trgt_rgbd[:, 3:, :, :]
        out_dict = {
            "images": ret,
            "videos": frames,
            "rgb": rgb,
            "depth": depth,
            "conditioning_depth": inp["trgt_rgbd"][:, 3, :, :],
            "depth_videos": depth_frames,
            "td_videos": td_frames,
            "td_depth_videos": td_depth_frames,
            "inp": inp,
            "render_poses": render_poses,
            "time_cond": time_cond,
        }
        return out_dict

    @torch.no_grad()
    def sample(self, batch_size=2, return_all_timesteps=False, inp=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        )
        return sample_fn(
            (batch_size, channels, image_size, image_size),
            return_all_timesteps=return_all_timesteps,
            inp=inp,
        )

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(
            reversed(range(0, t)), desc="interpolation sample time step", total=t
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long)
            )

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def compute_tv_norm(self, values):  # pylint: disable=g-doc-args
        # adapted from RegNeRF
        """Returns TV norm for input values.
        Note: The weighting / masking term was necessary to avoid degenerate
        solutions on GPU; only observed on individual DTU scenes.
        """
        v00 = values[:, :-1, :-1]
        v01 = values[:, :-1, 1:]
        v10 = values[:, 1:, :-1]

        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
        return loss

    def compute_errors_ssim(self, img0, img1, mask=None):
        b, c, h, w = img0.shape
        assert img0.shape == img1.shape
        assert c == 3

        errors = torch.mean(
            ssim(
                img0, img1, pad_reflection=False, gaussian_average=True, comp_mode=True
            ),
            dim=1,
        )
        return errors

    def edge_aware_smoothness(self, gt_img, depth, mask=None):
        bd, hd, wd = depth.shape
        depth = depth.reshape(-1, 1, hd, wd)

        b, c, h, w = gt_img.shape
        assert bd == b and hd == h and wd == w

        depth = 1 / depth.reshape(-1, 1, h, w).clamp(1e-3, 80)
        depth = depth / torch.mean(depth, dim=[2, 3], keepdim=True)

        d_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        d_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

        i_dx = torch.mean(
            torch.abs(gt_img[:, :, :, :-1] - gt_img[:, :, :, 1:]), 1, keepdim=True
        )
        i_dy = torch.mean(
            torch.abs(gt_img[:, :, :-1, :] - gt_img[:, :, 1:, :]), 1, keepdim=True
        )

        d_dx *= torch.exp(-i_dx)
        d_dy *= torch.exp(-i_dy)

        errors = F.pad(d_dx, pad=(0, 1), mode="constant", value=0) + F.pad(
            d_dy, pad=(0, 0, 0, 1), mode="constant", value=0
        )
        # print(f"errors shape: {errors.shape}")
        errors = errors.view(b, h, w)
        return errors

    def p_losses(self, inp, t, noise=None, render_video=False):
        num_target = inp["trgt_rgb"].shape[1]
        x_start = inp["trgt_rgb"][:, 0, ...]
        noise = default(noise, lambda: torch.randn_like(x_start))

        # print("ctxt c2w: ", inp["ctxt_c2w"])
        # print("trgt c2w: ", inp["trgt_c2w"])

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                ctxt_rgbd, trgt_rgbd, ctxt_feats = self.model.render_ctxt_from_trgt_cam(
                    ctxt_rgb=inp["ctxt_rgb"],
                    intrinsics=inp["intrinsics"],
                    xy_pix=inp["x_pix"],
                    ctxt_c2w=inp["ctxt_c2w"],
                    trgt_c2w=inp["trgt_c2w"],
                )
                inp["trgt_rgbd"] = trgt_rgbd
                clean_ctxt_feats = ctxt_feats
                inp["clean_ctxt_feats"] = clean_ctxt_feats

                inp["noisy_trgt_rgb"] = x
                pred_noise, x_self_cond, *_ = self.model_predictions(
                    inp, t, x_self_cond=None, clip_x_start=True, render_coarse=True
                )
                # x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach()

        if self.use_guidance:
            # render_cond = 1 with probability 0.9
            uncond = random() > 0.9
            if uncond:
                # print("unconditional")
                inp["ctxt_rgb"] = inp["ctxt_rgb"] * 0.0
        else:
            uncond = False 
        render_cond = True

        # predict and take gradient step
        inp["noisy_trgt_rgb"] = x

        model_out, depth, misc = self.model(
            inp, t, x_self_cond, render_cond=render_cond
        )

        weights, z_val = misc["weights"], misc["z_vals"]
        (
            rendered_ctxt_img,
            rendered_trgt_img,
            rendered_ctxt_depth,
            rendered_trgt_depth,
            rendered_trgt_feats,
        ) = (
            misc["rendered_ctxt_rgb"],
            misc["rendered_trgt_rgb"],
            misc["rendered_ctxt_depth"],
            misc["rendered_trgt_depth"],
            misc["rendered_trgt_feats"],
        )

        frames = None
        depth_frames = None
        full_images = None
        full_depths = None
        if render_video:
            with torch.no_grad():
                ctxt_rgbd, trgt_rgbd, ctxt_feats = self.model.render_ctxt_from_trgt_cam(
                    ctxt_rgb=inp["ctxt_rgb"],
                    intrinsics=inp["intrinsics"],
                    xy_pix=inp["x_pix"],
                    ctxt_c2w=inp["ctxt_c2w"],
                    trgt_c2w=inp["trgt_c2w"],
                    render_cond=render_cond,
                    ctxt_abs_camera_poses=inp["ctxt_abs_camera_poses"]
                )
                # ctxt_inp = torch.cat([inp["ctxt_rgb"], ctxt_rgbd], dim=1)
                # clean_ctxt_feats = self.model.noisy_trgt_enc(ctxt_inp, time_emb=t * 0)
                clean_ctxt_feats = ctxt_feats
                full_images, full_depths, *_ = self.model.render_full_image(
                    clean_ctxt_feats,
                    trgt_rgbd,
                    inp["x_pix"],
                    inp["intrinsics"],
                    inp["ctxt_c2w"],
                    inp["trgt_c2w"],
                    inp["noisy_trgt_rgb"],
                    t,
                    x_self_cond,
                    trgt_abs_camera_poses=inp["trgt_abs_camera_poses"]
                )

            full_images = self.unnormalize(full_images)
            frames, depth_frames, *_ = self.model.render_video(
                inp, t=t, n=20, x_self_cond=x_self_cond, num_videos=1
            )
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = inp["trgt_rgb_sampled"]
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        target = target.view(model_out.shape)
        t = repeat(t, "b -> (b c)", c=num_target)

        loss = self.loss_fn(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        lpips_loss = torch.zeros(1, device=model_out.device)
        depth_smooth_loss = torch.zeros(1, device=model_out.device)

        if self.model.sampling == "patch":
            len_render = self.model.len_render
            gt = rearrange(target, "b c (h w) -> b c h w", h=len_render, w=len_render)
            lpips_loss = self.perceptual_loss(model_out.view_as(gt), gt)
            # lpips_loss = self.compute_errors_ssim(model_out.view_as(gt), gt)
            depth = rearrange(
                depth, "b (h w) c -> b h (w c)", h=len_render, w=len_render,
            )
            # print(f"depth shape: {depth.shape}")
            # depth_smooth_loss = self.compute_tv_norm(depth)
            depth_smooth_loss = self.edge_aware_smoothness(gt, depth)

        loss_cond = self.loss_fn(
            rendered_trgt_img, inp["trgt_rgb"][:, 0, ...], reduction="none"
        )
        # loss_cond = loss * 0.0

        rgb_intermediate = None
        loss_intermediate = None
        if misc["rgb_intermediate"] is not None:
            loss_intermediate = self.loss_fn(
                misc["rgb_intermediate"], target, reduction="none"
            )
            loss_intermediate = reduce(loss_intermediate, "b ... -> b (...)", "mean")
            loss_intermediate = loss_intermediate * extract(
                self.p2_loss_weight, t, loss.shape
            )
            rgb_intermediate = self.unnormalize(misc["rgb_intermediate"])

        loss_cond = reduce(loss_cond, "b ... -> b (...)", "mean")
        loss1 = distortion_loss(
            weights, z_val, near=self.model.near, far=self.model.far
        )
        if uncond:
            loss_cond *= 0.0
        dist_loss = loss1

        # print(f"depth smooth loss: {depth_smooth_loss.shape}")
        losses = {
            "rgb_loss": loss.mean(),
            "dist_loss": dist_loss,
            "rgb_cond_loss": loss_cond.mean(),
            "depth_smooth_loss": depth_smooth_loss.mean(),
            "lpips_loss": lpips_loss.mean(),
            "rgb_intermediate_loss": loss_intermediate.mean()
            if loss_intermediate is not None
            else None,
        }

        rendered_ctxt_img = (
            None if rendered_ctxt_img is None else self.unnormalize(rendered_ctxt_img)
        )
        rendered_trgt_img = (
            None
            if misc["rendered_trgt_rgb"] is None
            else self.unnormalize(misc["rendered_trgt_rgb"])
        )
        return (
            losses,
            (
                self.unnormalize(x),
                # self.unnormalize(model_out),
                self.unnormalize(inp["trgt_rgb"]),
                self.unnormalize(inp["ctxt_rgb"]),
                t,
                full_depths,
                full_images,
                frames,
                rendered_ctxt_img,
                rendered_ctxt_depth,
                rendered_trgt_img,
                rendered_trgt_depth,
                rendered_trgt_feats,
                depth_frames,
                None if "trgt_masks" not in inp.keys() else inp["trgt_masks"],
                rearrange(
                    target,
                    "b c (h w) -> b c h w",
                    h=self.model.len_render,
                    w=self.model.len_render,
                ),
                rearrange(
                    model_out,
                    "b c (h w) -> b c h w",
                    h=self.model.len_render,
                    w=self.model.len_render,
                ),
                (
                    None
                    if rgb_intermediate is None
                    else rearrange(
                        rgb_intermediate,
                        "b c (h w) -> b c h w",
                        h=self.model.len_render,
                        w=self.model.len_render,
                    )
                ),
                x_self_cond,
                uncond,
            ),
        )

    def forward(self, inp, *args, **kwargs):
        img = inp["trgt_rgb"][:, 0, ...]
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(inp, t, *args, **kwargs)


# trainer class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        accelerator,
        dataloader=None,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_horizontal_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        sample_every=1000,
        wandb_every=100,
        save_every=1000,
        num_samples=25,
        # results_folder="./results",
        amp=False,
        fp16=False,
        split_batches=True,
        warmup_period=0,
        checkpoint_path=None,
        wandb_config=None,
        run_name="diffusion",
        dist_loss_weight=0.0,
        depth_smooth_loss_weight=0.0,
        lpips_loss_weight=0.0,
        cfg=None,
        num_context=1,
        load_pn=False
    ):
        super().__init__()

        self.accelerator = accelerator
        if self.accelerator is None:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            self.accelerator = Accelerator(
                split_batches=True, mixed_precision="no", kwargs_handlers=[ddp_kwargs],
            )

        # self.accelerator.native_amp = amp
        self.num_context = num_context
        self.model = diffusion_model

        # assert has_int_squareroot(
        #     num_samples
        # ), "number of samples must have an integer square root"
        self.num_samples = num_samples
        self.sample_every = sample_every
        self.save_every = save_every
        self.wandb_every = wandb_every
        self.dist_loss_weight = dist_loss_weight
        self.depth_smooth_loss_weight = depth_smooth_loss_weight
        self.lpips_loss_weight = lpips_loss_weight
        assert self.sample_every % self.wandb_every == 0

        self.batch_size = train_batch_size
        print(f"batch size: {self.batch_size}")
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = self.model.image_size

        # dataset and dataloader

        self.dataloader = self.accelerator.prepare(dataloader)
        self.dl = cycle(self.dataloader)

        # optimizer

        params = [
            p
            for n, p in diffusion_model.named_parameters()
            if "perceptual" not in n  # and "cnn_refine_model" not in n
        ]
        # cnn_refine_params = [
        #     p for n, p in diffusion_model.named_parameters() if "cnn_refine_model" in n
        # ]
        self.opt = Adam(params, lr=train_lr, betas=adam_betas)
        # self.opt = AdamW(params, lr=train_lr, weight_decay=1e-3, amsgrad=True,)
        lr_scheduler = get_cosine_schedule_with_warmup(
            self.opt, warmup_period, train_num_steps
        )
        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model, beta=ema_decay, update_every=ema_update_every
            )

        self.step = 0

        if checkpoint_path is None and cfg.resume_id is not None:
            folders = [
                os.path.join("wandb/", f, "files")
                for f in sorted(os.listdir("wandb"))
                if cfg.resume_id in f
            ]
            # find all *.pt files in all folders
            ckpt_files = []
            for folder in folders:
                ckpt_files.extend(
                    [
                        os.path.join(folder, f)
                        for f in os.listdir(folder)
                        if f.endswith(".pt")
                    ]
                )
            # find the latest checkpoint
            checkpoint_path = sorted(ckpt_files)[-1]

        self.load_pn =load_pn

        if checkpoint_path is not None and self.accelerator.is_main_process:
            print(f"checkpoint path: {checkpoint_path}")
            
            if self.load_pn:
                self.load_from_pixelnerf(checkpoint_path)
            else:
                self.load(checkpoint_path)

            # self.load_highres_from_lowres(checkpoint_path)
            # self.load_sep_from_joint(checkpoint_path)
            # self.load_feats_from_nofeats(checkpoint_path)
            # self.load_viewdep_from_noviewdep(checkpoint_path)
            # self.load_fine_from_coarse(checkpoint_path)
            # self.load_from_pixelnerf(checkpoint_path)
            # self.load_from_external_checkpoint(checkpoint_path)
        self.model, self.opt, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.opt, lr_scheduler
        )

        if self.accelerator.is_main_process:
            # run = wandb.init(config=cfg, **wandb_config)
            if cfg.wandb_id is not None:
                run = wandb.init(
                    config=cfg, **wandb_config, id=cfg.wandb_id, resume="allow"
                )
            elif cfg.resume_id is not None:
                run = wandb.init(
                    config=cfg, **wandb_config, id=cfg.resume_id, resume="must"
                )
            else:
                run = wandb.init(config=cfg, **wandb_config)
            # print(f"starting run {cfg.wandb_id}")
            wandb.run.log_code(".")
            wandb.run.name = run_name

            print(f"run dir: {run.dir}", flush=True)

            run_dir = run.dir
            wandb.save(os.path.join(run_dir, "checkpoint*"))
            wandb.save(os.path.join(run_dir, "video*"))
            self.results_folder = Path(run_dir)
            self.results_folder.mkdir(exist_ok=True)

        # prepare model, dataloader, optimizer with accelerator

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
            "version": __version__,
        }
        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))
        wandb.save(
            str(self.results_folder / f"model-{milestone}.pt"),
            base_path=self.results_folder,
        )
        # delete prev checkpoint if exists
        prev_milestone = milestone - 1
        prev_path = self.results_folder / f"model-{prev_milestone}.pt"
        if os.path.exists(prev_path):
            # delete prev checkpoint
            os.remove(prev_path)

    def load(self, path):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(path), map_location=torch.device("cpu"),)

        # model = self.accelerator.unwrap_model(self.model)
        model = self.model
        # print(f"model parameter names: {list(model.state_dict().keys())}")
        # load all parameteres
        model.load_state_dict(data["model"], strict=True)

        try:
            self.step = data["step"]
            self.opt.load_state_dict(data["opt"])
        except:
            print("step optimizer not found")

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"], strict=True)

        if "version" in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

        del data

    def load_from_pixelnerf(self, path):
        accelerator = self.accelerator
        device = accelerator.device

        model = self.accelerator.unwrap_model(self.model)
        data = torch.load(path, map_location=device)

        new_state_dict = OrderedDict()
        for key, value in data["model"].items():
            if "pixel" in key:
                new_state_dict[key.replace("pixelNeRF", "pixelNeRF_joint")] = value
                new_state_dict[key.replace("pixelNeRF", "pixelNeRF_joint_coarse")] = value
                new_state_dict[key] = value
            elif "enc" in key:
                if "enc.model.conv1" not in key:
                    new_state_dict[key.replace("enc", "noisy_trgt_enc")] = value
                new_state_dict[key] = value
            elif key in model.state_dict().keys():
                new_state_dict[key] = value
            else:
                print(f"key {key} not in model state dict")
        # exit()
        model.load_state_dict(new_state_dict, strict=True)

    def load_from_external_checkpoint(self, path):
        accelerator = self.accelerator
        device = accelerator.device

        model = self.accelerator.unwrap_model(self.model)
        data = torch.load(path, map_location=device)

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for key, value in data["model_state_dict"].items():
            if "enc" not in key:
                key = "model." + key[7:]  # remove `att.`
                new_state_dict[key] = value
                if key not in model.state_dict().keys():
                    print(f"enc key {key} not in model state dict")
            else:
                new_key = "model." + key[7:]  # remove `att.`
                key1 = new_key.replace(".enc.", ".clean_ctxt_enc.")
                new_key.replace(".enc.", ".noisy_trgt_enc.")
                new_state_dict[key1] = value

                if key1 not in model.state_dict().keys():
                    print(f"key {key1} not in model state dict")
        model.load_state_dict(new_state_dict, strict=False)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        # torch.cuda.set_device(device)
        print(f"device: {device}")
        torch.cuda.empty_cache()

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.0
                total_rgb_loss = 0.0
                total_rgb_cond_loss = 0.0
                total_dist_loss = 0.0
                total_depth_smooth_loss = 0.0
                total_lpips_loss = 0.0
                total_rgb_intermediate_loss = 0.0
                render_video = (
                    self.step % self.wandb_every == 0
                )  # and accelerator.is_main_process

                for _ in range(self.gradient_accumulate_every):
                    # if accelerator.is_main_process:
                    # self.dataloader.dataset.num_context = np.random.randint(
                    #     1, self.dataloader.dataset.max_num_context + 1
                    # )
                    # print(
                    #     f"num_context main: {self.dataloader.dataset.num_context}"
                    # )
                    num_context = np.random.randint(1, self.num_context + 1,)
                    data = next(self.dl)  # .to(device)
                    if isinstance(data, list):
                        # gt = data[1]
                        data = data[0]
                        for k, v in data.items():
                            if k in ["ctxt_rgb", "ctxt_c2w", "ctxt_abs_camera_poses"]:
                                data[k] = v[:, :num_context]
                        data = to_gpu(data, device)
                        # print(f"datashape: {data['ctxt_rgb'].shape}")
                    with self.accelerator.autocast():
                        losses, misc = self.model(data, render_video=render_video)
                        # print("losses computed")
                        rgb_loss = losses["rgb_loss"]
                        rgb_loss = rgb_loss / self.gradient_accumulate_every
                        total_rgb_loss += rgb_loss.item()

                        rgb_loss_cond = losses["rgb_cond_loss"]
                        rgb_loss_cond = rgb_loss_cond / self.gradient_accumulate_every
                        total_rgb_cond_loss += rgb_loss_cond.item()

                        dist_loss = losses["dist_loss"]
                        dist_loss = dist_loss / self.gradient_accumulate_every
                        total_dist_loss += dist_loss.item()

                        depth_smooth_loss = losses["depth_smooth_loss"]
                        depth_smooth_loss = (
                            depth_smooth_loss / self.gradient_accumulate_every
                        )
                        total_depth_smooth_loss += depth_smooth_loss.item()

                        rgb_intermediate_loss = losses["rgb_intermediate_loss"]
                        if rgb_intermediate_loss is not None:
                            rgb_intermediate_loss = (
                                rgb_intermediate_loss / self.gradient_accumulate_every
                            )
                            total_rgb_intermediate_loss += rgb_intermediate_loss.item()

                        lpips_loss = losses["lpips_loss"]
                        lpips_loss = lpips_loss / self.gradient_accumulate_every
                        total_lpips_loss += lpips_loss.item()

                        loss = (
                            rgb_loss
                            + self.dist_loss_weight * dist_loss
                            + rgb_loss_cond
                            + self.depth_smooth_loss_weight * depth_smooth_loss
                            + self.lpips_loss_weight * lpips_loss
                        )
                        if rgb_intermediate_loss is not None:
                            loss += rgb_intermediate_loss
                        total_loss += loss.item()

                    self.accelerator.backward(loss)  # TODO check if this is correct
                    # print("loss backwarded")
                if accelerator.is_main_process:
                    wandb.log(
                        {
                            "loss": total_loss,
                            "rgb_loss": total_rgb_loss,
                            "rgb_cond_loss": total_rgb_cond_loss,
                            "dist_loss": total_dist_loss,
                            "depth_smooth_loss": total_depth_smooth_loss,
                            "lpips_loss": total_lpips_loss,
                            "rgb_intermediate_loss": total_rgb_intermediate_loss,
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "num_context": data["ctxt_rgb"].shape[1],
                            "uncond": float(misc[-1]),
                        },
                        step=self.step,
                    )
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()
                # print(f"device: {device}")
                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                all_images = None
                all_videos_list = None
                all_rgb = None
                if accelerator.is_main_process:  # or not accelerator.is_main_process:
                    # self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            print(f"batch sizes: {batches}")
                            # batches = num_to_groups(self.num_samples, 2)
                            def sample(n):
                                data.update((k, v[:n]) for k, v in data.items())
                                return self.ema.ema_model.sample(batch_size=n, inp=data)
                                # return self.model.module.sample(batch_size=n, inp=data)

                            output_dict_list = list(map(lambda n: sample(n), batches,))
                            all_images_list = [o["images"] for o in output_dict_list]
                            all_videos_list = [o["videos"] for o in output_dict_list]
                            all_rgb = (
                                None
                                if output_dict_list[0]["rgb"] is None
                                else [o["rgb"] for o in output_dict_list]
                            )
                            # accelerator.wait_for_everyone()
                        all_images = torch.cat(all_images_list, dim=0)
                        all_rgb = None if all_rgb is None else torch.cat(all_rgb, dim=0)
                        if accelerator.is_main_process:
                            utils.save_image(
                                all_images,
                                str(self.results_folder / f"sample-{milestone}.png"),
                                nrow=int(math.sqrt(self.num_samples)),
                            )

                    if (
                        self.step % self.wandb_every == 0
                        and accelerator.is_main_process
                    ):
                        self.wandb_summary(all_images, all_videos_list, all_rgb, misc)

                        if self.step != 0 and self.step % self.save_every == 0:
                            milestone = self.step // self.save_every
                            self.save(milestone)
                            print(f"saved model at {milestone} milestones")

                self.step += 1

                # if self.warmup_step is not None:
                #     self.warmup_step()
                self.lr_scheduler.step()
                # wandb.log({"lr": self.lr_scheduler.get_last_lr()[0]}, step=self.step)
                # print(f"lr: {self.lr_scheduler.get_last_lr()}")
                pbar.update(1)

        accelerator.print("training complete")

    def wandb_summary(self, all_images, sampled_videos, all_rgb, misc):
        print("wandb summary")
        (
            input,
            t_gt,
            ctxt_rgb,
            t,
            depth,
            output,
            frames,
            rendered_ctxt_img,
            rendered_ctxt_depth,
            rendered_trgt_img,
            rendered_trgt_depth,
            rendered_trgt_feats,
            depth_frames,
            masks,
            target_patch,
            target_out,
            rgb_intermediate,
            x_self_cond,
            render_cond,
        ) = misc
        # gt = gt[:10]
        log_dict = {
            "sanity/denoised_min": output.min(),
            "sanity/denoised_max": output.max(),
            "sanity/noisy_input_min": input.min(),
            "sanity/noisy_input_max": input.max(),
            "sanity/ctxt_rgb_min": ctxt_rgb.min(),
            "sanity/ctxt_rgb_max": ctxt_rgb.max(),
            "sanity/depth_min": depth.min(),
            "sanity/depth_max": depth.max(),
            "sanity/render_cond": render_cond,
        }
        if rendered_trgt_depth is not None:
            log_dict.update(
                {
                    "sanity/rendered_trgt_depth_min": rendered_trgt_depth.min(),
                    "sanity/rendered_trgt_depth_max": rendered_trgt_depth.max(),
                    "sanity/rendered_trgt_img_min": rendered_trgt_img.min(),
                    "sanity/rendered_trgt_img_max": rendered_trgt_img.max(),
                }
            )
        if rendered_trgt_feats.numel() > 0:
            log_dict.update(
                {
                    "sanity/rendered_trgt_feats_min": rendered_trgt_feats.min(),
                    "sanity/rendered_trgt_feats_max": rendered_trgt_feats.max(),
                }
            )
        t_gt = rearrange(t_gt, "b t c h w -> (b t) c h w")
        ctxt_rgb = rearrange(ctxt_rgb, "b t c h w -> (b t) c h w")
        b, c, h, w = t_gt.shape
        depth = torch.from_numpy(
            jet_depth(depth[:].cpu().detach().view(-1, h, w))
        ).permute(0, 3, 1, 2)
        depths = make_grid(depth)
        depths = wandb.Image(depths.permute(1, 2, 0).numpy())

        def prepare_depths(depth):
            depth = torch.from_numpy(
                jet_depth(depth[:].cpu().detach().view(-1, h, w))
            ).permute(0, 3, 1, 2)
            depths = make_grid(depth)
            depths = wandb.Image(depths.permute(1, 2, 0).numpy())
            return depths

        # clamp input, output, target to [0, 1]
        input = torch.clamp(input, 0, 1)
        output = torch.clamp(output, 0, 1)
        t_gt = torch.clamp(t_gt, 0, 1)
        ctxt_rgb = torch.clamp(ctxt_rgb, 0, 1)

        # rendered_ctxt_depths = prepare_depths(rendered_ctxt_depth)
        image_dict = {
            "visualization/depth": depths,
            "visualization/noisy_input": wandb.Image(
                make_grid(input[:].cpu().detach()).permute(1, 2, 0).numpy()
            ),
            "result/output": wandb.Image(
                make_grid(output[:].cpu().detach()).permute(1, 2, 0).numpy()
            ),
            "result/target": wandb.Image(
                make_grid(t_gt[:].cpu().detach()).permute(1, 2, 0).numpy()
            ),
            "result/ctxt_rgb": wandb.Image(
                make_grid(ctxt_rgb[:].cpu().detach()).permute(1, 2, 0).numpy()
            ),
            "visualization/target_patch": wandb.Image(
                make_grid(target_patch[:].cpu().detach()).permute(1, 2, 0).numpy()
            ),
            "result/target_out": wandb.Image(
                make_grid(target_out[:].cpu().detach()).permute(1, 2, 0).numpy()
            ),
        }
        if rgb_intermediate is not None:
            image_dict.update(
                {
                    "visualization/rgb_intermediate": wandb.Image(
                        make_grid(rgb_intermediate[:].cpu().detach())
                        .permute(1, 2, 0)
                        .numpy()
                    )
                }
            )
        if x_self_cond is not None:
            image_dict.update(
                {
                    "result/x_self_cond": wandb.Image(
                        make_grid(x_self_cond[:].cpu().detach())
                        .permute(1, 2, 0)
                        .numpy()
                    )
                }
            )

        if masks is not None:
            masks = rearrange(masks, "b t c h w -> (b t) c h w").cpu().detach()
            masks = repeat(masks, "b c h w -> b (n c) h w", n=3)
            masks = make_grid(masks)
            masks = wandb.Image(masks.permute(1, 2, 0).numpy())
            image_dict.update({"visualization/masks": masks})

        if rendered_trgt_depth is not None:
            rendered_trgt_depths = prepare_depths(rendered_trgt_depth)
            image_dict.update(
                {
                    # "visualization/rendered_ctxt_depth": rendered_ctxt_depths,
                    # "visualization/rendered_ctxt_img": wandb.Image(
                    #     make_grid(rendered_ctxt_img[:10].cpu().detach())
                    #     .permute(1, 2, 0)
                    #     .numpy()
                    # ),
                    "visualization/rendered_trgt_depth": rendered_trgt_depths,
                }
            )
        # check if rendered_trgt_feats has 0 size
        # if rendered_trgt_feats.shape[0] != 0:
        # :
        if rendered_trgt_feats.numel() > 0:
            image_dict.update(
                {
                    "visualization/rendered_trgt_feats": wandb.Image(
                        make_grid(rendered_trgt_feats[:10][:, 3:6, ...].cpu().detach())
                        .permute(1, 2, 0)
                        .numpy()
                    ),
                }
            )

        if rendered_trgt_img is not None:
            image_dict.update(
                {
                    "visualization/rendered_trgt_img": wandb.Image(
                        make_grid(rendered_trgt_img[:10].cpu().detach())
                        .permute(1, 2, 0)
                        .numpy()
                    ),
                }
            )

        if all_images is not None:
            log_dict.update(
                {
                    "sanity/sample_min": all_images.min(),
                    "sanity/sample_max": all_images.max(),
                }
            )
            images = make_grid(all_images.cpu().detach())
            images = wandb.Image(images.permute(1, 2, 0).numpy())
            image_dict.update({"visualization/samples": images})
            if all_rgb is not None:
                rgb = make_grid(all_rgb.cpu().detach())
                rgb = wandb.Image(rgb.permute(1, 2, 0).numpy())
                image_dict.update({"visualization/rgb": rgb})

        wandb.log(log_dict)
        wandb.log(image_dict)

        run_dir = wandb.run.dir
        for f in range(len(frames)):
            frames[f] = rearrange(frames[f], "b h w c -> h (b w) c")
        denoised_f = os.path.join(run_dir, "denoised_view_circle.mp4")
        imageio.mimwrite(denoised_f, frames, fps=8, quality=7)

        if sampled_videos is not None:
            for f in range(len(sampled_videos[0])):
                sampled_videos[0][f] = rearrange(
                    sampled_videos[0][f], "b h w c -> h (b w) c"
                )
            sampled_f = os.path.join(run_dir, "sampled_view_circle.mp4")
            imageio.mimwrite(sampled_f, sampled_videos[0], fps=8, quality=7)
            wandb.log(
                {
                    "vid/sampled_view_circle": wandb.Video(
                        sampled_f, format="mp4", fps=8
                    ),
                }
            )
        wandb.log(
            {"vid/denoised_view_circle": wandb.Video(denoised_f, format="mp4", fps=8),}
        )

        for f in range(len(depth_frames)):
            depth_frames[f] = rearrange(depth_frames[f], "(n b) h w -> n h (b w)", n=1)

        depth = torch.cat(depth_frames, dim=0)
        depth = (
            torch.from_numpy(
                jet_depth(
                    depth[:].cpu().detach().view(depth.shape[0], self.image_size, -1)
                )
            )
            * 255
        )
        # convert depth to list of images
        depth_frames = []
        for i in range(depth.shape[0]):
            depth_frames.append(depth[i].cpu().detach().numpy().astype(np.uint8))

        denoised_f_depth = os.path.join(run_dir, "denoised_view_circle_depth.mp4")
        imageio.mimwrite(denoised_f_depth, depth_frames, fps=8, quality=7)
        wandb.log(
            {
                "vid/denoised_view_circle_depth": wandb.Video(
                    denoised_f_depth, format="mp4", fps=8
                ),
            }
        )

        print(f"end wandb summary sample")
