# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch

from pathlib import Path
from collections import namedtuple
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import Adam

from einops import reduce

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

# constants
ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


class PixelNeRFModelWrapper(nn.Module):
    def __init__(
        self, model, image_size, loss_type="l1", auto_normalize=True,
    ):
        super().__init__()
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.loss_type = loss_type

        # sampling related parameters

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
        self.perceptual_loss = lpips.LPIPS(net="vgg")

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def p_losses(self, inp, t, add_noise=False, render_video=False):
        model_out, depth, misc = self.model(inp, t, add_noise=add_noise)
        weights, z_val = misc["weights"], misc["z_vals"]

        frames = None
        full_images = None
        full_depths = None
        if render_video:
            full_images, full_depths = self.model.render_full_image(inp, t)
            frames = self.model.render_video(inp, n=20, t=t)

        target = inp["trgt_rgb_sampled"]
        target = target.view(model_out.shape)
        loss = self.loss_fn(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")

        dist_loss = distortion_loss(
            weights, z_val, near=self.model.near, far=self.model.far
        )

        lpips_loss = torch.zeros(1, device=model_out.device)
        # LPIPS loss between model_out and target
        if self.model.sampling == "patch":
            len_render = misc["len_render"]
            gt = rearrange(target, "b c (h w) -> b c h w", h=len_render, w=len_render)
            lpips_loss = self.perceptual_loss(model_out.view_as(gt), gt)

        # print(f"target shape: {target.shape}, model_out shape: {model_out.shape}")
        losses = {
            "rgb_loss": loss.mean(),
            "dist_loss": dist_loss,
            "lpips_loss": lpips_loss.mean(),
            # "loss": loss.mean() + dist_loss,
        }
        return (
            losses,
            (
                self.unnormalize(model_out),
                self.unnormalize(inp["trgt_rgb"]),
                self.unnormalize(inp["ctxt_rgb"]),
                t,
                full_depths,
                full_images,
                frames,
            ),
        )

    def forward(self, inp, *args, **kwargs):
        img = inp["ctxt_rgb"]
        b = img.shape[0]
        num_context = img.shape[1]
        device = img.device
        # assert (
        #     h == img_size and w == img_size
        # ), f"height and width of image must be {img_size}"

        t = torch.zeros((b, num_context), device=device).long()
        # t = None  # deterministic
        return self.p_losses(inp, t, *args, **kwargs)


# trainer class
class Trainer(object):
    def __init__(
        self,
        reconstruction_model,
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
        amp=False,
        fp16=False,
        split_batches=True,
        warmup_period=0,
        checkpoint_path=None,
        wandb_config=None,
        run_name="pixelnerf",
    ):
        super().__init__()

        self.accelerator = accelerator
        if self.accelerator is None:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

            self.accelerator = Accelerator(
                split_batches=split_batches,
                mixed_precision="fp16" if fp16 else "no",
                kwargs_handlers=[ddp_kwargs],
            )

        # self.accelerator.native_amp = amp

        self.model = reconstruction_model

        self.num_samples = num_samples
        self.sample_every = sample_every
        self.save_every = save_every
        self.wandb_every = wandb_every

        assert self.sample_every % self.wandb_every == 0

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        dl = self.accelerator.prepare(dataloader)
        self.dl = cycle(dl)

        # optimizer
        params = [p for n, p in self.model.named_parameters() if "perceptual" not in n]
        perceoptual_params = [
            n for n, p in self.model.named_parameters() if "perceptual" in n
        ]
        print(f"perceptual params: {perceoptual_params}")
        self.opt = Adam(params, lr=train_lr, betas=adam_betas)
        lr_scheduler = get_cosine_schedule_with_warmup(
            self.opt, warmup_period, train_num_steps
        )
        self.dist_loss_scheduler = get_constant_hyperparameter_schedule_with_warmup(
            warmup_period, train_num_steps
        )
        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.opt, lr_scheduler
        )

        if checkpoint_path is not None:
            self.load(checkpoint_path)
            # self.load_from_david_checkpoint(checkpoint_path)
            # self.load_from_external_checkpoint(checkpoint_path)

        if self.accelerator.is_main_process:
            run = wandb.init(**wandb_config)
            wandb.run.log_code(".")
            wandb.run.name = run_name
            print(f"run dir: {run.dir}")
            run_dir = run.dir
            wandb.save(os.path.join(run_dir, "checkpoint*"))
            wandb.save(os.path.join(run_dir, "video*"))
            self.results_folder = Path(run_dir)
            self.results_folder.mkdir(exist_ok=True)

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

    def load(self, path):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(
            # str(self.results_folder / f"model-{milestone}.pt"), map_location=device
            str(path),
            map_location=device,
        )

        model = self.accelerator.unwrap_model(self.model)
        # print(f"model parameter names: {list(model.state_dict().keys())}")
        model.load_state_dict(data["model"], strict=False)

        self.step = data["step"]

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"], strict=False)

        if "version" in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def load_from_external_checkpoint(self, path):
        accelerator = self.accelerator
        device = accelerator.device

        model = self.accelerator.unwrap_model(self.model)
        # print(f"model parameter names: {list(model.state_dict().keys())}")
        data = torch.load(path, map_location=device)

        new_state_dict = OrderedDict()
        for key, value in data["model_state_dict"].items():
            if "enc" not in key:
                key = "model." + key[7:]  # remove `att.`
                new_state_dict[key] = value
                if key not in model.state_dict().keys():
                    print(f"enc key {key} not in model state dict")
            else:
                # print(f"key: {key}")
                new_key = "model." + key[7:]  # remove `att.`
                # new_key = "model." + key  # remove `att.`
                key1 = new_key  # new_key.replace(".enc.", ".clean_ctxt_enc.")
                # key2 = new_key.replace(".enc.", ".noisy_trgt_enc.")

                #
                new_state_dict[key1] = value
                # new_state_dict[key2] = value

                if key1 not in model.state_dict().keys():
                    print(f"key {key1} not in model state dict")
                # if key2 not in model.state_dict().keys():
                #     print(f"key {key2} not in model state dict")
        # exit()
        model.load_state_dict(new_state_dict, strict=False)

    def load_from_david_checkpoint(self, path):
        accelerator = self.accelerator
        device = accelerator.device
        model = self.accelerator.unwrap_model(self.model)
        data = torch.load(path, map_location=device)
        new_state_dict = OrderedDict()
        for key, value in data["state_dict"].items():
            if "encoder" in key:
                key = key.replace("encoder", "enc")
            new_state_dict[key] = value

        model.load_state_dict(new_state_dict, strict=False)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.0
                total_rgb_loss = 0.0
                total_dist_loss = 0.0
                total_lpips_loss = 0.0
                render_video = (
                    self.step % self.wandb_every == 0
                )  # and accelerator.is_main_process

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)  # .to(device)
                    data = to_gpu(data, device)
                    if isinstance(data, list):
                        gt = data[1]
                        data = data[0]

                    add_noise = self.step < 0
                    with self.accelerator.autocast():
                        losses, misc = self.model(
                            data, render_video=render_video, add_noise=add_noise
                        )

                        # loss = losses["loss"]
                        # loss = loss / self.gradient_accumulate_every
                        # total_loss += loss.item()

                        rgb_loss = losses["rgb_loss"]
                        rgb_loss = rgb_loss / self.gradient_accumulate_every
                        total_rgb_loss += rgb_loss.item()

                        dist_loss = losses["dist_loss"]
                        dist_loss = dist_loss / self.gradient_accumulate_every
                        total_dist_loss += dist_loss.item()

                        dist_weight = self.dist_loss_scheduler(self.step) * 0.0

                        lpips_loss = losses["lpips_loss"]
                        total_lpips_loss += lpips_loss.item()
                        loss = rgb_loss + dist_weight * dist_loss + lpips_loss * 0.0

                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                if accelerator.is_main_process:
                    wandb.log(
                        {
                            "loss": total_loss,
                            "rgb_loss": total_rgb_loss,
                            "dist_loss": total_dist_loss,
                            "lpips_loss": total_lpips_loss,
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "dist_weight": dist_weight,
                        },
                        step=self.step,
                    )

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                all_images = None
                all_videos_list = None
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step % self.wandb_every == 0:
                        self.wandb_summary(misc)

                    if self.step != 0 and self.step % self.save_every == 0:
                        milestone = self.step // self.save_every
                        self.save(milestone)
                self.step += 1

                self.lr_scheduler.step()
                pbar.update(1)

        accelerator.print("training complete")

    def wandb_summary(self, misc):
        print("wandb summary")
        (sampled_output, t_gt, ctxt_rgb, t, depth, output, frames,) = misc

        log_dict = {
            "sanity/gt_min": t_gt.min(),
            "sanity/gt_max": t_gt.max(),
            "sanity/rendered_min": output.min(),
            "sanity/rendered_max": output.max(),
            "sanity/sampled_rendered_min": sampled_output.min(),
            "sanity/sampled_rendered_max": sampled_output.max(),
            "sanity/ctxt_rgb_min": ctxt_rgb.min(),
            "sanity/ctxt_rgb_max": ctxt_rgb.max(),
            "sanity/depth_min": depth.min(),
            "sanity/depth_max": depth.max(),
        }

        t_gt = rearrange(t_gt, "b t c h w -> (b t) c h w")
        ctxt_rgb = rearrange(ctxt_rgb, "b t c h w -> (b t) c h w")
        b, c, h, w = t_gt.shape
        depth = torch.from_numpy(
            jet_depth(depth.cpu().detach().view(-1, h, w))
        ).permute(0, 3, 1, 2)
        depths = make_grid(depth)
        depths = wandb.Image(depths.permute(1, 2, 0).numpy())

        image_dict = {
            "visualization/depth": depths,
            "visualization/output": wandb.Image(
                make_grid(output.cpu().detach()).permute(1, 2, 0).numpy()
            ),
            "visualization/target": wandb.Image(
                make_grid(t_gt.cpu().detach()).permute(1, 2, 0).numpy()
            ),
            "visualization/ctxt_rgb": wandb.Image(
                make_grid(ctxt_rgb.cpu().detach()).permute(1, 2, 0).numpy()
            ),
        }

        wandb.log(log_dict)
        wandb.log(image_dict)

        run_dir = wandb.run.dir
        denoised_f = os.path.join(run_dir, "denoised_view_circle.mp4")
        imageio.mimwrite(denoised_f, frames, fps=8, quality=7)

        wandb.log(
            {"vid/rendered_view_circle": wandb.Video(denoised_f, format="mp4", fps=8),}
        )
