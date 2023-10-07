import sys
import os
import wandb
import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    GaussianDiffusion,
    Trainer,
)
import data_io
from einops import rearrange

# from PixelNeRF import PixelNeRFModel, PixelNeRFImageModel
from PixelNeRF import PixelNeRFModelCond
import torch
import imageio
import numpy as np
from utils import *
from torchvision.utils import make_grid
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
from results_configs import re_indices
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image


@hydra.main(
    version_base=None, config_path="../configurations/", config_name="config",
)
def train(cfg: DictConfig):
    run = wandb.init(**cfg.wandb)
    wandb.run.log_code(".")
    wandb.run.name = cfg.name
    print(f"run dir: {run.dir}")
    run_dir = run.dir
    wandb.save(os.path.join(run_dir, "checkpoint*"))
    wandb.save(os.path.join(run_dir, "video*"))
    # initialize the accelerator at the beginning
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        split_batches=True, mixed_precision="no", kwargs_handlers=[ddp_kwargs],
    )

    # dataset
    train_batch_size = cfg.batch_size
    dataset = data_io.get_dataset(cfg)
    dl = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    torch.manual_seed(0)
    # torch.manual_seed(500)

    # model = PixelNeRFModel(near=1.2, far=4.0, dim=64, dim_mults=(1, 1, 2, 4)).cuda()
    render_settings = {
        "n_coarse": 64,
        "n_fine": 64,
        "n_coarse_coarse": 32,
        "n_coarse_fine": 0,
        "num_pixels": 64 ** 2,
        "n_feats_out": 64,
        "num_context": 1,
        "sampling": "patch",
        "cnn_refine": False,
        "self_condition": False,
        "lindisp": False,
        # "cnn_refine": True,
    }

    model = PixelNeRFModelCond(
        near=dataset.z_near,
        far=dataset.z_far,
        model=cfg.model_type,
        background_color=dataset.background_color,
        viz_type=cfg.dataset.viz_type,
        use_first_pool=cfg.use_first_pool,
        mode=cfg.mode,
        feats_cond=cfg.feats_cond,
        use_high_res_feats=True,
        render_settings=render_settings,
        use_viewdir=False,
        image_size=dataset.image_size,
        # use_viewdir=True,
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=dataset.image_size,
        timesteps=1000,  # number of steps
        sampling_timesteps=cfg.sampling_steps,
        loss_type="l2",  # L1 or L2
        objective="pred_x0",
        beta_schedule="cosine",
        use_guidance=cfg.use_guidance,
        guidance_scale=cfg.guidance_scale,
        temperature=cfg.temperature,
    ).cuda()

    print(f"using lr {cfg.lr}")
    trainer = Trainer(
        diffusion,
        dataloader=dl,
        train_batch_size=train_batch_size,
        train_lr=cfg.lr,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        sample_every=1000,
        wandb_every=50,
        save_every=5000,
        num_samples=1,
        warmup_period=1_000,
        checkpoint_path=cfg.checkpoint_path,
        wandb_config=cfg.wandb,
        run_name=cfg.name,
        accelerator=accelerator,
        cfg=cfg,
    )
    sampling_type = cfg.sampling_type
    use_dataset_pose = True
    render_orig_traj = True
    nsamples = 2

    if sampling_type == "simple":
        nsamples = 50
        with torch.no_grad():
            for _ in range(1):
                step = 150
                for i in re_indices.interesting_indices:
                    print(f"Starting rendering step {i}")
                    video_idx = i[0]

                    start_idx = 0
                    end_idx = start_idx + step

                    ctxt_idx = [start_idx]
                    trgt_idx = np.array([end_idx], dtype=np.int64)

                    if i[1] == "flip":
                        trgt_idx = np.array([start_idx], dtype=np.int64)
                        ctxt_idx = [end_idx]

                    ctxt_idx_np = np.array(ctxt_idx, dtype=np.int64)
                    trgt_idx_np = np.array(trgt_idx, dtype=np.int64)

                    print(f"Starting rendering step {ctxt_idx_np}, {trgt_idx_np}")
                    data = dataset.data_for_video(
                        video_idx=video_idx, ctxt_idx=ctxt_idx_np, trgt_idx=trgt_idx_np,
                    )
                    inp = to_gpu(data[0], "cuda")
                    for k in inp.keys():
                        inp[k] = inp[k].unsqueeze(0)
                    inp["num_frames_render"] = 30

                    if not use_dataset_pose:
                        poses = trainer.model.model.compute_poses(
                            "interpolation", inp, 5, max_angle=30
                        )
                        print(f"poses shape: {poses.shape}")
                        inp["render_poses"] = poses
                        inp["trgt_c2w"] = poses[-1].unsqueeze(0).unsqueeze(0).cuda()

                    if not render_orig_traj:
                        del inp["render_poses"]
                    print(f"len of idx: {len(ctxt_idx)}")
                    for j in range(nsamples):
                        print(f"Starting sample {j}")
                        out = trainer.ema.ema_model.sample(batch_size=1, inp=inp)
                        (
                            frames,
                            depth_frames,
                            conditioning_depth_img,
                        ) = prepare_video_viz(out)

                        # save to disk
                        folder = os.path.join(run_dir, f"videos_{video_idx}_{j}")
                        os.makedirs(
                            folder, exist_ok=True,
                        )
                        print(f"saving, len(frames): {len(frames)}")
                        for frame_idx, frame in enumerate(frames):
                            Image.fromarray(frame).save(
                                os.path.join(folder, f"frame_{frame_idx:04d}.png",)
                            )
                        for frame_idx, frame in enumerate(depth_frames):
                            Image.fromarray(frame).save(
                                os.path.join(
                                    folder, f"frame_depth_{frame_idx:04d}.png",
                                )
                            )

                        denoised_f = os.path.join(run_dir, "denoised_view_circle.mp4")
                        imageio.mimwrite(denoised_f, frames, fps=10, quality=10)
                        denoised_f_depth = os.path.join(
                            run_dir, "denoised_view_circle_depth.mp4"
                        )
                        imageio.mimwrite(
                            denoised_f_depth, depth_frames, fps=10, quality=10
                        )
                        wandb.log(
                            {
                                "vid/interp": wandb.Video(
                                    denoised_f,
                                    format="mp4",
                                    fps=10,
                                    caption=f"video {i}",
                                ),
                                "vid/interp_depth": wandb.Video(
                                    denoised_f_depth, format="mp4", fps=10
                                ),
                            }
                        )
                        ctxt_img = (
                            trainer.model.unnormalize(inp["ctxt_rgb"][:, 0])
                            .cpu()
                            .detach()
                        )
                        ctxt_img = torch.clip(ctxt_img, 0, 1)
                        trgt_img = (
                            trainer.model.unnormalize(inp["trgt_rgb"][:, 0])
                            .cpu()
                            .detach()
                        )
                        trgt_img = torch.clip(trgt_img, 0, 1)
                        image_dict = {
                            "result/trgt_rgb": wandb.Image(
                                make_grid(trgt_img).permute(1, 2, 0).numpy()
                            ),
                            "result/ctxt_rgb": wandb.Image(
                                make_grid(ctxt_img).permute(1, 2, 0).numpy()
                            ),
                            "result/input_render": wandb.Image(
                                make_grid(out["rgb"].cpu().detach())
                                .permute(1, 2, 0)
                                .numpy()
                            ),
                            "result/output": wandb.Image(
                                make_grid(
                                    trainer.model.normalize(out["images"])
                                    .cpu()
                                    .detach()
                                )
                                .permute(1, 2, 0)
                                .numpy()
                            ),
                            "result/input_depth": wandb.Image(
                                make_grid(conditioning_depth_img)
                                .permute(1, 2, 0)
                                .numpy()
                            ),
                        }

                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection="3d")
                        trans = out["render_poses"][:, :3, -1].cpu()
                        print(f"trans {trans.shape}")
                        ax.plot(trans[:, 0], trans[:, 1], trans[:, 2])
                        ax.view_init(elev=10.0, azim=45)

                        # Turn off tick labels
                        ax.set_yticklabels([])
                        ax.set_xticklabels([])
                        ax.set_zticklabels([])

                        plt.savefig(f"{run_dir}/trajectory.png")
                        plt.close(fig)
                        image_dict["result/trajectory"] = wandb.Image(
                            plt.imread(f"{run_dir}/trajectory.png")
                        )

                        # concat all frames along width into one image
                        for f in range(len(frames)):
                            frames[f] = torch.from_numpy(frames[f])
                            depth_frames[f] = torch.from_numpy(depth_frames[f])
                        print(len(frames))

                        target_depth = depth_frames[-1]
                        frames = frames[1:][::4][2:-1]
                        depth_frames = depth_frames[1:][::4][2:-1]

                        frames_cat = torch.cat(frames, dim=1)
                        depth_frames_cat = torch.cat(depth_frames, dim=1)

                        image_dict["result/concat_video"] = wandb.Image(
                            make_grid(frames_cat).numpy()
                        )
                        image_dict["result/concat_video_depth"] = wandb.Image(
                            make_grid(depth_frames_cat).numpy()
                        )
                        image_dict["result/target_depth"] = wandb.Image(
                            make_grid(target_depth).numpy()
                        )

                        wandb.log(image_dict)
    elif sampling_type == "autoregressive":
        ###### use generated images autoregressively to sample ######
        for video_data in re_indices.interesting_indices:
            video_idx = video_data[0]
            flip = video_data[1]

            sampled_frame = None
            prev_trgt_c2w = None
            step = 150
            nsamples = 2
            ntimesteps = 3
            with torch.no_grad():
                for _ in range(nsamples):
                    for i in range(ntimesteps):
                        if flip == "flip":
                            start_idx = (ntimesteps - i - 1) * step
                            end_idx = ntimesteps * step - 1
                            trgt_idx = [start_idx]
                            ctxt_idx = [end_idx]
                        else:
                            start_idx = 0
                            end_idx = (i + 1) * step - 1
                            trgt_idx = [end_idx]
                            ctxt_idx = [start_idx]

                        ctxt_idx_np = np.array(ctxt_idx, dtype=np.int64)
                        trgt_idx_np = np.array(trgt_idx, dtype=np.int64)

                        print(f"Starting rendering step {ctxt_idx_np}, {trgt_idx_np}")
                        if i == 0:
                            data = dataset.data_for_video(
                                video_idx=video_idx,
                                ctxt_idx=ctxt_idx_np,
                                trgt_idx=trgt_idx_np,
                            )
                            inp = to_gpu(data[0], "cuda")
                            if not render_orig_traj:
                                del inp["render_poses"]

                            for k in inp.keys():
                                inp[k] = inp[k].unsqueeze(0)

                            if not use_dataset_pose:
                                poses = trainer.model.model.compute_poses(
                                    "spherical", inp, radius=0.0, n=90, max_angle=340
                                )
                                inp["render_poses"] = poses[: end_idx + 1]
                                print(f"poses shape: {poses.shape}")
                        else:
                            # concatenate the sampled frame to the input ctxt_rgb
                            inp["ctxt_rgb"] = torch.cat(
                                (
                                    inp["ctxt_rgb"],  # [:, -1:],
                                    trainer.model.normalize(sampled_frame).unsqueeze(0),
                                ),
                                dim=1,
                            )
                            inp["ctxt_c2w"] = torch.cat(
                                (inp["ctxt_c2w"], prev_trgt_c2w), dim=1
                            )
                            # hard code the target camera pose
                            if use_dataset_pose:
                                data = dataset.data_for_video(
                                    video_idx=video_idx,
                                    ctxt_idx=ctxt_idx_np,
                                    trgt_idx=trgt_idx_np,
                                )
                                inp_temp = to_gpu(data[0], "cuda")
                                inp["trgt_c2w"] = inp_temp["trgt_c2w"].unsqueeze(0)
                                inp["render_poses"] = inp_temp[
                                    "render_poses"
                                ].unsqueeze(0)
                            else:
                                inp["trgt_c2w"] = (
                                    poses[end_idx].unsqueeze(0).unsqueeze(0).cuda()
                                )
                                inp["render_poses"] = poses[: end_idx + 1, ...]

                        num_context = inp["ctxt_rgb"].shape[1]
                        print(f"num_context: {inp['ctxt_c2w'].shape[1]}")

                        for j in range(1):
                            print(f"Starting sample {j}")
                            inp["render_top_down"] = False
                            inp["num_frames_render"] = 30
                            out = trainer.ema.ema_model.sample(batch_size=1, inp=inp)
                            sampled_frame = out["images"]
                            (
                                frames,
                                depth_frames,
                                conditioning_depth_img,
                            ) = prepare_video_viz(out)

                            # save to disk
                            folder = os.path.join(run_dir, f"videos_{video_idx}_{_}")
                            os.makedirs(
                                folder, exist_ok=True,
                            )
                            print(f"saving, len(frames): {len(frames)}")
                            for frame_idx, frame in enumerate(frames):
                                Image.fromarray(frame).save(
                                    os.path.join(folder, f"frame_{frame_idx:04d}.png",)
                                )
                            for frame_idx, frame in enumerate(depth_frames):
                                Image.fromarray(frame).save(
                                    os.path.join(
                                        folder, f"frame_depth_{frame_idx:04d}.png",
                                    )
                                )

                            denoised_f = os.path.join(
                                run_dir, "denoised_view_circle.mp4"
                            )
                            imageio.mimwrite(denoised_f, frames, fps=10, quality=10)
                            denoised_f_depth = os.path.join(
                                run_dir, "denoised_view_circle_depth.mp4"
                            )
                            imageio.mimwrite(
                                denoised_f_depth, depth_frames, fps=10, quality=10
                            )
                            wandb.log(
                                {
                                    "vid/interp": wandb.Video(
                                        denoised_f,
                                        format="mp4",
                                        fps=10,
                                        caption=f"video {i}",
                                    ),
                                    "vid/interp_depth": wandb.Video(
                                        denoised_f_depth, format="mp4", fps=10
                                    ),
                                }
                            )
                            ctxt_img = (
                                trainer.model.unnormalize(inp["ctxt_rgb"][:, 0])
                                .cpu()
                                .detach()
                            )
                            ctxt_img = torch.clip(ctxt_img, 0, 1)
                            trgt_img = (
                                trainer.model.unnormalize(inp["trgt_rgb"][:, 0])
                                .cpu()
                                .detach()
                            )
                            trgt_img = torch.clip(trgt_img, 0, 1)
                            print(
                                f"trgt img shape: {trgt_img.shape}, cond depth: {conditioning_depth_img.shape}"
                            )
                            image_dict = {
                                "result/trgt_rgb": wandb.Image(
                                    make_grid(trgt_img).permute(1, 2, 0).numpy()
                                ),
                                "result/ctxt_rgb": wandb.Image(
                                    make_grid(ctxt_img).permute(1, 2, 0).numpy()
                                ),
                                "result/input_render": wandb.Image(
                                    make_grid(out["rgb"].cpu().detach())
                                    .permute(1, 2, 0)
                                    .numpy()
                                ),
                                "result/output": wandb.Image(
                                    make_grid(
                                        trainer.model.normalize(out["images"])
                                        .cpu()
                                        .detach()
                                    )
                                    .permute(1, 2, 0)
                                    .numpy()
                                ),
                                "result/input_depth": wandb.Image(
                                    make_grid(conditioning_depth_img)
                                    .permute(1, 2, 0)
                                    .numpy()
                                ),
                            }

                            fig = plt.figure()
                            ax = fig.add_subplot(111, projection="3d")
                            trans = out["render_poses"][:, :3, -1].cpu()
                            print(f"trans {trans.shape}")
                            ax.plot(trans[:, 0], trans[:, 1], trans[:, 2])
                            ax.view_init(elev=10.0, azim=45)

                            # Turn off tick labels
                            ax.set_yticklabels([])
                            ax.set_xticklabels([])
                            ax.set_zticklabels([])

                            plt.savefig(f"{run_dir}/trajectory.png")
                            plt.close(fig)
                            image_dict["result/trajectory"] = wandb.Image(
                                plt.imread(f"{run_dir}/trajectory.png")
                            )

                            # concat all frames along width into one image
                            for f in range(len(frames)):
                                frames[f] = torch.from_numpy(frames[f])
                                depth_frames[f] = torch.from_numpy(depth_frames[f])
                            print(len(frames))
                            frames = frames[3:][::4]
                            depth_frames = depth_frames[3:][::4]

                            frames_cat = torch.cat(frames, dim=1)
                            depth_frames_cat = torch.cat(depth_frames, dim=1)
                            # add to image dict
                            image_dict["result/concat_video"] = wandb.Image(
                                make_grid(frames_cat).numpy()
                            )
                            image_dict["result/concat_video_depth"] = wandb.Image(
                                make_grid(depth_frames_cat).numpy()
                            )
                            wandb.log(image_dict)
                        # ctxt_idx = ctxt_idx + [end_idx]
                        prev_trgt_c2w = inp["trgt_c2w"]



def prepare_video_viz(out):
    frames = out["videos"]
    for f in range(len(frames)):
        frames[f] = rearrange(frames[f], "b h w c -> h (b w) c")

    depth_frames = out["depth_videos"]
    for f in range(len(depth_frames)):
        depth_frames[f] = rearrange(depth_frames[f], "(n b) h w -> n h (b w)", n=1)

    conditioning_depth = out["conditioning_depth"].cpu()
    # resize to depth_frames
    conditioning_depth = F.interpolate(
        conditioning_depth[:, None],
        size=depth_frames[0].shape[-2:],
        mode="bilinear",
        antialias=True,
    )[:, 0]
    depth_frames.append(conditioning_depth)

    depth = torch.cat(depth_frames, dim=0)
    print(f"depth shape: {depth.shape}")

    depth = (
        torch.from_numpy(
            jet_depth(depth[:].cpu().detach().view(depth.shape[0], depth.shape[-1], -1))
        )
        * 255
    )
    # convert depth to list of images
    depth_frames = []
    for i in range(depth.shape[0]):
        depth_frames.append(depth[i].cpu().detach().numpy().astype(np.uint8))
    return (
        frames,
        depth_frames[:-1],
        rearrange(torch.from_numpy(depth_frames[-1]), "h w c -> () c h w"),
    )

if __name__ == "__main__":
    train()
