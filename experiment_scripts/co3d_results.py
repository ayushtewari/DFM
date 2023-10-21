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
from accelerate import Accelerator

from PixelNeRF import PixelNeRFModelCond
import torch
import imageio
import numpy as np
from utils import *
from torchvision.utils import make_grid
from numpy import random
from accelerate.utils import set_seed
import lpips 
from einops import rearrange
from PIL import Image 

def video_summary(run_dir, frames, depth_frames, name, fps=8):
    denoised_f = os.path.join(run_dir, f"rgb_{name}.mp4")
    imageio.mimwrite(denoised_f, frames, fps=fps, quality=10)
    denoised_f_depth = os.path.join(
        run_dir, f"depth_{name}.mp4"
    )
    imageio.mimwrite(
        denoised_f_depth, depth_frames, fps=fps, quality=10
    )
    return denoised_f, denoised_f_depth

def prepare_depth(depth_frames, resolution=64):
    depth = torch.cat(depth_frames, dim=0)
    depth = (
        torch.from_numpy(
            jet_depth(depth[:].cpu().detach().view(-1, resolution, resolution))
        )
        * 255
    )

    depth_frames = []
    for i in range(depth.shape[0]):
        depth_frames.append(
            depth[i].cpu().detach().numpy().astype(np.uint8)
        )
    return depth_frames

def prepare_video_out(out, resolution=64):
    frames = out["videos"]
    depth_frames = out["depth_videos"]
    conditioning_depth = out["conditioning_depth"].cpu()
    # resize to depth_frames
    conditioning_depth_img = torch.nn.functional.interpolate(
        conditioning_depth[:, None],
        size=depth_frames[0].shape[-2:],
        mode="bilinear",
        antialias=True,
    )[:, 0]
    conditioning_depth_img = torch.from_numpy(jet_depth(conditioning_depth_img.cpu().detach()[0]) * 255.0).permute(2, 0, 1)[None]
        
    depth_frames = prepare_depth(depth_frames, resolution=resolution)

    return frames, depth_frames, conditioning_depth_img, out["depth_videos"] 

def concat_list_to_image(frames, stride=4):
    new_frames = [] 
    for f in range(len(frames)):
        new_frames.append(torch.from_numpy(frames[f]))

    frames = new_frames
    frames = frames[::stride]
    frames_cat = torch.cat(frames, dim=1)

    return frames_cat 

@hydra.main(
    version_base=None, config_path="../configurations/", config_name="config",
)
def train(cfg: DictConfig):
    accelerator = Accelerator(
        split_batches=True, mixed_precision="no", 
    )
    set_seed(cfg.seed)

    run = wandb.init(**cfg.wandb)
    wandb.run.log_code(".")
    wandb.run.name = cfg.name
    print(f"run dir: {run.dir}")
    run_dir = run.dir
    wandb.save(os.path.join(run_dir, "checkpoint*"))
    wandb.save(os.path.join(run_dir, "video*"))

    # dataset
    train_batch_size = 1
    cfg.stage = 'val'
    dataset = data_io.get_dataset(cfg)
    dataset.num_context = 1
    dl = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
    )

    render_settings = {
        "n_coarse": 64,
        "n_fine": 64,
        "n_coarse_coarse": 64,
        "n_coarse_fine": 0,
        "num_pixels": 32 ** 2,
        "n_feats_out": 64,
        "num_context": 1,
        "sampling": "patch",
        "self_condition": False,
        "cnn_refine": False,
        "lindisp": False
    }
    model = PixelNeRFModelCond(
        near=1.0, # dataset.z_near, we set this to be slightly larger than the one we used for training to avoid floaters 
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
        use_abs_pose=cfg.use_abs_pose
    ).to(accelerator.device)

    diffusion = GaussianDiffusion(
        model,
        image_size=dataset.image_size,
        timesteps=1000,  # number of steps
        sampling_timesteps=50,
        loss_type="l2",  # L1 or L2
        objective="pred_x0",
        beta_schedule="cosine",
    ).to(accelerator.device)

    trainer = Trainer(
        diffusion,
        accelerator=accelerator,
        dataloader=dl,
        train_batch_size=train_batch_size,
        train_lr=cfg.lr,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        sample_every=1000,
        wandb_every=500,
        save_every=5000,
        num_samples=1,
        warmup_period=1_000,
        checkpoint_path=cfg.checkpoint_path,
        wandb_config=cfg.wandb,
        run_name=cfg.name,
        cfg=cfg
    )

    perceptual_loss = lpips.LPIPS(net="vgg")

    sampling_type = cfg.sampling_type
    use_dataset_pose = cfg.use_dataset_pose

    video_indices = list(range(len(dataset)))

    if sampling_type == "autoregressive":
        with torch.no_grad():
            for video_idx in video_indices: 
                data = dataset.data_for_video(
                    video_idx=video_idx,
                    load_all_frames=True
                )[0] 

                data = to_gpu(data, accelerator.device)

                for j in range(1):
                    print(f"Starting sample {j}")

                    sampled_frames = [] 
                    step = cfg.test_autoregressive_stepsize 

                    all_ctxt_rgb = data['ctxt_rgb']
                    all_ctxt_c2w = data['ctxt_c2w']
                    all_trgt_rgb = data['trgt_rgb']
                    all_trgt_c2w = data['trgt_c2w']
                    all_intrinsics = data['intrinsics']
                    all_x_pix = data['x_pix']
                    all_ctxt_abs_camera_poses = data['ctxt_abs_camera_poses']
                    all_trgt_abs_camera_poses = data['trgt_abs_camera_poses']
                    
                    current_ctxt_rgb = all_ctxt_rgb[0:1]
                    current_ctxt_c2w = all_ctxt_c2w[0:1]
                    current_all_ctxt_abs_camera_poses = all_ctxt_abs_camera_poses[0:1]

                    if not use_dataset_pose:
                        render_poses = [] 
                        render_step = 10 
                        for index in range(len(all_ctxt_c2w) // render_step):
                            ctxt_idx = index * render_step 
                            trgt_idx = min((index + 1) * render_step, len(all_trgt_rgb)-1)
                            temp_input = {
                                "ctxt_c2w": all_ctxt_c2w[ctxt_idx:ctxt_idx+1].unsqueeze(0), 
                                "trgt_c2w": all_ctxt_c2w[trgt_idx:trgt_idx+1].unsqueeze(0)
                            }
                            pose = trainer.model.model.compute_poses(
                                "interpolation", temp_input, 10
                            )
                            render_poses.append(pose)

                        render_poses.append(all_ctxt_c2w[-1:])
                        render_poses = torch.cat(render_poses, dim=0)
                        

                    for i in range(len(all_trgt_rgb) // step):
                        ctxt_idx = i * step 
                        trgt_idx = min((i + 1) * step, len(all_trgt_rgb)-1)
                        print(f"video_idx: {video_idx}, step: {i}, ctxt_idx: {ctxt_idx}, trgt_idx: {trgt_idx}")

                        inp = {
                            'ctxt_rgb': current_ctxt_rgb,
                            'ctxt_c2w': current_ctxt_c2w,
                            'trgt_c2w': all_trgt_c2w[trgt_idx:trgt_idx+1],
                            'intrinsics': all_intrinsics,
                            'trgt_rgb': all_trgt_rgb[trgt_idx:trgt_idx+1],
                            'x_pix': all_x_pix,
                            'ctxt_abs_camera_poses': current_all_ctxt_abs_camera_poses,
                            'trgt_abs_camera_poses': all_trgt_abs_camera_poses[trgt_idx:trgt_idx+1]
                        }

                        for k in inp.keys():
                            inp[k] = inp[k].unsqueeze(0)

                        if i == len(all_trgt_rgb) // step -1:
                            # last step render all frames 
                            if use_dataset_pose:
                                inp['render_poses'] = all_trgt_c2w.unsqueeze(0)
                            else:
                                # we do an interpolation between all frames 
                                inp['render_poses'] = render_poses        
                                inp["trgt_c2w"] = all_trgt_c2w[-1:].unsqueeze(0)
                                inp["trgt_abs_camera_poses"] = all_trgt_abs_camera_poses[-1:].unsqueeze(0)
                                
                        else:
                            inp['render_poses'] = inp['trgt_c2w'] # [0]


                        out = trainer.ema.ema_model.sample(batch_size=1, inp=inp)
                        sampled_frames.append(out["images"])

                        current_ctxt_rgb = torch.cat([
                            current_ctxt_rgb, trainer.model.normalize(sampled_frames[-1])], 
                            dim=0
                        )

                        current_ctxt_c2w = torch.cat([current_ctxt_c2w, all_trgt_c2w[trgt_idx:trgt_idx+1]], dim=0)
                        current_all_ctxt_abs_camera_poses = torch.cat([
                            current_all_ctxt_abs_camera_poses, all_trgt_abs_camera_poses[trgt_idx:trgt_idx+1]
                        ], dim=0)

                    frame_intermediate_rendering = (current_ctxt_rgb * 0.5 + 0.5).clamp(min=0, max=1).cpu().permute(0, 2, 3, 1)
                    frame_intermediate_rendering = torch.cat( [frame for frame in frame_intermediate_rendering], dim=1)

                    frames, depth_frames, conditioning_depth_img, depth_videos = prepare_video_out(out, resolution=dataset.image_size)

                    frames = [frame[0] for frame in frames]

                    target = (all_trgt_rgb* 0.5 + 0.5).clamp(min=0, max=1).cpu()
                    target_frames = [((frame*255).permute(1, 2, 0).numpy().astype(np.uint8)) for frame in target]

                    denoised_f, denoised_f_depth = video_summary(run_dir, frames, depth_frames, "denoised_view", fps=10)

                    target_f = os.path.join(run_dir, "target_view_circle.mp4")
                    imageio.mimwrite(
                        target_f,
                        target_frames,
                        fps=5,
                        quality=10,
                    )

                    # concat all frames along width into one image
                    frames_cat = concat_list_to_image(frames, 50)
                    depth_frames_cat = concat_list_to_image(depth_frames, 50)

                    # add to image dict
                    concat_intermediate_rendering = wandb.Image(
                        make_grid(frame_intermediate_rendering).numpy()
                    )
                    concat_video = wandb.Image(
                        make_grid(frames_cat).numpy()
                    )
                    concat_video_depth = wandb.Image(
                        make_grid(depth_frames_cat).numpy()
                    )

                    data_dict = {
                        "result/concat_video": concat_video,
                        "result/concat_video_depth": concat_video_depth,
                        "result/concat_intermediate_rendering": concat_intermediate_rendering,
                        "vid/interp": wandb.Video(
                            denoised_f,
                            format="mp4",
                            fps=4,
                            caption=f"{video_idx}",
                        ),
                        "vid/interp_depth": wandb.Video(
                            denoised_f_depth, format="mp4", fps=4
                        ),
                        "vid/target": wandb.Video(
                            target_f,
                            format="mp4",
                            fps=4,
                            caption=f"{video_idx}",
                        ),
                        "result/trgt_rgb": wandb.Image(
                            (make_grid(inp["trgt_rgb"][:, 0].cpu().detach()*0.5+0.5).clamp(min=0, max=1)*255.0)
                                .permute(1, 2, 0)
                                .numpy().astype(np.uint8)
                        ),
                        "result/ctxt_rgb": wandb.Image(
                            (make_grid(inp["ctxt_rgb"][:, 0].cpu().detach()*0.5+0.5).clamp(min=0, max=1)*255.0)
                            .permute(1, 2, 0)
                            .numpy().astype(np.uint8)
                        ),
                        "result/input_render": wandb.Image(
                            (make_grid(
                                out["rgb"].cpu().detach().clamp(min=0, max=1)
                            ) * 255.0)
                            .permute(1, 2, 0)
                            .numpy().astype(np.uint8)
                        ),
                        "result/input_depth": wandb.Image(
                            make_grid(conditioning_depth_img)
                            .permute(1, 2, 0)
                            .numpy().astype(np.uint8)
                        ),
                        "result/output": wandb.Image(
                            make_grid(
                                out["images"].clamp(min=0, max=1)
                                .cpu()
                                .detach()
                                * 255.0
                            )
                            .permute(1, 2, 0)
                            .numpy().astype(np.uint8)
                        )

                    }

                    wandb.log(
                        data_dict
                    )

                    # save all frames to disk 
                    os.makedirs(os.path.join(run_dir, "generated_frames"), exist_ok=True)
                    for frame_idx, frame in enumerate(frames):
                        Image.fromarray(frame).save(f"{run_dir}/generated_frames/video_{video_idx:04d}_frame_{frame_idx:04d}_var{j}.png")               

                    for frame_idx, frame in enumerate(frames):
                        # save depth map
                        depth = depth_videos[frame_idx][0].unsqueeze(-1)
                        rgbd = torch.cat([torch.from_numpy(frame), depth], dim=-1)
                        np.save(
                            f"{run_dir}/generated_frames/{video_idx:04d}_frame_{frame_idx:04d}_depth_var{j}.npy",
                            rgbd.numpy(),
                        )

    elif sampling_type == "oneshot":
        with torch.no_grad():
            for video_idx in video_indices:
                print(f"video_idx: {video_idx}, len: {len(dataset)}")

                data = dataset.data_for_video(
                    video_idx=video_idx,
                    load_all_frames=True
                )

                inp = to_gpu(data[0], "cuda")
                for k in inp.keys():
                    inp[k] = inp[k].unsqueeze(0)

                if use_dataset_pose:
                    inp['render_poses'] = inp['trgt_c2w']
                else:
                    poses = trainer.model.model.compute_poses(
                        "interpolation", inp, 40
                    )

                    last_frame_pose = inp["trgt_c2w"][:, 0:1]
                    poses = torch.cat([poses, last_frame_pose[0]], dim=0)
                    inp["render_poses"] = poses.unsqueeze(0) # [20, 4, 4]
                
                inp["ctxt_rgb"] = inp["ctxt_rgb"][:, 0:1]
                inp["ctxt_c2w"] = inp["ctxt_c2w"][:, 0:1]
                inp["ctxt_abs_camera_poses"] = inp["ctxt_abs_camera_poses"][:, 0:1]
                inp["trgt_c2w"] = inp["trgt_c2w"][:, 100:101]
                inp["trgt_abs_camera_poses"] = inp["trgt_abs_camera_poses"][:, 100:101]

                for j in range(1):
                    print(f"Starting sample {j}")
                    out = trainer.ema.ema_model.sample(batch_size=1, inp=inp)
                    frames, depth_frames, conditioning_depth_img, depth_videos = prepare_video_out(out, resolution=dataset.image_size)

                    frames = [frame[0] for frame in frames]

                    target = (inp["trgt_rgb"][0]* 0.5 + 0.5).clamp(min=0, max=1).cpu()
                    target_frames = [((frame*255).permute(1, 2, 0).numpy().astype(np.uint8)) for frame in target]
                    
                    denoised_f, denoised_f_depth = video_summary(run_dir, frames, depth_frames, "denoised_view", fps=10)

                    target_f = os.path.join(run_dir, "rgb_target_view.mp4")
                    imageio.mimwrite(
                        target_f,
                        target_frames,
                        fps=10,
                        quality=10,
                    )

                    # concat all frames along width into one image
                    frames_cat = concat_list_to_image(frames, stride=4)
                    depth_frames_cat = concat_list_to_image(depth_frames, stride=4)
                    
                    concat_video = wandb.Image(
                        make_grid(frames_cat).numpy()
                    )
                    concat_video_depth = wandb.Image(
                        make_grid(depth_frames_cat).numpy()
                    )

                    data_dict = {
                        "result/concat_video": concat_video,
                        "result/concat_video_depth": concat_video_depth,
                        "vid/interp": wandb.Video(
                            denoised_f,
                            format="mp4",
                            fps=4,
                            caption=f"{video_idx}",
                        ),
                        "vid/interp_depth": wandb.Video(
                            denoised_f_depth, format="mp4", fps=4
                        ),
                        "vid/target": wandb.Video(
                            target_f,
                            format="mp4",
                            fps=4,
                            caption=f"{video_idx}",
                        ),
                        
                        "result/trgt_rgb": wandb.Image(
                            (make_grid(inp["trgt_rgb"][:, 0].cpu().detach()*0.5+0.5).clamp(min=0, max=1)*255.0)
                                .permute(1, 2, 0)
                                .numpy().astype(np.uint8)
                        ),
                        "result/ctxt_rgb": wandb.Image(
                            (make_grid(inp["ctxt_rgb"][:, 0].cpu().detach()*0.5+0.5).clamp(min=0, max=1)*255.0)
                            .permute(1, 2, 0)
                            .numpy().astype(np.uint8)
                        ),
                        "result/input_render": wandb.Image(
                            (make_grid(
                                out["rgb"].cpu().detach().clamp(min=0, max=1)
                            ) * 255.0)
                            .permute(1, 2, 0)
                            .numpy().astype(np.uint8)
                        ),
                        "result/input_depth": wandb.Image(
                            make_grid(conditioning_depth_img)
                            .permute(1, 2, 0)
                            .numpy().astype(np.uint8)
                        ),
                        "result/output": wandb.Image(
                            make_grid(
                                out["images"].clamp(min=0, max=1)
                                .cpu()
                                .detach()
                                * 255.0
                            )
                            .permute(1, 2, 0)
                            .numpy().astype(np.uint8)
                        )
                    }

                    wandb.log(
                        data_dict
                    )

                    # save all frames to disk 
                    os.makedirs(os.path.join(run_dir, "generated_frames"), exist_ok=True)
                    for frame_idx, frame in enumerate(frames):
                        Image.fromarray(frame).save(f"{run_dir}/generated_frames/video_{video_idx:04d}_frame_{frame_idx:04d}.png")               

                    for frame_idx, frame in enumerate(frames):
                        
                        # save depth map
                        depth = depth_videos[frame_idx][0].unsqueeze(-1)
                        rgbd = torch.cat([torch.from_numpy(frame), depth], dim=-1)
                        np.save(
                            f"{run_dir}/generated_frames/{video_idx:04d}_frame_{frame_idx:04d}_depth.npy",
                            rgbd.numpy(),
                        )

if __name__ == "__main__":
    train()
