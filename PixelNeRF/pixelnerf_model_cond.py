import torch, torchvision
from torch import nn, einsum
import torch.nn.functional as F
import sys
import os
import numpy as np
import functools
from einops import rearrange, repeat

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers import *
from PixelNeRF.renderer import *
from PixelNeRF.resnetfc import *
from utils import *

from PixelNeRF.pixelnerf_helpers import *
from PixelNeRF.resnet import PixelNeRFTimeEmbed, BasicBlockTimeEmbed
from PixelNeRF.transformer.DiT import DiT


class PixelNeRFModelCond(nn.Module):
    def __init__(
        self,
        near,
        far,
        model="dit",
        viz_type="spherical",
        channels=3,
        background_color=torch.ones((3,), dtype=torch.float32),
        use_first_pool=True,
        mode="nocond",
        feats_cond=False,
        use_high_res_feats=False,
        render_settings=None,
        use_viewdir=False,
        image_size=64,
        use_abs_pose=False
    ):
        super().__init__()
        n_coarse = render_settings["n_coarse"]
        n_fine = render_settings["n_fine"]
        n_coarse_coarse = render_settings["n_coarse_coarse"]
        n_coarse_fine = render_settings["n_coarse_fine"]
        n_feats_out = render_settings["n_feats_out"]
        self.n_feats_out = n_feats_out
        self.sampling = render_settings["sampling"]
        self.self_condition = render_settings["self_condition"]
        self.image_size = image_size

        self.cnn_refine = render_settings["cnn_refine"]
        if self.cnn_refine:
            # define the cnn refine model
            self.cnn_refine_model = RefineOut(in_channels=64 + 3, out_channels=3)

        self.model = model
        print("model", model)
        self.feats_cond = feats_cond
        if model == "resnet":
            self.enc = PixelNeRFTimeEmbed(
                block=BasicBlockTimeEmbed,
                layers=[3, 4, 6, 3],
                use_first_pool=use_first_pool,
                use_viewdir=use_viewdir,
            )
        # elif model == "vit":
        #     backbone = self.backbone
        #     self.enc = DPTDepthModel(path=path, backbone=backbone, non_negative=True,)
        #     self.conv_map = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        elif model == "dit":
            self.enc = DiT(
                input_size=(image_size // 2, image_size // 2),
                depth=28,
                hidden_size=1152,
                patch_size=2,
                num_heads=16,
                in_channels=64,
                out_channels=512,
                cond_feats_dim=(74 if self.self_condition else 71)
                if feats_cond
                else 4,  #  self.n_feats_out + 7,  #   128 if feats_cond else 4,
                use_high_res=use_high_res_feats,
                use_abs_pose=use_abs_pose
            )
        self.mode = mode

        self.near = near
        self.far = far
        # self.pixelNeRF = PixelAlignedRadianceFieldTimeEmbed(n_feats=512)
        if not feats_cond:
            n_feats_out = 0
            self.n_feats_out = n_feats_out

        self.pixelNeRF_joint  = PixelAlignedRadianceFieldTimeEmbed(
            n_feats=512, n_feats_out=n_feats_out, use_viewdir=use_viewdir
        )

        self.pixelNeRF_joint_coarse = PixelAlignedRadianceFieldTimeEmbed(
            n_feats=512, n_feats_out=n_feats_out, use_viewdir=use_viewdir
        )
        
        self.renderer = VolumeRenderer(
            near=near,
            far=far,
            n_samples=n_coarse,
            n_fine_samples=n_fine,
            backgrd_color=background_color,
            use_viewdir=use_viewdir,
            lindisp=render_settings["lindisp"],
        ).cuda()

        # self.renderer_coarse = self.renderer
        self.renderer_coarse = VolumeRenderer(
            near=near,
            far=far,
            n_samples=n_coarse_coarse,
            n_fine_samples=n_coarse_fine,
            backgrd_color=background_color,
            use_viewdir=use_viewdir,
            lindisp=render_settings["lindisp"],
        ).cuda()

        print("feats_cond", feats_cond)
        # if self.feats_cond:
        #     self.ctxt_feat_collector = NoVolumeRenderer(
        #         near=near, far=far, n_samples=64, n_fine_samples=0, n_feats=512,
        #     ).cuda()

        self.num_render_pixels = render_settings["num_pixels"]
        self.num_render_pixels_no_grad = (
            self.num_render_pixels
        )  # 48x48 pixels max on V100
        self.len_render = torch.sqrt(torch.tensor(self.num_render_pixels)).long()

        self.channels = channels
        self.out_dim = channels
        self.normalize = normalize_to_neg_one_to_one
        self.unnormalize = unnormalize_to_zero_to_one
        self.viz_type = viz_type

    def get_feats(self, ctxt_inp, t, abs_camera_poses=None):
        if self.model == "resnet":
            latent = self.enc(ctxt_inp, time_emb=t)
        elif self.model == "dit":
            latent = self.enc(ctxt_inp, t=t, abs_camera_poses=abs_camera_poses)
        return latent

    # @torch.no_grad()
    def render_full_in_patches(
        self,
        xy_pix,
        c2w,
        intrinsics,
        rf,
        b,
        num_target,
        h,
        w,
        render_coarse=False,
        top_down=False,
    ):
        intrinsics = repeat(intrinsics, "b h w -> (b t) h w", t=num_target)
        xy_pix = repeat(xy_pix, "b h w -> (b t) h w", t=num_target)
        all_c2w = rearrange(c2w, "b t h w -> (b t) h w")
        rendered_rgb = torch.zeros(
            b * num_target, 3, h * w, device=xy_pix.device, dtype=torch.float32
        )
        rendered_feats = torch.zeros(
            b * num_target,
            self.n_feats_out,
            h * w,
            device=xy_pix.device,
            dtype=torch.float32,
        )
        rendered_depth = torch.zeros(
            b * num_target, h * w, device=xy_pix.device, dtype=torch.float32,
        )
        num_render_pixels = min(self.num_render_pixels_no_grad, h * w)

        # no gradients when refining
        # with torch.set_grad_enabled(not self.cnn_refine):
        for start in range(0, h * w, num_render_pixels):
            # compute linear indices of length self.num_render_pixels
            end = min(start + num_render_pixels, h * w)
            indices = torch.arange(start, end)
            xy_pix_sampled = xy_pix[:, indices, :]
            if render_coarse:
                rgb, depth, misc = self.renderer_coarse(
                    all_c2w, intrinsics, xy_pix_sampled, rf,
                )
            else:
                rgb, depth, misc = self.renderer(
                    all_c2w, intrinsics, xy_pix_sampled, rf, top_down=top_down,
                )
            rgbfeats = rearrange(rgb, "b n c -> b c n",)
            feats = rgbfeats[:, 3:, :]
            rgb = rgbfeats[:, :3, :]
            rendered_rgb[:, :, indices] = rgb
            rendered_feats[:, :, indices] = feats
            rendered_depth[:, indices] = depth[..., 0]

        rendered_rgb_intermediate = rearrange(
            rendered_rgb, "b c (h w) -> b c h w", h=h, w=w
        )
        rendered_feats = rearrange(rendered_feats, "b c (h w) -> b c h w", h=h, w=w)
        rendered_depth = rearrange(rendered_depth, "b (h w) -> b h w", h=h, w=w)

        if self.cnn_refine:
            # print("refining")
            rgbfeats = torch.cat([rendered_rgb_intermediate, rendered_feats], dim=1)
            rendered_rgb = self.cnn_refine_model(rgbfeats)
        else:
            rendered_rgb = rendered_rgb_intermediate
        return rendered_rgb, rendered_depth, rendered_feats

    def collect_ctxt_feats(self, xy_pix, c2w, intrinsics, rf, b, num_target, h, w):
        intrinsics = repeat(intrinsics, "b h w -> (b t) h w", t=num_target)
        xy_pix = repeat(xy_pix, "b h w -> (b t) h w", t=num_target)
        all_c2w = rearrange(c2w, "b t h w -> (b t) h w")
        feats = self.ctxt_feat_collector(all_c2w, intrinsics, xy_pix, rf,)
        feats = rearrange(feats, "b (h w) c -> b c h w", h=h, w=w)
        return feats

    def render_ctxt_from_trgt_cam(
        self, ctxt_rgb, intrinsics, xy_pix, ctxt_c2w, trgt_c2w, render_cond=True,
        ctxt_abs_camera_poses=None
    ):
        # render_cond = True
        # if cond_drop_prob > 0.0:
        #     # set render_cond to False with probability cond_drop_prob
        #     render_cond = torch.rand(1).item() > cond_drop_prob
        b, num_context, c, h, w = ctxt_rgb.shape
        ctxt_rgb = rearrange(ctxt_rgb, "b t h w c -> (b t) h w c")
        t = torch.zeros((b, num_context), device=ctxt_rgb.device, dtype=torch.long)

        t_resnet = rearrange(t, "b t -> (b t)")
        ctxt_inp = ctxt_rgb
        clean_ctxt_feats = self.get_feats(ctxt_inp, t_resnet, abs_camera_poses=ctxt_abs_camera_poses)
        clean_ctxt_feats = rearrange(
            clean_ctxt_feats, "(b t) c h w -> b t c h w", t=num_context,
        )
        if self.mode == "nocond":
            ctxt_rgbd = None
            trgt_rgbdfeats = None
        elif render_cond:
            ctxt_rgbd = None
            # if not self.feats_cond:
            "render from the target viewpoint to be used for conditioning"
            rf_ctxt = self.radiance_field_joint_coarse(
                all_feats=clean_ctxt_feats,
                all_c2w=ctxt_c2w,
                intrinsics=intrinsics,
                time_embed=t,
                return_mlp_input=False,
            )

            # concat ctxt and trgt c2w
            # c2w = torch.cat([ctxt_c2w, trgt_c2w[:, :1, ...]], dim=1,)
            # render from the target viewpoint to be used for conditioning
            rgb, depth, rendered_feats = self.render_full_in_patches(
                xy_pix=xy_pix,
                c2w=trgt_c2w[:, :1],
                intrinsics=intrinsics,
                rf=rf_ctxt,
                num_target=1,
                b=b,
                h=h,
                w=w,
                render_coarse=True,
            )
            # print(f"rgb: {rgb.shape}, depth: {depth.shape}")
            # depth = depth.unsqueeze(1)
            rgb = rearrange(rgb, "(b t) c h w -> b t c h w", t=1)
            depth = rearrange(depth, "(b t) h w -> b t () h w", t=1)
            rendered_feats = rearrange(rendered_feats, "(b t) c h w -> b t c h w", t=1)

            rgb = self.normalize(rgb)
            rgbdfeats = torch.cat([rgb, depth, rendered_feats], dim=2)
            trgt_rgbdfeats = rgbdfeats[:, 0]  # (b, c, h, w)
        else:
            ctxt_rgbd = None
            trgt_rgbdfeats = torch.zeros(
                b, 4 + self.n_feats_out, h, w, device=ctxt_rgb.device
            )

        return ctxt_rgbd, trgt_rgbdfeats, clean_ctxt_feats  # context and target

    # @torch.no_grad()
    def render_full_image(
        self,
        clean_ctxt_feats,
        trgt_rgbd,
        xy_pix,
        intrinsics,
        ctxt_c2w,
        trgt_c2w,
        noisy_trgt_rgb,
        t=None,
        x_self_cond=None,
        render_coarse=False,
        guidance_scale=1.0,
        uncond_trgt_rgbd=None,
        uncond_clean_ctxt_feats=None,
        render_high_res=False,
        xy_pix_high_res=None,
        trgt_abs_camera_poses=None
    ):
        b, num_context, c, h, w = clean_ctxt_feats.shape
        b, _, h, w = noisy_trgt_rgb.shape
        num_target = trgt_c2w.shape[1]

        if self.mode == "nocond":
            trgt_inp = noisy_trgt_rgb
        else:
            if self.feats_cond:
                trgt_inp = torch.cat([noisy_trgt_rgb, trgt_rgbd[:, :]], dim=1)
            else:
                trgt_inp = torch.cat([noisy_trgt_rgb, trgt_rgbd[:, :4]], dim=1)

        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(trgt_inp[:, :3])
            trgt_inp = torch.cat([trgt_inp, x_self_cond], dim=1)
        noisy_trgt_feats = self.get_feats(trgt_inp, t, abs_camera_poses=trgt_abs_camera_poses[:, :1] if trgt_abs_camera_poses is not None else None)
        noisy_trgt_feats = rearrange(noisy_trgt_feats, "b c h w -> b () c h w")

        all_feats = torch.cat([clean_ctxt_feats, noisy_trgt_feats], dim=1)
        all_c2w = torch.cat([ctxt_c2w, trgt_c2w[:, :1, ...]], dim=1)
        clean_t = torch.zeros((b, num_context), device=t.device)
        all_time_embed = torch.cat(
            [clean_t, t.unsqueeze(1)], dim=1
        )  # all clean context frames have time embedding 0; ensure target is the last frame
        rf = self.radiance_field_joint(
            all_feats, all_c2w, intrinsics, time_embed=all_time_embed,
        )

        rendered_rgb, rendered_depth, rendered_feats = self.render_full_in_patches(
            xy_pix=xy_pix if not render_high_res else xy_pix_high_res,
            c2w=trgt_c2w,
            intrinsics=intrinsics,
            rf=rf,
            num_target=num_target,
            b=b,
            h=h if not render_high_res else self.image_size,
            w=w if not render_high_res else self.image_size,
            render_coarse=render_coarse,
        )

        if guidance_scale > 1.0:
            if self.mode == "nocond":
                trgt_inp = noisy_trgt_rgb
            else:
                if self.feats_cond:
                    trgt_inp = torch.cat(
                        [noisy_trgt_rgb, uncond_trgt_rgbd[:, :]], dim=1
                    )
                else:
                    trgt_inp = torch.cat(
                        [noisy_trgt_rgb, uncond_trgt_rgbd[:, :4]], dim=1
                    )
            if self.self_condition:
                if x_self_cond is None:
                    x_self_cond = torch.zeros_like(trgt_inp[:, :3])
                trgt_inp = torch.cat([trgt_inp, x_self_cond], dim=1)
            noisy_trgt_feats = self.get_feats(trgt_inp, t, abs_camera_poses=trgt_abs_camera_poses[:, :1] if trgt_abs_camera_poses is not None else None)
            noisy_trgt_feats = rearrange(noisy_trgt_feats, "b c h w -> b () c h w")

            all_feats = torch.cat([uncond_clean_ctxt_feats, noisy_trgt_feats], dim=1)
            all_c2w = torch.cat([ctxt_c2w, trgt_c2w[:, :1, ...]], dim=1)
            clean_t = torch.zeros((b, num_context), device=t.device)
            all_time_embed = torch.cat(
                [clean_t, t.unsqueeze(1)], dim=1
            )  # all clean context frames have time embedding 0; ensure target is the last frame
            rf = self.radiance_field_joint(
                all_feats, all_c2w, intrinsics, time_embed=all_time_embed,
            )
            (
                uncond_rendered_rgb,
                uncond_rendered_depth,
                uncond_rendered_feats,
            ) = self.render_full_in_patches(
                xy_pix=xy_pix if not render_high_res else xy_pix_high_res,
                c2w=trgt_c2w,
                intrinsics=intrinsics,
                rf=rf,
                num_target=num_target,
                b=b,
                h=h if not render_high_res else self.image_size,
                w=w if not render_high_res else self.image_size,
                render_coarse=render_coarse,
            )
        else:
            uncond_rendered_rgb = None
            uncond_rendered_depth = None

        return (
            self.normalize(rendered_rgb),
            rendered_depth,
            self.normalize(uncond_rendered_rgb)
            if uncond_rendered_rgb is not None
            else None,
            uncond_rendered_depth,
        )

    # def render_top_down(self, clean_ctxt_feats, trgt_rgbd, xy_pix, intrinsics):
    def render_deterministic(
        self,
        model_input,
        n,
        x_self_cond=None,
        top_down=False,
        num_videos=None,
        render_high_res=False,
        ctxt_abs_camera_poses=None
    ):
        if "render_poses" not in model_input.keys():
            render_poses = self.compute_poses(self.viz_type, model_input, n,)
            print("using computed poses")
        else:
            render_poses = model_input["render_poses"][0]
            n = len(render_poses)
            print("using provided poses", render_poses.shape)
        intrinsics = model_input["intrinsics"]
        xy_pix = model_input["x_pix"]
        if num_videos is None:
            num_videos = xy_pix.shape[0]

        b, num_context, h, w, c = model_input["ctxt_rgb"].shape
        ctxt_rgb = rearrange(
            model_input["ctxt_rgb"][:num_videos], "b t h w c -> (b t) h w c"
        )
        t = torch.zeros((b, num_context), device=ctxt_rgb.device, dtype=torch.long)
        t_resnet = rearrange(t, "b t -> (b t)")
        ctxt_inp = ctxt_rgb

        clean_ctxt_feats = self.get_feats(ctxt_inp, t_resnet, abs_camera_poses=ctxt_abs_camera_poses)
        clean_ctxt_feats = rearrange(
            clean_ctxt_feats,
            "(b t) c h w -> b t c h w",
            t=model_input["ctxt_rgb"].shape[1],
        )
        rf_ctxt = self.radiance_field_joint_coarse(
            all_feats=clean_ctxt_feats,
            all_c2w=model_input["ctxt_c2w"][:num_videos],
            intrinsics=intrinsics,
            time_embed=t,
            return_mlp_input=False,
        )

        w, h = model_input["trgt_rgb"].shape[-1], model_input["trgt_rgb"].shape[-2]
        frames = []
        depth_frames = []
        xy_pix = (
            model_input["x_pix"] if not render_high_res else model_input["x_pix_128"]
        )
        h = 128 if render_high_res else h
        w = 128 if render_high_res else w
        print(f"render_poses {render_poses.shape}, {xy_pix.shape}")
        for i in range(n):
            rgb, depth, feats = self.render_full_in_patches(
                xy_pix[:num_videos],
                render_poses[i : i + 1][:, None, ...].cuda(),
                intrinsics[:num_videos],
                rf_ctxt,
                b=num_videos,
                num_target=1,
                h=h,
                w=w,
                top_down=top_down,
            )
            rgb = rearrange(rgb, "b c h w -> b h w c")
            rgb = rgb * 255.0
            frames.append(rgb.float().cpu().detach().numpy().astype(np.uint8))
            depth_frames.append(depth.float().cpu().detach())
        print(f"frames {len(frames)}")
        return frames, depth_frames, render_poses

    @torch.no_grad()
    def render_video(
        self,
        model_input,
        t,
        n,
        x_self_cond=None,
        top_down=False,
        num_videos=None,
        render_high_res=False,
    ):
        if "render_poses" not in model_input.keys():
            render_poses = self.compute_poses(self.viz_type, model_input, n,)
            print("using computed poses")
        else:
            render_poses = model_input["render_poses"][0]
            n = len(render_poses)
            print("using provided poses", render_poses.shape)
        intrinsics = model_input["intrinsics"]
        xy_pix = model_input["x_pix"]

        if num_videos is None:
            num_videos = xy_pix.shape[0]
        rf, ctxt_rgbd, trgt_rgbdfeats = self.prepare_input(
            ctxt_rgb=model_input["ctxt_rgb"][:num_videos],
            noisy_trgt_rgb=model_input["noisy_trgt_rgb"][:num_videos],
            intrinsics=intrinsics[:num_videos],
            xy_pix=xy_pix[:num_videos],
            ctxt_c2w=model_input["ctxt_c2w"][:num_videos],
            trgt_c2w=model_input["trgt_c2w"][:num_videos],
            t=t[:num_videos],
            ctxt_abs_camera_poses=model_input["ctxt_abs_camera_poses"][:num_videos],
            trgt_abs_camera_poses=model_input["trgt_abs_camera_poses"][:num_videos]
        )
        w, h = model_input["ctxt_rgb"].shape[-1], model_input["ctxt_rgb"].shape[-2]
        frames = []
        depth_frames = []

        xy_pix = (
            model_input["x_pix"] if not render_high_res else model_input["x_pix_128"]
        )
        h = 128 if render_high_res else h
        w = 128 if render_high_res else w
        print(f"render_poses {render_poses.shape}, {xy_pix.shape}")
        for i in range(n):
            # if i % 10 == 0:
            #     print(f"Rendering frame {i}/{n}", flush=True)
            # print(i)
            rgb, depth, feats = self.render_full_in_patches(
                xy_pix[:num_videos],
                render_poses[i : i + 1][:, None, ...].cuda(),
                # render_poses[0][i : i + 1][:, None, ...].cuda(),
                intrinsics[:num_videos],
                rf,
                b=num_videos,
                num_target=1,
                h=h,
                w=w,
                top_down=top_down,
            )
            rgb = rearrange(rgb, "b c h w -> b h w c")
            rgb = rgb * 255.0
            frames.append(rgb.float().cpu().detach().numpy().astype(np.uint8))
            depth_frames.append(depth.float().cpu().detach())
        print(f"frames {len(frames)}")
        return frames, depth_frames, render_poses

    def prepare_input(
        self,
        ctxt_rgb,
        noisy_trgt_rgb,
        intrinsics,
        xy_pix,
        ctxt_c2w,
        trgt_c2w,
        t,
        render_cond=True,
        x_self_cond=None,
        ctxt_abs_camera_poses=None,
        trgt_abs_camera_poses=None
    ):
        b, num_context, c, h, w = ctxt_rgb.shape

        ctxt_rgbd, trgt_rgbdfeats, ctxt_feats = self.render_ctxt_from_trgt_cam(
            ctxt_rgb,
            intrinsics,
            xy_pix,
            ctxt_c2w=ctxt_c2w,
            trgt_c2w=trgt_c2w,
            render_cond=True,
            ctxt_abs_camera_poses=ctxt_abs_camera_poses
        )
        clean_ctxt_feats = ctxt_feats

        if self.mode == "nocond":
            trgt_inp = noisy_trgt_rgb
        else:
            if not self.feats_cond:
                trgt_inp = torch.cat([noisy_trgt_rgb, trgt_rgbdfeats[:, :4]], dim=1)
            else:
                trgt_inp = torch.cat([noisy_trgt_rgb, trgt_rgbdfeats[:, :]], dim=1)

        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(trgt_inp[:, :3])

            trgt_inp = torch.cat([trgt_inp, x_self_cond], dim=1)
        # ctxt_inp = torch.cat([ctxt_rgb, ctxt_rgbd], dim=1)

        # noisy_trgt_feats = self.enc(trgt_inp, time_emb=t)
        noisy_trgt_feats = self.get_feats(trgt_inp, t, abs_camera_poses=trgt_abs_camera_poses[:, :1])
        noisy_trgt_feats = rearrange(noisy_trgt_feats, "b c h w -> b () c h w")

        all_feats = torch.cat([clean_ctxt_feats, noisy_trgt_feats], dim=1)
        all_c2w = torch.cat([ctxt_c2w, trgt_c2w[:, :1, ...]], dim=1)

        clean_t = torch.zeros((b, num_context), device=t.device)
        all_time_embed = torch.cat(
            [clean_t, t.unsqueeze(1)], dim=1
        )  # all clean context frames have time embedding 0; ensure target is the last frame

        rf = self.radiance_field_joint(
            all_feats=all_feats,
            all_c2w=all_c2w,
            intrinsics=intrinsics,
            time_embed=all_time_embed,
        )
        return rf, ctxt_rgbd, trgt_rgbdfeats

    def forward(self, model_input, t, x_self_cond=None, render_cond=True):
        intrinsics = model_input["intrinsics"]
        xy_pix = model_input["x_pix"]

        with torch.set_grad_enabled(not self.cnn_refine):
            rf, ctxt_rgbd, trgt_rgbdfeats = self.prepare_input(
                ctxt_rgb=model_input["ctxt_rgb"],
                noisy_trgt_rgb=model_input["noisy_trgt_rgb"],
                intrinsics=intrinsics,
                xy_pix=xy_pix,
                ctxt_c2w=model_input["ctxt_c2w"],
                trgt_c2w=model_input["trgt_c2w"],
                t=t,
                render_cond=render_cond,
                x_self_cond=x_self_cond,
                ctxt_abs_camera_poses=model_input["ctxt_abs_camera_poses"],
                trgt_abs_camera_poses=model_input["trgt_abs_camera_poses"]
            )

            b, num_target, c, h, w = model_input["trgt_rgb"].shape

            intrinsics = repeat(intrinsics, "b h w -> (b t) h w", t=num_target)
            xy_pix = repeat(xy_pix, "b n c -> b t n c", t=num_target)
            trgt_c2w = rearrange(model_input["trgt_c2w"], "b t h w -> (b t) h w")

            # sample points for rendering
            # if "trgt_masks" not in model_input.keys():
            if self.sampling == "random":
                random_indices = torch.randint(0, h * w, (self.num_render_pixels,))
                new_xy = xy_pix[:, :, random_indices, :]

                model_input["trgt_rgb_sampled"] = rearrange(
                    model_input["trgt_rgb"], "b t c h w -> b t c (h w)",
                )[..., random_indices]
            else:
                starth = np.random.randint(h - self.len_render + 1, size=b)
                startw = np.random.randint(w - self.len_render + 1, size=b)

                new_xy = torch.zeros(
                    (b, num_target, self.len_render, self.len_render, 2),
                    dtype=torch.float32,
                    device=xy_pix.device,
                )

                xy_pix = rearrange(xy_pix, "b t (h w) c -> b t h w c", h=h, w=w)
                model_input["trgt_rgb_sampled"] = torch.zeros(
                    (b, num_target, self.channels, self.len_render, self.len_render),
                    dtype=torch.float32,
                    device=xy_pix.device,
                )
                for b1 in range(b):
                    new_xy[b1] = xy_pix[
                        b1,
                        :,
                        starth[b1] : starth[b1] + self.len_render,
                        startw[b1] : startw[b1] + self.len_render,
                        :,
                    ]
                    model_input["trgt_rgb_sampled"][b1] = model_input["trgt_rgb"][b1][
                        ...,
                        starth[b1] : starth[b1] + self.len_render,
                        startw[b1] : startw[b1] + self.len_render,
                    ]
                model_input["trgt_rgb_sampled"] = rearrange(
                    model_input["trgt_rgb_sampled"], "b t c h w -> b t c (h w)",
                )
                new_xy = rearrange(new_xy, "b t h w c -> b t (h w) c")

            new_xy = rearrange(new_xy, "b t n c -> (b t) n c")
            rgbfeats, depth, misc = self.renderer(trgt_c2w, intrinsics, new_xy, rf)
            rgbfeats = rearrange(
                rgbfeats, "b (h w) c -> b c h w", h=self.len_render, w=self.len_render
            )
            rgb = rgbfeats[:, :3, ...]
            rgb = rearrange(rgb, "b c h w -> b c (h w)")

        if self.cnn_refine:
            rgb_intermediate = rgb
            rgb_refine = self.cnn_refine_model(rgbfeats)
            rgb_refine = rearrange(rgb_refine, "b c h w -> b c (h w)")
        else:
            rgb_refine = rgb
            rgb_intermediate = None

        misc["rendered_ctxt_rgb"] = (
            None if ctxt_rgbd is None else ctxt_rgbd[:, :3, :, :]
        )
        misc["rgb_intermediate"] = (
            None if rgb_intermediate is None else self.normalize(rgb_intermediate)
        )
        misc["rendered_ctxt_depth"] = (
            None
            if (ctxt_rgbd is None or ctxt_rgbd.shape[1] == 3)
            else ctxt_rgbd[:, 3, :, :].unsqueeze(1)
        )
        misc["rendered_trgt_rgb"] = (
            None if trgt_rgbdfeats is None else trgt_rgbdfeats[:, :3, :, :]
        )
        misc["rendered_trgt_depth"] = (
            None
            if (trgt_rgbdfeats is None or trgt_rgbdfeats.shape[1] == 3)
            else trgt_rgbdfeats[:, 3:4, :, :].unsqueeze(1)
        )
        misc["rendered_trgt_feats"] = (
            None
            if (trgt_rgbdfeats is None or trgt_rgbdfeats.shape[1] == 3)
            else trgt_rgbdfeats[:, 4:, :, :]
        )

        return self.normalize(rgb_refine), depth, misc

    @typechecked
    def radiance_field_joint(
        self, all_feats, all_c2w, intrinsics, time_embed=None, return_mlp_input=False
    ) -> Callable:
        # time_embed = None
        return lambda x, v, fine: self.pixelNeRF_joint(
            xyz=x,
            viewdir=v,
            feats=all_feats,
            c2w=all_c2w,
            intrinsics=intrinsics,
            t=time_embed,
            return_mlp_input=return_mlp_input,
            fine=fine,
        )

    @typechecked
    def radiance_field_joint_coarse(
        self, all_feats, all_c2w, intrinsics, time_embed=None, return_mlp_input=False
    ) -> Callable:
        # time_embed = None
        return lambda x, v, fine: self.pixelNeRF_joint_coarse(
            xyz=x,
            viewdir=v,
            feats=all_feats,
            c2w=all_c2w,
            intrinsics=intrinsics,
            t=time_embed,
            return_mlp_input=return_mlp_input,
            fine=fine,
        )

    @torch.no_grad()
    def compute_poses(
        self, type, model_input, n, radius=None, max_angle=None, canonical=False
    ):
        if type == "spherical":
            if radius is None:
                radius = (self.near + self.far) * 0.5
            if max_angle is None:
                max_angle = 60

            render_poses = []
            for angle in np.linspace(0, max_angle, n + 1)[:-1]:
                pose = pose_spherical(0, -angle, radius).cpu()
                # pose = pose_spherical(angle, 0, radius).cpu()
                if canonical:
                    pose = torch.einsum(
                        "ij, jk -> ik", model_input["inv_ctxt_c2w"][0, 0].cpu(), pose,
                    )
                else:
                    pose[2, -1] += radius
                render_poses.append(pose)
            render_poses = torch.stack(render_poses, 0)
        elif type == "interpolation":
            render_poses = torch.stack(
                [
                    interpolate_pose_wobble(
                        model_input["ctxt_c2w"][0][0],
                        model_input["trgt_c2w"][0][0],
                        t / n,
                        wobble=False,
                    )
                    for t in range(n)
                ],
                0,
            )
        else:
            raise ValueError("Unknown video type", type)
        print(f"render_poses: {render_poses.shape}")
        return render_poses


# define CNN refinement network
class RefineOut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.skip = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)

        self.skip0 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)

        self.conv0 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.torgb = nn.Conv2d(256, 3, kernel_size=1)
        # initialize torgb weights to 0
        self.torgb.weight.data.zero_()
        self.torgb.bias.data.zero_()

        self.relu = nn.ReLU()

    def forward(self, x):
        skip = self.skip(x)
        skip0 = self.skip0(x)
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(torch.cat([x, skip0], dim=1)))
        x = self.relu(self.conv4(x))

        x = self.torgb(x)
        rgb = x + skip  # torch.sigmoid(x)
        return rgb
