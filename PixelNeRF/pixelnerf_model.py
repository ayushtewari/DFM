import torch, torchvision
from torch import nn
import torch.nn.functional as F
import sys
import os
import numpy as np
from einops import rearrange, repeat

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers import *
from utils import *
from PixelNeRF.pixelnerf_helpers import *
from PixelNeRF.resnet import PixelNeRFTimeEmbed, BasicBlockTimeEmbed
# from PixelNeRF.transformer.midas.dpt_depth import DPTDepthModel
from PixelNeRF.transformer.DiT import DiT


class PixelNeRFModel(nn.Module):
    def __init__(
        self, near, far, viz_type="spherical",
    ):
        super().__init__()
        self.near = near
        self.far = far
        self.self_condition = False
        self.normalize = normalize_to_neg_one_to_one
        self.viz_type = viz_type

    @torch.no_grad()
    def render_full_in_patches(self, xy_pix, c2w, intrinsics, rf, b, num_target, h, w):
        intrinsics = repeat(intrinsics, "b h w -> (b t) h w", t=num_target)
        xy_pix = repeat(xy_pix, "b h w -> (b t) h w", t=num_target)
        all_c2w = rearrange(c2w, "b t h w -> (b t) h w")
        rendered_rgb = torch.zeros(
            b * num_target, 3, h * w, device=xy_pix.device, dtype=torch.float32
        )
        rendered_depth = torch.zeros(
            b * num_target, h * w, device=xy_pix.device, dtype=torch.float32,
        )
        for start in range(0, h * w, self.num_render_pixels):
            # compute linear indices of length self.num_render_pixels
            end = min(start + self.num_render_pixels, h * w)
            indices = torch.arange(start, end)
            xy_pix_sampled = xy_pix[:, indices, :]
            rgb, depth, misc = self.renderer(all_c2w, intrinsics, xy_pix_sampled, rf,)
            rgb = rearrange(rgb, "b n c -> b c n",)

            rendered_rgb = rendered_rgb.to(rgb.dtype)
            rendered_depth = rendered_depth.to(depth.dtype)

            rendered_rgb[:, :, indices] = rgb
            rendered_depth[:, indices] = depth[..., 0]
        rendered_rgb = rearrange(rendered_rgb, "b c (h w) -> b c h w", h=h, w=w)
        rendered_depth = rearrange(rendered_depth, "b (h w) -> b h w", h=h, w=w)
        return rendered_rgb, rendered_depth

    @torch.no_grad()
    def render_full_image(self, model_input, t=None):
        b, num_target, c, h, w = model_input["trgt_rgb"].shape
        b, num_context, c, h, w = model_input["ctxt_rgb"].shape
        xy_pix = model_input["x_pix"]  # [B, H * W, 2]
        intrinsics = model_input["intrinsics"]

        # clean_ctxt_feats = self.enc(model_input["ctxt_rgb"], time_emb=t)  # , t=t * 0)
        ctxt_inp = rearrange(model_input["ctxt_rgb"], "b t c h w -> (b t) c h w")
        ctxt_feats = self.get_feats(ctxt_inp, t, abs_camera_poses=model_input["ctxt_abs_camera_poses"])
        ctxt_feats = rearrange(ctxt_feats, "(b t) c h w -> b t c h w", t=num_context,)

        rf = self.radiance_field_cond(
            ctxt_feats, model_input["ctxt_c2w"], intrinsics, time_embed=t,
        )

        # rf = self.radiance_field_cond(
        #     clean_ctxt_feats, None, model_input["ctxt_c2w"], None, intrinsics,
        # )
        num_target = model_input["trgt_c2w"].shape[1]
        rendered_rgb, rendered_depth = self.render_full_in_patches(
            xy_pix, model_input["trgt_c2w"], intrinsics, rf, b, num_target, h, w,
        )
        return rendered_rgb, rendered_depth

    @torch.no_grad()
    def render_video(self, model_input, n, t=None):
        # only renders the first sample
        b, num_context, c, h, w = model_input["ctxt_rgb"].shape
        # print("num_context", num_context)
        # render_poses = self.compute_poses(self.viz_type, model_input, n,)
        if "render_poses" not in model_input.keys():
            render_poses = self.compute_poses(self.viz_type, model_input, n,)
            print("using computed poses")
        else:
            render_poses = model_input["render_poses"][0]
            n = len(render_poses)
            print("using provided poses", len(render_poses))

        intrinsics = model_input["intrinsics"]
        xy_pix = model_input["x_pix"]
        frames = []
        ctxt_inp = rearrange(model_input["ctxt_rgb"], "b t c h w -> (b t) c h w")
        ctxt_feats = self.get_feats(ctxt_inp, t, abs_camera_poses=model_input["ctxt_abs_camera_poses"])
        ctxt_feats = rearrange(ctxt_feats, "(b t) c h w -> b t c h w", t=num_context,)
        # second_frame = None if num_context == 1 else ctxt_feats[:1, 1]
        # second_c2w = None if num_context == 1 else model_input["ctxt_c2w"][:1, 1]

        cond = self.radiance_field_cond(
            ctxt_feats[:1],
            model_input["ctxt_c2w"][:1],
            intrinsics[:1],
            time_embed=t[:1],
        )
        rf = cond

        for i in range(n):
            if i % 10 == 0:
                print(f"Rendering frame {i}/{n}")
            rgb, depth = self.render_full_in_patches(
                xy_pix[:1],
                render_poses[i : i + 1][:, None, ...].cuda(),
                intrinsics[:1],
                rf,
                1,
                1,
                model_input["trgt_rgb"].shape[-2],
                model_input["trgt_rgb"].shape[-1],
            )
            rgb = rearrange(rgb, "b c h w -> b h w c")
            rgb = rgb * 255.0
            frames.append(rgb.float().cpu().detach().numpy()[0].astype(np.uint8))
        return frames

    @torch.no_grad()
    def compute_poses(self, type, model_input, n):
        if type == "spherical":
            radius = (self.near + self.far) * 0.5
            render_poses = torch.stack(
                [
                    torch.einsum(
                        "ij, jk -> ik",
                        model_input["ctxt_c2w"][0][0].cpu(),
                        pose_spherical(angle, -0.0, radius).cpu(),
                    )
                    for angle in np.linspace(-180, 180, n + 1)[:-1]
                ],
                0,
            )
        elif type == "interpolation":
            num_context = model_input["ctxt_c2w"].shape[1]
            trgt_c2w = (
                model_input["trgt_c2w"][0][0]
                if num_context == 1
                else model_input["ctxt_c2w"][0][-1]
            )
            render_poses = torch.stack(
                [
                    interpolate_pose(model_input["ctxt_c2w"][0][0], trgt_c2w, t / n,)
                    for t in range(n)
                ],
                0,
            )
        else:
            raise ValueError("Unknown video type", type)
        return render_poses


class PixelNeRFModelVanilla(PixelNeRFModel):
    def __init__(
        self,
        near,
        far,
        model="vit",  # resnet, vit
        backbone="vitb_rn50_384",
        viz_type="spherical",
        channels=3,
        background_color=torch.ones((3,), dtype=torch.float32),
        use_first_pool=True,
        lindisp=False,
        path=None,
        use_abs_pose=False,
        use_high_res_feats=False
    ):
        super().__init__(near, far, viz_type)
        # self.enc = PixelNeRFEncoderOriginal(use_first_pool=use_first_pool)
        # self.pixelNeRF = PixelAlignedRadianceField(n_feats=512)
        self.pixelNeRF = PixelAlignedRadianceFieldTimeEmbed(
            n_feats=512, n_feats_out=0, use_viewdir=False
        )
        self.model = model
        self.backbone = backbone
        if model == "resnet":
            self.enc = PixelNeRFTimeEmbed(
                block=BasicBlockTimeEmbed,
                layers=[3, 4, 6, 3],
                use_first_pool=use_first_pool,
            )
        elif model == "vit":
            backbone = self.backbone
            self.enc = DPTDepthModel(path=path, backbone=backbone, non_negative=True,)
            self.conv_map = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        elif model == "dit":
            self.enc = DiT(
                depth=28,
                hidden_size=1152,
                patch_size=2,
                num_heads=16,
                in_channels=64,
                out_channels=512,
                use_high_res=use_high_res_feats,
                use_abs_pose=use_abs_pose
            )
        self.near = near
        self.far = far
        self.renderer = VolumeRenderer(
            near=near,
            far=far,
            n_samples=64,
            n_fine_samples=64,
            backgrd_color=background_color,
            lindisp=lindisp,
        ).cuda()
        self.len_render = 20
        self.num_render_pixels = self.len_render ** 2  # 36 x36 pixels
        self.channels = channels
        self.out_dim = channels
        self.sampling = "patch"  # "patch" or "random"

    def get_feats(self, ctxt_inp, t, abs_camera_poses=None):
        if self.model == "resnet":
            t = rearrange(t, "b t -> (b t)")
            # print(f"t.shape: {t.shape}, ctxt_inp.shape: {ctxt_inp.shape}")
            latent = self.enc(ctxt_inp, time_emb=t)
        elif self.model == "vit":
            enc = self.enc(ctxt_inp)
            z_conv = self.conv_map(ctxt_inp)
            latents = enc + [z_conv]
            latent_sz = latents[-1].shape[-2:]
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i], tuple(latent_sz), mode="bilinear", align_corners=True,
                )
            latent = torch.cat(latents, dim=1)
        elif self.model == "dit":
            latent = self.enc(ctxt_inp, abs_camera_poses=abs_camera_poses)

        return latent

    def forward(self, model_input, t=None, add_noise=False, x_cond=False):
        b, num_target, c, h, w = model_input["trgt_rgb"].shape
        b, num_context, c, h, w = model_input["ctxt_rgb"].shape
        # print("b, num_target, c, h, w", b, num_target, c, h, w)
        xy_pix = model_input["x_pix"]  # [B, H * W, 2]
        intrinsics = model_input["intrinsics"]
        # t = None
        ctxt_inp = rearrange(model_input["ctxt_rgb"], "b t c h w -> (b t) c h w")
        ctxt_feats = self.get_feats(ctxt_inp, t, abs_camera_poses=model_input["ctxt_abs_camera_poses"])
        ctxt_feats = rearrange(ctxt_feats, "(b t) c h w -> b t c h w", t=num_context,)

        # second_frame = None
        # second_c2w = None
        # if num_context > 1:
        #     second_frame = ctxt_feats[:, 1]
        #     second_c2w = model_input["ctxt_c2w"][:, 1]

        rf = self.radiance_field_cond(
            ctxt_feats, model_input["ctxt_c2w"], intrinsics, time_embed=t,
        )

        # sample points for rendering
        if self.sampling == "random":
            random_indices = torch.randint(0, h * w, (self.num_render_pixels,))
            xy_pix = xy_pix[:, random_indices, :]
            model_input["trgt_rgb_sampled"] = rearrange(
                model_input["trgt_rgb"], "b t c h w -> b t c (h w)",
            )[..., random_indices]
        else:
            starth = np.random.randint(h - self.len_render + 1, size=b)
            startw = np.random.randint(w - self.len_render + 1, size=b)
            new_xy = torch.zeros(
                (b, self.len_render, self.len_render, 2),
                dtype=torch.float32,
                device=xy_pix.device,
            )
            xy_pix = rearrange(xy_pix, "b (h w) c -> b h w c", h=h, w=w)
            model_input["trgt_rgb_sampled"] = torch.zeros(
                (b, num_target, self.channels, self.len_render, self.len_render),
                dtype=torch.float32,
                device=xy_pix.device,
            )

            for b1 in range(b):
                new_xy[b1] = xy_pix[
                    b1,
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
            xy_pix = rearrange(new_xy, "b h w c -> b (h w) c")

        intrinsics = repeat(intrinsics, "b h w -> (b t) h w", t=num_target)
        xy_pix = repeat(xy_pix, "b n c -> (b t) n c", t=num_target)
        all_c2w = rearrange(model_input["trgt_c2w"], "b t h w -> (b t) h w")
        rgb, depth, misc = self.renderer(all_c2w, intrinsics, xy_pix, rf,)
        rgb = rearrange(rgb, "b n c -> b c n",)
        rgb = rgb[:, :3, :]

        if self.sampling == "patch":
            misc["starth"] = starth
            misc["startw"] = startw
            misc["len_render"] = self.len_render

        return self.normalize(rgb), depth, misc

    @typechecked
    def radiance_field_cond(self, feats, c2w, intrinsics, time_embed=None,) -> Callable:
        return lambda x, v, fine: self.pixelNeRF(
            xyz=x,
            viewdir=v,
            feats=feats,
            c2w=c2w,
            intrinsics=intrinsics,
            t=time_embed,
            return_mlp_input=False,
            fine=fine,
        )
