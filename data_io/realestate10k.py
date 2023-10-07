from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import rearrange, repeat
from torch.utils.data import Dataset
from numpy.random import default_rng
from utils import *
from geometry import get_opencv_pixel_coordinates
from numpy import random
import scipy
import cv2
from PIL import Image
from numpy.random import default_rng

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


Stage = Literal["train", "test", "val"]

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[:4]
        fx = fx * (640.0 / 360.0)
        assert np.allclose(cx, 0.5)
        assert np.allclose(cy, 0.5)

        self.intrinsics = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32,
        )

        w2c_mat = np.array(entry[6:], dtype=np.float32).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4, dtype=np.float32)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

class RealEstate10kDatasetOM(Dataset):
    examples: List[Path]
    pose_file: Path
    stage: Stage
    to_tensor: tf.ToTensor
    overfit_to_index: Optional[int]
    num_target: int
    context_min_distance: int
    context_max_distance: int

    z_near: float = 0.7
    z_far: float = 10.0
    image_size: int = 64
    background_color: torch.tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    def __init__(
        self,
        root: Union[str, Path],
        # pose_file: Union[str, Path],
        num_context: int,
        num_target: int,
        context_min_distance: int,
        context_max_distance: int,
        stage: Stage = "train",
        overfit_to_index: Optional[int] = None,
        max_scenes: Optional[int] = None,
        pose_root: Optional[Union[str, Path]] = None,
        image_size: Optional[int] = 64,
    ) -> None:
        super().__init__()
        self.overfit_to_index = overfit_to_index
        self.num_context = num_context
        self.num_target = num_target
        self.context_min_distance = context_min_distance
        self.context_max_distance = context_max_distance
        self.image_size = image_size
        sub_dir = {"train": "train", "test": "test", "val": "test",}[stage]
        image_root = Path(root) / sub_dir
        scene_path_list = sorted(list(Path(image_root).glob("*/")))

        if max_scenes is not None:
            scene_path_list = scene_path_list[:max_scenes]
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        self.rng = default_rng()
        self.normalize = normalize_to_neg_one_to_one

        if pose_root is None:
            pose_root = root
        print("loading pose file")
        pose_file = Path(pose_root) / f"{stage}.mat"
        self.all_cam_params = scipy.io.loadmat(pose_file)

        dummy_img_path = str(next(scene_path_list[0].glob("*.jpg")))
        dummy_img = cv2.imread(dummy_img_path)
        h, w = dummy_img.shape[:2]

        assert w == 640 and h == 360

        self.border = 140
        self.xy_pix = get_opencv_pixel_coordinates(
            x_resolution=self.image_size, y_resolution=self.image_size
        )
        self.xy_pix_128 = get_opencv_pixel_coordinates(
            x_resolution=128, y_resolution=128
        )

        self.len = 0
        all_rgb_files = []
        all_timestamps = []
        self.scene_path_list = []
        for i, scene_path in enumerate(scene_path_list):
            rgb_files = sorted(scene_path.glob("*.jpg"))
            self.len += len(rgb_files)
            timestamps = [int(rgb_file.name.split(".")[0]) for rgb_file in rgb_files]
            sorted_ids = np.argsort(timestamps)
            all_rgb_files.append(np.array(rgb_files)[sorted_ids])
            self.scene_path_list.append(scene_path)
            all_timestamps.append(np.array(timestamps)[sorted_ids])
        self.all_rgb_files = np.concatenate(all_rgb_files)
        self.indices = torch.arange(0, len(self.scene_path_list))
        self.all_rgb_files = all_rgb_files
        self.all_timestamps = all_timestamps
        print("NUM IMAGES", self.len)

    # @lru_cache(maxsize=None)
    def read_image(self, rgb_files, id):
        rgb_file = rgb_files[id]
        # print(f"reading {rgb_file}")
        rgb = (
            torch.tensor(
                np.asarray(Image.open(rgb_file)).astype(np.float32)[
                    :, self.border : -self.border, :
                ]
            ).permute(2, 0, 1)
            / 255.0
        )
        # print(rgb.shape, "SHAPE")
        rgb = F.interpolate(
            rgb.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            antialias=True,
        )[0]

        cam_param = self.all_cam_params[str(rgb_file.parent.name)][id][1:]
        cam_param = Camera(cam_param.flatten().tolist())
        return rgb, cam_param

    def __len__(self) -> int:
        return len(self.all_rgb_files)

    def __getitem__(self, index: int):
        scene_idx = random.randint(0, len(self.all_rgb_files) - 1)
        # start from reverse
        # scene_idx = len(self.all_rgb_files) - index - 1
        if self.overfit_to_index is not None:
            scene_idx = self.overfit_to_index

        def fallback():
            """Used if the desired index can't be loaded."""
            return self[random.randint(0, len(self.all_rgb_files) - 1)]

        rgb_files = self.all_rgb_files[scene_idx]
        timestamps = self.all_timestamps[scene_idx]
        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)
        if num_frames < self.num_target + 1:
            return fallback()

        start_idx = self.rng.choice(len(rgb_files), 1)[0]
        # context_min_distance = self.context_min_distance  # * self.num_context
        # context_max_distance = self.context_max_distance  # * self.num_context

        # num_context = self.rng.choice(np.arange(1, self.num_context + 1), 1)[0]
        num_context = self.num_context
        context_min_distance = self.context_min_distance * num_context
        context_max_distance = self.context_max_distance * num_context

        end_idx = self.rng.choice(
            np.arange(
                start_idx + context_min_distance, start_idx + context_max_distance,
            ),
            1,
            replace=False,
        )[0]
        if end_idx >= len(rgb_files):
            return fallback()
        trgt_idx = self.rng.choice(
            np.arange(start_idx, end_idx), self.num_target, replace=False
        )

        flip = self.rng.choice([True, False])
        if flip:
            temp = start_idx
            start_idx = end_idx
            end_idx = temp

        ctxt_idx = [start_idx]
        trgt_idx[0] = end_idx
        if num_context != 1:
            distance = self.rng.choice(
                np.arange(context_min_distance, context_max_distance), 1,
            )
            if start_idx < end_idx:
                extra_ctxt_idx = self.rng.choice(
                    np.arange(
                        start_idx, max(start_idx + num_context - 1, end_idx - distance),
                    ),
                    num_context - 1,
                    replace=False,
                )
            else:
                extra_ctxt_idx = self.rng.choice(
                    np.arange(
                        min(start_idx - num_context + 1, end_idx + distance), start_idx,
                    ),
                    num_context - 1,
                    replace=False,
                )
            ctxt_idx.extend(extra_ctxt_idx)

        if flip:
            # sort the target indices increasingly
            trgt_idx = np.sort(trgt_idx)
        else:
            # sort the target indices decreasingly
            trgt_idx = np.sort(trgt_idx)[::-1]
        # if len(rgb_files) < trgt_idx.max() + 1:
        #     return fallback()

        trgt_rgbs = []
        trgt_c2w = []
        trgt_intrinsics = []
        for id in trgt_idx:
            rgb, cam_param = self.read_image(rgb_files, id)
            trgt_rgbs.append(rgb)
            trgt_intrinsics.append(cam_param.intrinsics)
            trgt_c2w.append(cam_param.c2w_mat)
        trgt_c2w = torch.tensor(np.array(trgt_c2w)).float()
        trgt_rgb = torch.stack(trgt_rgbs, axis=0)

        # load the ctxt
        ctxt_rgbs = []
        ctxt_c2w = []
        ctxt_intrinsics = []
        for id in ctxt_idx:
            rgb, cam_param = self.read_image(rgb_files, id)
            ctxt_rgbs.append(rgb)
            ctxt_intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
            ctxt_c2w.append(cam_param.c2w_mat)
        ctxt_c2w = torch.tensor(np.array(ctxt_c2w)).float()
        ctxt_rgb = torch.stack(ctxt_rgbs, axis=0)
        ctxt_intrinsics = torch.stack(ctxt_intrinsics, axis=0)

        # ctxt_rgb, ctxt_cam_param = self.read_image(rgb_files, ctxt_idx)
        # ctxt_c2w = ctxt_cam_param.c2w_mat
        # ctxt_c2w = torch.tensor(np.array(ctxt_c2w)).float()
        # ctxt_intrinsics = ctxt_cam_param.intrinsics
        # ctxt_intrinsics = torch.tensor(np.array(ctxt_intrinsics)).float()
        inv_ctxt_c2w = torch.inverse(ctxt_c2w[0])
        ctxt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(num_context, 1, 1)
        trgt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            self.num_target, 1, 1
        )

        return (
            {
                "ctxt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", ctxt_inv_ctxt_c2w_repeat, ctxt_c2w
                ),
                "trgt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", trgt_inv_ctxt_c2w_repeat, trgt_c2w
                ),
                "ctxt_rgb": self.normalize(ctxt_rgb),
                "trgt_rgb": self.normalize(trgt_rgb),
                "ctxt_abs_camera_poses": None,
                "intrinsics": ctxt_intrinsics[0],
                "x_pix": rearrange(self.xy_pix, "h w c -> (h w) c"),
                "idx": torch.tensor([index]),
                "image_shape": torch.tensor([self.image_size, self.image_size, 3]),
                "num_context": torch.tensor([num_context]),
            },
            trgt_rgb,  # rearrange(rendered["image"], "c h w -> (h w) c"),
        )

    def data_for_video(self, video_idx, ctxt_idx, trgt_idx, num_frames_render=20):
        scene_idx = video_idx
        rgb_files = self.all_rgb_files[scene_idx]
        # print(rgb_files, scene_idx)
        timestamps = self.all_timestamps[scene_idx]
        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)

        trgt_rgbs = []
        trgt_c2w = []
        trgt_intrinsics = []
        for id in trgt_idx:
            id = min(id, len(rgb_files) - 1)
            id = max(id, 0)
            rgb, cam_param = self.read_image(rgb_files, id)
            trgt_rgbs.append(rgb)
            trgt_intrinsics.append(cam_param.intrinsics)
            trgt_c2w.append(cam_param.c2w_mat)
        trgt_c2w = torch.tensor(np.array(trgt_c2w)).float()
        trgt_rgb = torch.stack(trgt_rgbs, axis=0)

        # load the ctxt
        ctxt_rgbs = []
        ctxt_c2w = []
        ctxt_intrinsics = []
        for id in ctxt_idx:
            id = min(id, len(rgb_files) - 1)
            id = max(id, 0)
            rgb, cam_param = self.read_image(rgb_files, id)
            ctxt_rgbs.append(rgb)
            ctxt_intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
            ctxt_c2w.append(cam_param.c2w_mat)
        ctxt_c2w = torch.tensor(np.array(ctxt_c2w)).float()
        ctxt_rgb = torch.stack(ctxt_rgbs, axis=0)
        ctxt_intrinsics = torch.stack(ctxt_intrinsics, axis=0)

        render_poses = []
        num_frames_render = min(ctxt_idx[0], len(rgb_files) - 1) - min(
            trgt_idx[0], len(rgb_files) - 1
        )
        noflip = False
        if num_frames_render < 0:
            noflip = True
            num_frames_render *= -1

        for i in range(1, num_frames_render + 1):
            # id = ctxt_idx[0] + i * (trgt_idx[0] - ctxt_idx[0]) // (num_frames_render)
            if noflip:
                id = ctxt_idx[0] + i
            else:
                id = trgt_idx[0] + i
            rgb_file = rgb_files[id]
            cam_param = self.all_cam_params[str(rgb_file.parent.name)][id][1:]
            cam_param = Camera(cam_param.flatten().tolist())
            render_poses.append(cam_param.c2w_mat)
        render_poses = torch.tensor(np.array(render_poses)).float()

        print(
            f"ctxt_idx: {ctxt_idx}, trgt_idx: {trgt_idx}, num_frames_render: {num_frames_render}, len(rgb_files): {len(rgb_files)}"
        )

        num_frames_render = render_poses.shape[0]
        inv_ctxt_c2w = torch.inverse(ctxt_c2w[0])
        ctxt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            self.num_context, 1, 1
        )
        trgt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            self.num_target, 1, 1
        )
        trgt_inv_ctxt_c2w_repeat_video = inv_ctxt_c2w.unsqueeze(0).repeat(
            num_frames_render, 1, 1
        )
        return (
            {
                "ctxt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", ctxt_inv_ctxt_c2w_repeat, ctxt_c2w
                ),
                "trgt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", trgt_inv_ctxt_c2w_repeat, trgt_c2w
                ),
                # "render_poses": torch.einsum(
                #     "ijk, ikl -> ijl", trgt_inv_ctxt_c2w_repeat_video, render_poses
                # ),
                "ctxt_rgb": self.normalize(ctxt_rgb),
                "trgt_rgb": self.normalize(trgt_rgb),
                "ctxt_abs_camera_poses": ctxt_c2w,
                "trgt_abs_camera_poses": trgt_c2w,
                "intrinsics": ctxt_intrinsics[0],
                "x_pix": rearrange(self.xy_pix, "h w c -> (h w) c"),
                "x_pix_128": rearrange(self.xy_pix_128, "h w c -> (h w) c"),
                # "idx": torch.tensor([index]),
                "image_shape": torch.tensor([self.image_size, self.image_size, 3]),
                # "folder_path": str(rgb_files[0].parent),
            },
            trgt_rgb,  # rearrange(rendered["image"], "c h w -> (h w) c"),
        )

    def data_for_video_GT(self, video_idx, ctxt_idx, trgt_idx, num_frames_render=20):
        scene_idx = video_idx
        rgb_files = self.all_rgb_files[scene_idx]
        timestamps = self.all_timestamps[scene_idx]
        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)

        trgt_rgbs = []
        trgt_c2w = []
        trgt_intrinsics = []
        for id in trgt_idx:
            id = min(id, len(rgb_files) - 1)
            rgb, cam_param = self.read_image(rgb_files, id)
            trgt_rgbs.append(rgb)
            trgt_intrinsics.append(cam_param.intrinsics)
            trgt_c2w.append(cam_param.c2w_mat)
        trgt_c2w = torch.tensor(np.array(trgt_c2w)).float()
        trgt_rgb = torch.stack(trgt_rgbs, axis=0)

        # load the ctxt
        ctxt_rgbs = []
        ctxt_c2w = []
        ctxt_intrinsics = []
        for id in ctxt_idx:
            id = min(id, len(rgb_files) - 1)
            rgb, cam_param = self.read_image(rgb_files, id)
            ctxt_rgbs.append(rgb)
            ctxt_intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
            ctxt_c2w.append(cam_param.c2w_mat)
        ctxt_c2w = torch.tensor(np.array(ctxt_c2w)).float()
        ctxt_rgb = torch.stack(ctxt_rgbs, axis=0)
        ctxt_intrinsics = torch.stack(ctxt_intrinsics, axis=0)

        render_poses = []
        num_frames_render = min(ctxt_idx[0], len(rgb_files) - 1) - min(
            trgt_idx[0], len(rgb_files) - 1
        )
        trgt_rgbs = []
        for i in range(1, num_frames_render + 1):
            # id = ctxt_idx[0] + i * (trgt_idx[0] - ctxt_idx[0]) // (num_frames_render)
            id = trgt_idx[0] + i
            rgb_file = rgb_files[id]
            cam_param = self.all_cam_params[str(rgb_file.parent.name)][id][1:]
            cam_param = Camera(cam_param.flatten().tolist())
            render_poses.append(cam_param.c2w_mat)
            rgb, cam_param = self.read_image(rgb_files, id)
            trgt_rgbs.append(rgb)
        trgt_rgb = torch.stack(trgt_rgbs, axis=0)
        render_poses = torch.tensor(np.array(render_poses)).float()
        inv_ctxt_c2w = torch.inverse(ctxt_c2w[0])
        ctxt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            self.num_context, 1, 1
        )
        trgt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            self.num_target, 1, 1
        )
        trgt_inv_ctxt_c2w_repeat_video = inv_ctxt_c2w.unsqueeze(0).repeat(
            num_frames_render, 1, 1
        )
        return (
            {
                "ctxt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", ctxt_inv_ctxt_c2w_repeat, ctxt_c2w
                ),
                "trgt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", trgt_inv_ctxt_c2w_repeat, trgt_c2w
                ),
                "render_poses": torch.einsum(
                    "ijk, ikl -> ijl", trgt_inv_ctxt_c2w_repeat_video, render_poses
                ),
                "ctxt_rgb": self.normalize(ctxt_rgb),
                "trgt_rgb": self.normalize(trgt_rgb),
                "intrinsics": ctxt_intrinsics[0],
                "x_pix": rearrange(self.xy_pix, "h w c -> (h w) c"),
                "x_pix_128": rearrange(self.xy_pix_128, "h w c -> (h w) c"),
                # "idx": torch.tensor([index]),
                "image_shape": torch.tensor([self.image_size, self.image_size, 3]),
                "folder_path": str(rgb_files[0].parent),
            },
            trgt_rgb,  # rearrange(rendered["image"], "c h w -> (h w) c"),
        )
