import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from jaxtyping import Float
from omegaconf import DictConfig

from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.dataset_map_provider import DatasetMap
from data_io.co3d.json_index_dataset_map_provider_v2 import JsonIndexDatasetMapProviderV2
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.utils import opencv_from_cameras_projection
from geometry import get_opencv_pixel_coordinates
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import os 

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def get_dataset_map(
    dataset_root: str,
    category: str,
    subset_name: str,
) -> DatasetMap:
    """
    Obtain the dataset map that contains the train/val/test dataset objects.
    """
    expand_args_fields(JsonIndexDatasetMapProviderV2)
    dataset_map_provider = JsonIndexDatasetMapProviderV2(
        category=category,
        subset_name=subset_name,
        dataset_root=dataset_root,
        test_on_train=False,
        only_test_set=False,
        load_eval_batches=True,
        dataset_JsonIndexDataset_args=DictConfig(
            {
                "remove_empty_masks": False,
                "load_point_clouds": False,
                "load_depth_masks": False,
                "load_depths": False,
                "center_crop": True,
                "load_masks": False 
            }
        ),
    )
    return dataset_map_provider.get_dataset_map()


class CO3DDataset(Dataset):
    root: Path
    stage: str
    dataset: JsonIndexDatasetMapProviderV2
    image_size: int
    num_context_views: int
    num_target_views: int
    mean_scale: float  
    z_near: float = 0.5
    z_far: float = 40
    background_color: torch.tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32) 
    context_min_distance: int = 20
    context_max_distance: int = 40

    def __init__(
        self,
        root: str,
        num_context: int,
        num_target: int,
        categories: List[str] = ['hydrant'],
        image_size: int = 64,
        stage: str = "train",
        filter_num: int = 500,
        scale_aug_ratio: float = 0,
        mean_depth_scale: float =10.0,
        noise: float = 0.0,
        eval_10_frame: bool = False 
    ) -> None:
        super().__init__()
        self.stage = stage
        self.root = Path(root) if root is not None else root 
        self.image_size = image_size
        self.num_context_views = num_context
        self.num_target_views = num_target

        # whether or not we want to filter the dataset 
        ISFILTER = stage == 'train' and filter_num > 0

        datasets = [] 
        per_category_camera_score_threshold = []
        camera_quality_score_list = [] 
        for category_index, category in enumerate(categories):
            cache_path = f'dataset_cache_new/{category}_{stage}.pt'
            dataset_map = torch.load(cache_path, map_location='cpu') 
            datasets.append(dataset_map)

            # load the per-class camera quality score 
            if ISFILTER:
                camera_quality_score = torch.load(f'dataset_cache_new/camera_quality_dict_{category}.pt', 
                    map_location='cpu')
                scores = torch.stack([v for v in camera_quality_score.values()])
                score_threshold = torch.sort(scores, descending=True)[0][filter_num]
                per_category_camera_score_threshold.append(score_threshold)
                camera_quality_score_list.append(camera_quality_score)


        self.datasets = datasets 
        self.all_sequences = [] 
        self.index_to_class = [] 
        for dataset_class, dataset in enumerate(self.datasets):
            all_sequence = sorted(list(dataset.sequence_names()))

            if ISFILTER: 
                filtered_sequence = [] 
                for sequence in all_sequence:
                    if (camera_quality_score_list[dataset_class][sequence] >
                        per_category_camera_score_threshold[dataset_class]):
                        filtered_sequence.append(sequence)

                all_sequence = filtered_sequence

            if eval_10_frame:
                all_sequence = all_sequence[:10]

            self.all_sequences.extend(all_sequence)
            self.index_to_class.append(np.ones(len(all_sequence), dtype=int) * dataset_class)

        self.index_to_class = np.concatenate(self.index_to_class).tolist()
        self.xy_pix = get_opencv_pixel_coordinates(
            x_resolution=self.image_size, y_resolution=self.image_size
        )
        self.normalize = normalize_to_neg_one_to_one
        self.scene_scales = torch.load(f'./dataset_cache_new/scale_dict.pt', map_location='cpu')
        self.mean_scale = mean_depth_scale
        self.scale_aug_ratio = scale_aug_ratio
        self.noise = noise 

    def __len__(self) -> int:
        return len(self.all_sequences)

    def resize(self, image: Float[Tensor, "C H W"]) -> Float[Tensor, "C RH RW"]:

        image_npy = image
        return torch.nn.functional.interpolate(
            image_npy[None],
            size=(self.image_size, self.image_size),
            mode="bilinear",
            antialias=True,
        )[0]

    def extract_frame_data(self, frame_data: FrameData, scale:float):
        _, h, w = frame_data.image_rgb.shape
        r, t, k = opencv_from_cameras_projection(
            frame_data.camera, torch.tensor([[h, w]])
        )
        # Extract c2w (OpenCV-style camera-to-world transformation, extrinsics).
        r = r[0]
        t = t[0]
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, :3] = r.T
        c2w[:3, 3] = -r.T @ t / scale  # * self.mean_scale

        if self.noise > 0:
            c2w[:3, 3] += torch.randn(3) * self.noise

        c2w[:3, 3] = c2w[:3, 3] * self.mean_scale

        if torch.isnan(c2w).any():
            import pdb; pdb.set_trace()

        # Extract K (camera intrinsics).
        k = k[0]
        k[:2] /= torch.tensor([w, h])[:, None]

        rgb = frame_data.image_rgb

        return {
            "key": Path(frame_data.image_path).stem,
            "c2w": c2w,
            "k": k,
            "rgb": self.resize(rgb).clip(min=0, max=1),
        }

    def __getitem__(self, index: int):
        # Get a dataset (one per category) and sequence from the merged dataset.
        sequence = self.all_sequences[index]
        current_class = self.index_to_class[index]
        dataset = self.datasets[current_class]

        indices = list(dataset.sequence_indices_in_order(sequence))

        scene_scale = self.scene_scales[sequence]

        if self.stage == 'train' and self.scale_aug_ratio > 0:
            aug_ratio = (torch.rand(1).item() * 2 - 1)*self.scale_aug_ratio + 1
            scene_scale = scene_scale * aug_ratio


        if scene_scale == 0:
            print(f"BROKEN sequence {sequence}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        num_context = self.num_context_views
        context_min_distance = self.context_min_distance * num_context
        context_max_distance = self.context_max_distance * num_context

        start_idx = random.sample(np.arange(len(indices)).tolist(), 1)[0]

        end_idx = random.choice(
            np.arange(
                start_idx + context_min_distance, start_idx + context_max_distance,
            )
        )

        if end_idx >= len(indices):
            return self.__getitem__(random.randint(0, len(self) - 1))

        trgt_idx = random.sample(
            np.arange(start_idx, end_idx).tolist(), k=self.num_target_views
        )

        flip = random.choice([True, False])
        if flip:
            temp = start_idx
            start_idx = end_idx
            end_idx = temp

        ctxt_idx = [start_idx]
        trgt_idx[0] = end_idx

        if num_context != 1:
            distance = random.choice(
                np.arange(self.context_min_distance, self.context_max_distance)
            )
            if start_idx < end_idx:
                extra_ctxt_idx = random.sample(
                    np.arange(
                        start_idx, max(start_idx + num_context - 1, end_idx - distance),
                    ).tolist(),
                    k=num_context - 1
                )
            else:
                extra_ctxt_idx = random.sample(
                    np.arange(
                        min(start_idx - num_context + 1, end_idx + distance), start_idx,
                    ).tolist(),
                    k=num_context - 1
                )
            ctxt_idx.extend(extra_ctxt_idx)

        if flip:
            # sort the target indices increasingly
            trgt_idx = np.sort(trgt_idx)
        else:
            # sort the target indices decreasingly
            trgt_idx = np.sort(trgt_idx)[::-1]

        context_indices = [indices[idx] for idx in ctxt_idx] 
        target_indices = [indices[idx] for idx in trgt_idx] 

        context_frames = [
            self.extract_frame_data(dataset[context_index], scale=scene_scale)
            for context_index in context_indices
        ]

        target_frames = [
            self.extract_frame_data(dataset[target_index], scale=scene_scale)
            for target_index in target_indices
        ]

        ctxt_rgb = torch.stack([self.normalize(frame["rgb"]) for frame in context_frames])
        trgt_rgb = torch.stack([self.normalize(frame["rgb"]) for frame in target_frames])

        ctxt_c2w = torch.stack([frame["c2w"] for frame in context_frames])
        trgt_c2w = torch.stack([frame["c2w"] for frame in target_frames])

        inv_ctxt_c2w = torch.inverse(ctxt_c2w[0])
        inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(trgt_rgb.shape[0], 1, 1)
        ctxt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            ctxt_rgb.shape[0], 1, 1
        )

        all_ctx_intrinsics = torch.stack([frame["k"] for frame in context_frames])
        all_trgt_intrinsics = torch.stack([frame["k"] for frame in target_frames])

        return (
            {
                "ctxt_rgb": ctxt_rgb,
                "trgt_rgb": trgt_rgb,
                "ctxt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", ctxt_inv_ctxt_c2w_repeat, ctxt_c2w
                ),
                "trgt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", inv_ctxt_c2w_repeat, trgt_c2w
                ),
                "inv_ctxt_c2w": ctxt_inv_ctxt_c2w_repeat,
                "intrinsics": context_frames[0]["k"],
                "x_pix": rearrange(self.xy_pix, "h w c -> (h w) c"),
                "image_shape": torch.tensor([self.image_size, self.image_size, 3]),
                "idx": torch.tensor([index]),
                "all_ctx_intrinsics": all_ctx_intrinsics,
                "all_trgt_intrinsics": all_trgt_intrinsics,
                "ctxt_abs_camera_poses": ctxt_c2w,
                "trgt_abs_camera_poses": trgt_c2w
            },
            rearrange(trgt_rgb[0], "c h w -> (h w) c"),  
        )


    def data_for_video(self, video_idx: int, load_all_frames: bool = False):
        # extract the middle frame and 20 frames before and after
        sequence = self.all_sequences[video_idx]
        current_class = self.index_to_class[video_idx]
        dataset = self.datasets[current_class]

        indices = (list(dataset.sequence_indices_in_order(sequence)))

        if load_all_frames:
            context_indices = indices 
            target_indices = indices
        else:
            random_idx = random.randint(0, len(indices)-50)
            indices = indices[random_idx:random_idx+40:2]
            ctxt_idx = 0 
            context_indices = [indices[ctxt_idx]]
            target_indices = [indices[-1]] + indices

        scene_scale = self.scene_scales[sequence]
        
        context_frames = [
            self.extract_frame_data(dataset[context_index], scale=scene_scale)
            for context_index in context_indices
        ]

        target_frames = [
            self.extract_frame_data(dataset[target_index], scale=scene_scale)
            for target_index in target_indices
        ]

        ctxt_rgb = torch.stack([self.normalize(frame["rgb"]) for frame in context_frames])
        trgt_rgb = torch.stack([self.normalize(frame["rgb"]) for frame in target_frames])

        ctxt_c2w = torch.stack([frame["c2w"] for frame in context_frames])
        trgt_c2w = torch.stack([frame["c2w"] for frame in target_frames])

        inv_ctxt_c2w = torch.inverse(ctxt_c2w[0])
        inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(trgt_rgb.shape[0], 1, 1)
        ctxt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            ctxt_rgb.shape[0], 1, 1
        )

        return (
            {
                "ctxt_rgb": ctxt_rgb,
                "trgt_rgb": trgt_rgb,
                "ctxt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", ctxt_inv_ctxt_c2w_repeat, ctxt_c2w
                ),
                "trgt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", inv_ctxt_c2w_repeat, trgt_c2w
                ),
                "intrinsics": context_frames[0]["k"],
                "x_pix": rearrange(self.xy_pix, "h w c -> (h w) c"),
                "ctxt_abs_camera_poses": ctxt_c2w,
                "trgt_abs_camera_poses": trgt_c2w,
                "start_idx": torch.tensor(random_idx) if not load_all_frames else torch.tensor(0) 

            },
            rearrange(trgt_rgb[0], "c h w -> (h w) c"),  
        )

        
def init_info_file(categories, dataset_root):
    os.makedirs('dataset_cache_new', exist_ok=True)

    for category_index, category in enumerate(tqdm(categories)):
        dataset_map = get_dataset_map(dataset_root, category, 'fewview_dev')
        for stage in ["train", "val"]:
            cache_path = f'dataset_cache_new/{category}_{stage}.pt'
            current_map = dataset_map[stage]
            torch.save(current_map, cache_path)

    print("finish generating dataset map")

def compute_scene_scale(categories, dataset_root):

    class CO3DDepth(Dataset):
        def __init__(self, root: str, num_context: int, num_target: int, categories: List[str] = ['hydrant'], image_size: int = 64, stage: str = "train") -> None:
            super().__init__()
            self.stage = stage
            self.root = Path(root)
            self.image_size = image_size
            self.num_context_views = num_context
            self.num_target_views = num_target

            datasets = [] 
            for category_index, category in enumerate(categories):
                cache_path = f'dataset_cache_new/{category}_{stage}.pt'
                dataset_map = torch.load(cache_path, map_location='cpu')
                datasets.append(dataset_map)

            self.datasets = datasets 

            self.all_sequences = [] 
            self.index_to_class = [] 
            for dataset_class, dataset in enumerate(self.datasets):
                all_sequence = sorted(list(dataset.sequence_names()))
                self.all_sequences.extend(all_sequence)
                self.index_to_class.append(np.ones(len(all_sequence)) * dataset_class)

            self.index_to_class = np.concatenate(self.index_to_class).astype(int).tolist()

        def __len__(self) -> int:
            return len(self.all_sequences)

        def get_depth_map_median(self, frame_data):
            mask = (
                frame_data.fg_probability> 0.4 
                if frame_data.fg_probability is not None
                else None
            )        

            depth_map = frame_data.depth_map
            if mask.sum() <= 0:
                return None 
            else:
                return depth_map[mask].median()

        def __getitem__(self, index: int):
            sequence = self.all_sequences[index]
            current_class = self.index_to_class[index]
            dataset = self.datasets[current_class]

            dataset.center_crop = True 
            dataset.load_masks = True 
            dataset.load_depths = True
            dataset.load_depth_masks = True 

            indices = (list(dataset.sequence_indices_in_order(sequence)))

            scales = [] 

            for idx in indices:
                frame_data = dataset[idx]
                depth_map_median = self.get_depth_map_median(frame_data)
                if depth_map_median is not None:
                    scales.append(depth_map_median)

            if len(scales) == 0:
                return 10, sequence 
            
            return torch.mean(torch.stack(scales)).item(), sequence

    train_dataset = CO3DDepth(
        root=dataset_root,
        num_context=1,
        num_target=1,
        stage='train',
        image_size=128,
        categories=categories
    ) 
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=8, shuffle=False, collate_fn=lambda x: x[0])

    scale_dict = {} 
    for i, data in tqdm(enumerate(dataloader), total=len(train_dataset)):
        scale, sequence_name = data
        scale_dict[sequence_name] = scale

    val_dataset = CO3DDepth(
        root=dataset_root,
        num_context=1,
        num_target=1,
        stage='val',
        image_size=128,
        categories=categories
    ) 
    dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=8, shuffle=False, collate_fn=lambda x: x[0])

    for i, data in tqdm(enumerate(dataloader), total=len(val_dataset)):
        scale, sequence_name = data
        scale_dict[sequence_name] = scale

    torch.save(scale_dict, 'dataset_cache_new/scale_dict.pt')
    print("finish computing per scene scale")


def compute_camera_quality_score(categories, dataset_root):

    class CO3DCamera(Dataset):
        def __init__(self, root: str, num_context: int, num_target: int, categories: List[str] = ['hydrant'], image_size: int = 64, stage: str = "train", recompute: bool = False) -> None:
            super().__init__()
            self.stage = stage
            self.root = Path(root)
            self.image_size = image_size
            self.num_context_views = num_context
            self.num_target_views = num_target

            datasets = [] 
            for category_index, category in enumerate(categories):
                cache_path = f'dataset_cache_new/{category}_{stage}.pt'
                dataset_map = torch.load(cache_path, map_location='cpu')   
                datasets.append(dataset_map)

            self.datasets = datasets 

            self.all_sequences = [] 
            self.index_to_class = [] 
            for dataset_class, dataset in enumerate(self.datasets):
                all_sequence = sorted(list(dataset.sequence_names()))
                self.all_sequences.extend(all_sequence)
                self.index_to_class.append(np.ones(len(all_sequence)) * dataset_class)

            self.index_to_class = np.concatenate(self.index_to_class).astype(int).tolist()

        def __len__(self) -> int:
            return len(self.all_sequences)

        def __getitem__(self, index: int):
            sequence = self.all_sequences[index]
            current_class = self.index_to_class[index]
            dataset = self.datasets[current_class]
            indices = list(dataset.sequence_indices_in_order(sequence))
            return dataset[indices[0]].camera_quality_score, sequence

    # store camera quality score for each class indivudally to perform per class  top-k filtering later 
    for category in categories:
        train_dataset = CO3DCamera(
            root=dataset_root,
            num_context=1,
            num_target=1,
            stage='train',
            image_size=128,
            categories=[category]
        ) 
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=8, shuffle=False, collate_fn=lambda x: x[0])

        score_dict = {}
        for i, data in tqdm(enumerate(dataloader), total=len(train_dataset)):
            quality_score, sequence_name = data
            score_dict[sequence_name] = quality_score 

        val_dataset = CO3DCamera(
            root=dataset_root,
            num_context=1,
            num_target=1,
            stage='val',
            image_size=128,
            categories=[category]
        ) 
        dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=8, shuffle=False, collate_fn=lambda x: x[0])
        for i, data in tqdm(enumerate(dataloader), total=len(val_dataset)):
            quality_score, sequence_name = data
            score_dict[sequence_name] = quality_score 

        torch.save(score_dict, f'dataset_cache_new/camera_quality_dict_{category}.pt')

    print("finish computing camera quality score")

def test_dataloader(dataset_root, categories):
    dataset = CO3DDataset(
        root=dataset_root,
        num_context=1,
        num_target=1,
        stage='train',
        image_size=128,
        categories=categories
    ) 

    print(dataset[0])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_info_file', action='store_true')
    parser.add_argument('--generate_camera_quality_file', action='store_true')
    parser.add_argument("--generate_per_scene_scale", action="store_true")
    parser.add_argument("--all_classes", action="store_true")
    parser.add_argument("--dataset_root", type=str, required=True)
    args = parser.parse_args()

    if args.all_classes:
        categories = ["apple", "ball",  "bench",  "cake",  "donut",  "hydrant", "plant", "suitcase", "teddybear", "vase"]
    else:
        categories = ["hydrant"] 

    if args.generate_info_file:
        init_info_file(categories, args.dataset_root)

    if args.generate_camera_quality_file:
        compute_camera_quality_score(categories, args.dataset_root)

    if args.generate_per_scene_scale:
        compute_scene_scale(categories, args.dataset_root)

    test_dataloader(args.dataset_root, categories) 
