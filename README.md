# Diffusion with Forward Models: Solving Stochastic Inverse Problems Without Direct Supervision 

[Project Page](https://diffusion-with-forward-models.github.io) | [Paper](https://arxiv.org/abs/2306.11719)

![](images/teaser.gif)

## Abstract 

Denoising diffusion models have emerged as a powerful class of generative models
capable of capturing the distributions of complex, real-world signals. However,
current approaches can only model distributions for which training samples are
directly accessible, which is not the case in many real-world tasks. In inverse
graphics, for instance, we seek to sample from a distribution over 3D scenes
consistent with an image but do not have access to ground-truth 3D scenes, only
2D images. We present a new class of conditional denoising diffusion probabilistic
models that learn to sample from distributions of signals that are never observed
directly, but instead are only measured through a known differentiable forward
model that generates partial observations of the unknown signal. To accomplish
this, we directly integrate the forward model into the denoising process. At test
time, our approach enables us to sample from the distribution over underlying
signals consistent with some partial observation. We demonstrate the efficacy of
our approach on three challenging computer vision tasks. For instance, in inverse
graphics, we demonstrate that our model in combination with a 3D-structured
conditioning method enables us to directly sample from the distribution of 3D
scenes consistent with a single 2D input image.

## Usage

### Environment Setup 


```bash
conda create -n dfm python=3.9 -y 
conda activate dfm 
pip install torch==2.0.1 torchvision
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath 
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html
pip install -r requirements.txt
python setup.py develop
```


### Pretrained Models

You can download the pretrained mode from [here](https://mitprod-my.sharepoint.com/:f:/g/personal/ayusht_mit_edu/EllGm7veCjVBp6UN8YMyncQBVN17FMLhgro0lowDsu8FTQ?e=sptoYF) and place it in the files folder.

### Prepare CO3D Dataset

```bash 
python data_io/co3d_new.py --generate_info_file  --generate_camera_quality_file --generate_per_scene_scale --dataset_root CO3D_ROOT 
```

The scene scale calculation can take a few hours. You can also download our precomputed statistics from [here](https://mitprod-my.sharepoint.com/:f:/g/personal/ayusht_mit_edu/EllGm7veCjVBp6UN8YMyncQBVN17FMLhgro0lowDsu8FTQ?e=sptoYF), and skip this flag during dataset preparation. 

### CO3D Inference 

```bash
# hydrant one shot (faster, used for metric comparison)
python experiment_scripts/co3d_results.py dataset=CO3D name=co3d_oneshot_debug_new_branch ngpus=1 feats_cond=True wandb=online checkpoint_path=files/co3d_model.pt   use_abs_pose=True sampling_type=oneshot use_dataset_pose=True image_size=128

# hydrant 5-step  (slower, used for visualization)
python experiment_scripts/co3d_results.py dataset=CO3D name=co3d_autoregressive_5step ngpus=1 feats_cond=True wandb=online checkpoint_path=files/co3d_model.pt  use_abs_pose=True sampling_type=autoregressive use_dataset_pose=True  all_class=True test_autoregressive_stepsize=41 image_size=128
```

### CO3D Training
```bash 
# first train two-view pixelnerf  
torchrun  --nnodes 1 --nproc_per_node 8   experiment_scripts/train_pixelnerf.py dataset=CO3D name=pn_2ctxt  num_context=2 num_target=2 lr=2e-5 batch_size=16  wandb=online use_abs_pose=true scale_aug_ratio=0.2

# train at 64 resolution
torchrun  --nnodes 1 --nproc_per_node 8 experiment_scripts/train_3D_diffusion.py use_abs_pose=True dataset=CO3D lr=2e-5 ngpus=8 setting_name=co3d_3ctxt feats_cond=True wandb=online dataset.lpips_loss_weight=0.2 name=co3d scale_aug_ratio=0.2 load_pn=True checkpoint_path=PN_PATH

# finetune model at 128 resolution 
torchrun  --nnodes 1 --nproc_per_node 8 experiment_scripts/train_3D_diffusion.py use_abs_pose=True dataset=CO3D lr=2e-5 ngpus=8 setting_name=co3d_3ctxt feats_cond=True wandb=online dataset.lpips_loss_weight=0.2 name=co3d_128res scale_aug_ratio=0.2 checkpoint_path=CKPT_64  image_size=128
```

### Prepare RealEstate10k Dataset

Download the dataset following the instructions [here](https://github.com/yilundu/cross_attention_renderer/blob/master/data_download/README.md).

### RealEstate10k Inference 

```bash
python experiment_scripts/re_results.py dataset=realestate batch_size=1 num_target=1 num_context=1 model_type=dit feats_cond=true sampling_type=simple max_scenes=10000 stage=test use_guidance=true guidance_scale=2.0 temperature=0.85 sampling_steps=50 name=re10k_inference image_size=128 checkpoint_path=files/re10k_model.pt wandb=online
```

### RealEstate10k Training

```bash 
# train at 64 resolution
torchrun  --nnodes 1 --nproc_per_node 8 experiment_scripts/train_3D_diffusion.py dataset=realestate setting_name=re name=re10k mode=cond feats_cond=true wandb=online ngpus=8 use_guidance=true image_size=64

# finetune at 128 resolution 
torchrun  --nnodes 1 --nproc_per_node 8 experiment_scripts/train_3D_diffusion.py dataset=realestate setting_name=re_128res name=re10k mode=cond feats_cond=true wandb=online ngpus=8 use_guidance=true checkpoint_path=TBA image_size=128
```

### Logging

We use wandb for logging. Enter the relevant information in configurations/wandb/online.yaml to use this feature. Logging can be disabled by setting wandb=local. 


### Citation
If you find our work useful in your research, please cite:
```
@article{tewari2023diffusion,
      title={Diffusion with Forward Models: Solving Stochastic Inverse Problems Without Direct Supervision}, 
      author={Ayush Tewari and Tianwei Yin and George Cazenavette and Semon Rezchikov and Joshua B. Tenenbaum and Frédo Durand and William T. Freeman and Vincent Sitzmann},
      year={2023},
      journal={NeurIPS}
}
```

