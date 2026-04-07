# VolSegGS: Segmentation and Tracking in Dynamic Volumetric Scenes via Deformable 3D Gaussians

[Paper](https://arxiv.org/abs/2507.12667)

This repository contains the official implementation associated with the paper "VolSegGS: Segmentation and Tracking in Dynamic Volumetric Scenes via Deformable 3D Gaussians".

## Run

### Environment

```shell
git clone https://github.com/JCBreath/VolSegGS
cd VolSegGS

conda create -n vsgs python=3.9
conda activate vsgs

# install pytorch
conda install -c nvidia/label/cuda-12.8.1 cuda-toolkit
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
conda install gxx_linux-64

# install submodules
cd VolSegGS
python -m pip install submodules/simple-knn/
python -m pip install submodules/diff-gaussian-rasterization

# install dependencies
python -m pip install imageio tqdm scipy plyfile open3d numpy==1.24
python -m pip install opencv-python dearpygui 
python -m pip install lpips
python -m pip install mmcv
```

### Train

```shell
python train.py -s data/vortex --expname "vortex" --configs arguments/dynerf/dnerf_default_visnerf.py
```

### View with GUI

```shell
python gui.py --model_path output/vortex/
```



