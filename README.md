# Cooperative 3D Object Detection for V2X (DAIR-V2X)

> **Status: work in progress.** The model and training scaffolding is in place. Training and evaluation results are not available yet.

Vehicle-infrastructure cooperative perception: fuse LiDAR point clouds from a roadside unit and an ego vehicle to detect cars, pedestrians and cyclists beyond the range of a single sensor. Infrastructure-to-vehicle communication over 5G is the motivating use case.

## Approach

```
infrastructure LiDAR ─► PointPillars backbone ─┐
                                               ├─► cross-attention fusion ─► multi-group detection head ─► 3D boxes
vehicle LiDAR ───────► PointPillars backbone ─┘     (intermediate / feature-level)   (Car, Pedestrian, Cyclist)
```

- `src/models/backbones/pointpillars.py`: pillar feature encoder
- `src/models/fusion/intermediate_fusion.py`: self-attention per agent, then cross-attention between infrastructure and vehicle BEV features
- `src/models/heads/multi_group_head.py`: classification and box regression
- `src/data/`: DAIR-V2X cooperative dataset loader, I2V calibration, voxelisation, augmentations
- `configs/base.yaml`: dataset splits, voxel grid, model and training hyper-parameters

## Setup

The dataset is **not** included. Request DAIR-V2X from the [official site](https://thudair.baai.ac.cn/index) and place it under `data/dair_v2x/`.

```bash
scripts/build_docker.sh      # CUDA image (docker/Dockerfile)
scripts/run_docker.sh        # starts the container and opens a shell
python scripts/train.py   # inside the container; reads configs/base.yaml
```

## Roadmap

- [ ] Finish the train/eval loop and report AP@0.5/0.7 against late-fusion and vehicle-only baselines
- [ ] Simulate communication bandwidth and latency limits for the shared features
- [ ] Visualise predictions (`viz/visualize_predictions.py`)

## Stack

PyTorch, einops, Docker (NVIDIA runtime), DAIR-V2X.
