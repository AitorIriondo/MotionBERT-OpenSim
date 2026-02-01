# MotionBERT-OpenSim

Video-to-OpenSim pipeline for markerless motion capture using YOLOv8-Pose + MotionBERT + Pose2Sim.

## Overview

This pipeline converts monocular video of human movement into OpenSim-compatible joint angle data (.mot files), enabling biomechanical analysis from a single consumer camera.

```
Video (MP4) → YOLOv8-Pose → MotionBERT → Pose2Sim → OpenSim (.mot)
     │              │            │           │           │
     │         2D keypoints  3D lifting   IK solver   Joint angles
     │         (COCO17)      (H36M)       (scaling)   (40 DOF)
```

## Features

- **Monocular 3D pose estimation** using MotionBERT transformer model
- **Automatic skeleton scaling** to match participant height
- **Bone length normalization** to fix monocular depth estimation errors
- **Synthetic marker generation** for anatomically correct head tracking
- **Direct integration with Pose2Sim** for OpenSim inverse kinematics

## Performance

Tested on NVIDIA RTX 2080 (8GB VRAM):

| Component | Speed |
|-----------|-------|
| YOLOv8 2D Detection | ~50 FPS |
| MotionBERT 3D Lifting | ~1700 FPS |
| Pose2Sim IK | ~140 FPS |
| **End-to-End** | **~19 FPS** (0.62x real-time) |

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n motionbert-opensim python=3.10 -y
conda activate motionbert-opensim

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Model Weights

Download the MotionBERT checkpoint (~64MB):
```
MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
```

From: [MotionBERT releases](https://github.com/Walter0807/MotionBERT/releases)

### 3. Run Pipeline

```bash
python run_full_pipeline.py --input input/your_video.mp4 --output output/results --height 1.75
```

### 4. View Results

Open in OpenSim GUI:
1. Load model: `output/results/kinematics/motion_scaled.osim`
2. Load motion: `output/results/kinematics/motion.mot`

## Usage

```bash
# Basic usage
python run_full_pipeline.py --input video.mp4

# With all options
python run_full_pipeline.py \
    --input input/video.mp4 \
    --output output/my_results \
    --height 1.80 \
    --device cuda:0

# CPU mode (no GPU)
python run_full_pipeline.py --input video.mp4 --device cpu
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | required | Input video file |
| `--output` | `output/pipeline_output` | Output directory |
| `--height` | `1.75` | Subject height in meters |
| `--device` | `cuda:0` | Device (cuda:0 or cpu) |
| `--no-timing` | off | Disable timing output |

## Output Files

```
output/results/
├── motion.trc                    # Marker trajectories
├── pose-3d/
│   └── motion.trc               # Copy for Pose2Sim
└── kinematics/
    ├── motion.mot               # Joint angles (OpenSim motion)
    ├── motion_scaled.osim       # Scaled OpenSim model
    ├── _ik_marker_errors.sto    # IK error report
    └── opensim_logs.txt         # Processing logs
```

## Pipeline Stages

| Stage | Description | Output |
|-------|-------------|--------|
| 1 | YOLOv8-Pose 2D detection | COCO17 keypoints |
| 2 | COCO17 → H36M conversion | H36M format |
| 3 | MotionBERT 2D→3D lifting | 3D coordinates |
| 4 | Bone length normalization | Consistent skeleton |
| 5 | Coordinate transformation | OpenSim axes |
| 6 | Marker extraction | 14 COCO17 markers |
| 7 | TRC export | OpenSim format |
| 8 | Pose2Sim IK | Joint angles |

## Documentation

- [SETUP.md](SETUP.md) - Detailed installation guide
- [HOW_TO_RUN.md](HOW_TO_RUN.md) - Usage examples and workflows
- [DOCUMENTATION.md](DOCUMENTATION.md) - Full technical documentation

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (recommended)
- 8GB+ VRAM for GPU mode
- 16GB RAM

## Dependencies

- PyTorch >= 2.0
- ultralytics (YOLOv8)
- Pose2Sim >= 0.10
- OpenCV
- NumPy

## License

MIT License

## Acknowledgments

- [MotionBERT](https://github.com/Walter0807/MotionBERT) - 2D to 3D lifting
- [YOLOv8](https://github.com/ultralytics/ultralytics) - 2D pose detection
- [Pose2Sim](https://github.com/perfanalytics/pose2sim) - OpenSim integration
- [OpenSim](https://opensim.stanford.edu/) - Biomechanical simulation
