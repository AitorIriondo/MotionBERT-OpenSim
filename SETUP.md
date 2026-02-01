# MotionBERT-OpenSim Setup Guide

Video-to-OpenSim pipeline using YOLOv8-Pose + MotionBERT + Pose2Sim for markerless motion capture.

## System Requirements

- **OS**: Windows 10/11 (tested), Linux (should work)
- **GPU**: NVIDIA GPU with CUDA support (recommended)
  - Tested on: RTX 2080 (8GB VRAM)
  - Minimum: ~4GB VRAM
- **RAM**: 16GB recommended
- **Disk**: ~5GB for environment + models

## Quick Setup

### Option 1: Conda Environment (Recommended)

```bash
# Create new conda environment
conda create -n motionbert-opensim python=3.10 -y
conda activate motionbert-opensim

# Install PyTorch with CUDA (adjust CUDA version as needed)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
# pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

### Option 2: pip only

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Verify Installation

```bash
# Activate environment
conda activate motionbert-opensim

# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLOv8: OK')"
python -c "from Pose2Sim import Pose2Sim; print('Pose2Sim: OK')"
python -c "from lib.model.DSTformer import DSTformer; print('MotionBERT: OK')"
```

Expected output:
```
PyTorch: 2.x.x, CUDA: True
YOLOv8: OK
Pose2Sim: OK
MotionBERT: OK
```

## Project Structure

```
MotionBERT-OpenSim/
├── input/                          # Input videos
│   └── aitor_garden_walk.mp4       # Example video
├── output/                         # Output files
├── src/                            # Source modules
│   ├── trc_motionbert.py           # TRC export for MotionBERT
│   └── ...
├── utils/                          # Utilities
│   └── keypoint_converter.py       # COCO17 to H36M conversion
├── MotionBERT/                     # MotionBERT model
│   ├── lib/                        # Model code
│   ├── configs/                    # Model configs
│   └── checkpoint/                 # Model weights (64MB)
├── models/                         # OpenSim models
│   └── Model_Pose2Sim_simple.osim
├── run_full_pipeline.py            # Main pipeline script
├── process_video.py                # Video processing only
├── requirements.txt                # Python dependencies
├── SETUP.md                        # This file
├── HOW_TO_RUN.md                   # Usage guide
└── DOCUMENTATION.md                # Full documentation
```

## Model Files

The repository includes all necessary model files:

| File | Size | Description |
|------|------|-------------|
| `MotionBERT/checkpoint/.../best_epoch.bin` | 64MB | MotionBERT weights (H36M fine-tuned) |
| `models/Model_Pose2Sim_simple.osim` | 284KB | OpenSim simple skeleton model |

**Note**: YOLOv8 weights (`yolov8m-pose.pt`, ~50MB) are downloaded automatically on first run.

## Troubleshooting

### CUDA Not Available

```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### OpenSim/Pose2Sim Issues

```bash
# Reinstall Pose2Sim
pip uninstall pose2sim
pip install pose2sim
```

### Memory Issues (Out of VRAM)

Use CPU mode:
```bash
python run_full_pipeline.py --input video.mp4 --device cpu
```

### Import Errors

Make sure you're in the project root directory:
```bash
cd E:\MotionBERT-OpenSim
python run_full_pipeline.py --input input/video.mp4
```

## Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0 | Deep learning framework |
| torchvision | >=0.15 | Image processing |
| ultralytics | >=8.0 | YOLOv8 pose detection |
| opencv-python | >=4.8 | Video I/O |
| numpy | >=1.24 | Array operations |
| tqdm | >=4.65 | Progress bars |
| pyyaml | >=6.0 | Config loading |
| easydict | >=1.10 | Config handling |
| pose2sim | >=0.10 | OpenSim interface |

## Hardware Performance

Tested on NVIDIA RTX 2080 (8GB VRAM):

| Component | Speed | % Total Time |
|-----------|-------|--------------|
| YOLOv8 2D Detection | ~50 FPS | 42% |
| MotionBERT 3D Lifting | ~1700 FPS | 2% |
| Pose2Sim IK | ~140 FPS | 13% |
| **End-to-End** | **~19 FPS** | 100% |

Real-time ratio: **0.62x** (processes 19 FPS video content per second)
