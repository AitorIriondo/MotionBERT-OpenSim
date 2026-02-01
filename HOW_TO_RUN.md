# How to Run MotionBERT-OpenSim

This guide shows all available commands and their arguments.

## Prerequisites

1. Activate the conda environment:
   ```bash
   conda activate motionbert-opensim
   ```

2. Navigate to the project directory:
   ```bash
   cd E:\MotionBERT-OpenSim
   ```

---

## Main Pipeline: Video to OpenSim

The main script `run_full_pipeline.py` runs the complete pipeline from video to OpenSim joint angles.

### Basic Usage

```bash
python run_full_pipeline.py --input input/video.mp4
```

### All Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | **required** | Input video file path |
| `--output` | `-o` | `output/pipeline_output` | Output directory |
| `--height` | | `1.75` | Subject height in meters |
| `--device` | `-d` | `cuda:0` | Device (`cuda:0` or `cpu`) |
| `--no-timing` | | disabled | Disable timing output |

### Examples

```bash
# Basic run with default settings
python run_full_pipeline.py --input input/aitor_garden_walk.mp4

# Specify subject height (1.80m) and output directory
python run_full_pipeline.py --input input/my_video.mp4 --output output/my_results --height 1.80

# Run on CPU (slower but no GPU required)
python run_full_pipeline.py --input input/video.mp4 --device cpu

# Disable timing output for cleaner logs
python run_full_pipeline.py --input input/video.mp4 --no-timing

# Full example with all options
python run_full_pipeline.py \
    --input input/aitor_garden_walk.mp4 \
    --output output/aitor_results \
    --height 1.75 \
    --device cuda:0
```

### Output Files

After running, you'll find these files in the output directory:

```
output/my_results/
├── motion.trc                      # Marker trajectories (TRC format)
├── pose-3d/
│   └── motion.trc                  # Copy for Pose2Sim
└── kinematics/
    ├── motion.mot                  # Joint angles (OpenSim motion file)
    ├── motion_scaled.osim          # Scaled OpenSim model
    ├── motion_ik_setup.xml         # IK configuration
    ├── motion_scaling_setup.xml    # Scaling configuration
    ├── _ik_marker_errors.sto       # Marker error report
    └── opensim_logs.txt            # OpenSim processing logs
```

---

## Video Processing Only (No OpenSim)

Use `process_video.py` if you only want to extract 3D poses without running OpenSim.

### Basic Usage

```bash
python process_video.py --input input/video.mp4 --output output/poses
```

### All Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | **required** | Input video file path |
| `--output` | `-o` | `output` | Output directory |
| `--device` | `-d` | `cuda:0` | Device (`cuda:0` or `cpu`) |
| `--no-progress` | | disabled | Disable progress bars |

### Output Files

```
output/poses/
├── keypoints_2d.json               # YOLOv8 2D keypoints (COCO17)
├── keypoints_2d_h36m.pkl           # Converted to H36M format
└── keypoints_3d.pkl                # MotionBERT 3D keypoints
```

---

## Alternative Pipeline Scripts

### Timed Pipeline (with Detailed Benchmarks)

For performance analysis with detailed timing:

```bash
python run_full_pipeline_timed.py --input input/video.mp4 --output output/timed_test --height 1.75
```

### MotionBERT Pipeline (from Pickle)

If you already have H36M format pickle files:

```bash
python run_motionbert_pipeline.py
```

This script reads from a pre-configured pickle file and runs the MotionBERT -> OpenSim pipeline.

---

## Common Workflows

### Workflow 1: Process a New Video

```bash
# Step 1: Run full pipeline
python run_full_pipeline.py --input input/my_video.mp4 --output output/my_session --height 1.80

# Step 2: Open results in OpenSim GUI
# - Open: output/my_session/kinematics/motion_scaled.osim
# - Load motion: output/my_session/kinematics/motion.mot
```

### Workflow 2: Batch Process Multiple Videos

```bash
# Windows batch script
for %%f in (input\*.mp4) do (
    python run_full_pipeline.py --input "%%f" --output "output\%%~nf" --height 1.75
)
```

```bash
# Linux/Mac bash script
for f in input/*.mp4; do
    name=$(basename "$f" .mp4)
    python run_full_pipeline.py --input "$f" --output "output/$name" --height 1.75
done
```

### Workflow 3: Different Subject Heights

```bash
# Person A: 1.65m
python run_full_pipeline.py --input input/person_a.mp4 --output output/person_a --height 1.65

# Person B: 1.85m
python run_full_pipeline.py --input input/person_b.mp4 --output output/person_b --height 1.85
```

### Workflow 4: Test on CPU First

```bash
# Quick test on CPU (slower)
python run_full_pipeline.py --input input/short_clip.mp4 --device cpu

# Full run on GPU
python run_full_pipeline.py --input input/full_video.mp4 --device cuda:0
```

---

## Pipeline Stages

The full pipeline runs these 8 stages:

| Stage | Description | Model/Tool |
|-------|-------------|------------|
| 1 | 2D keypoint detection | YOLOv8-Pose |
| 2 | COCO17 -> H36M conversion | Custom converter |
| 3 | 2D -> 3D lifting | MotionBERT (DSTformer) |
| 4 | Bone length normalization | Custom algorithm |
| 5 | Coordinate transformation | MotionBERT -> OpenSim axes |
| 6 | Marker extraction | H36M -> COCO17 (14 markers) |
| 7 | TRC file export | OpenSim format |
| 8 | Scaling + Inverse Kinematics | Pose2Sim + OpenSim |

---

## Performance Expectations

On NVIDIA RTX 2080:

| Video Length | Processing Time | Speed |
|--------------|-----------------|-------|
| 10 seconds | ~16 seconds | 0.62x real-time |
| 30 seconds | ~48 seconds | 0.62x real-time |
| 1 minute | ~96 seconds | 0.62x real-time |
| 5 minutes | ~8 minutes | 0.62x real-time |

**Bottlenecks:**
- YOLOv8 detection: 42% of total time
- Pose2Sim IK: 13% of total time
- MotionBERT: 2% of total time

---

## Troubleshooting

### Error: "CUDA out of memory"

```bash
# Use CPU mode
python run_full_pipeline.py --input video.mp4 --device cpu
```

### Error: "No module named 'ultralytics'"

```bash
pip install ultralytics
```

### Error: "No module named 'Pose2Sim'"

```bash
pip install pose2sim
```

### Error: Video file not found

Make sure the path is correct and use forward slashes or escaped backslashes:
```bash
# Correct
python run_full_pipeline.py --input input/video.mp4
python run_full_pipeline.py --input "E:/Videos/my video.mp4"

# Wrong
python run_full_pipeline.py --input E:\Videos\my video.mp4
```

### Warning: "CUDA not available, using CPU"

Check your PyTorch CUDA installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False, reinstall PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Quick Reference

```bash
# Most common command
python run_full_pipeline.py -i input/video.mp4 -o output/results --height 1.75

# Short form with defaults
python run_full_pipeline.py -i input/video.mp4

# CPU mode
python run_full_pipeline.py -i input/video.mp4 -d cpu
```
