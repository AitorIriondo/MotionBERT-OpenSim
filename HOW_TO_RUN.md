# How to Run MotionBERT-OpenSim

## 1. Setup

```cmd
cd C:\MotionBERTOpenSim\MotionBERT-OpenSim
venv\Scripts\activate
```

## 2. Run the Pipeline

```cmd
python run_full_pipeline.py --input video.mp4 --height 1.75 --filter
```

## 3. Command-Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | **required** | Input video file path |
| `--output` | `-o` | Auto | Output directory (auto-generates as `output_YYYYMMDD_HHMMSS_videoname`) |
| `--height` | | `1.75` | Subject height in meters |
| `--device` | `-d` | `cuda:0` | Device (`cuda:0` for GPU, `cpu` for CPU) |
| `--filter` | `-f` | Off | Enable Butterworth smoothing filter (recommended) |
| `--filter-cutoff` | | `6.0` | Filter cutoff frequency in Hz (lower = smoother) |
| `--export-fbx` | | Off | Export motion to FBX format (requires Blender) |
| `--blender-path` | | Auto | Path to Blender executable (auto-detected) |
| `--no-timing` | | Off | Disable timing output |

## 4. Examples

```cmd
# Basic run
python run_full_pipeline.py --input video.mp4 --height 1.75

# With smoothing filter (recommended)
python run_full_pipeline.py --input video.mp4 --height 1.75 --filter

# With FBX export for Unity/Unreal/Blender
python run_full_pipeline.py --input video.mp4 --height 1.75 --filter --export-fbx

# Custom filter cutoff (lower = more smoothing)
python run_full_pipeline.py --input video.mp4 --height 1.75 --filter --filter-cutoff 4.0

# Specify output folder
python run_full_pipeline.py --input video.mp4 --output my_output --height 1.75 --filter

# Run on CPU
python run_full_pipeline.py --input video.mp4 --height 1.75 --device cpu

# Full example with all options
python run_full_pipeline.py --input video.mp4 --height 1.75 --filter --export-fbx
```

## 5. Output Files

```
output_YYYYMMDD_HHMMSS_videoname/
├── motion.trc                    # Marker trajectories
├── motion.fbx                    # FBX animation (if --export-fbx)
├── processing_report.txt         # Processing summary
├── processing_report.json        # Processing data (JSON)
├── pose-3d/
│   └── motion.trc
└── kinematics/
    ├── motion.mot                # Joint angles (load in OpenSim)
    ├── motion.osim               # Scaled model (open in OpenSim)
    └── _ik_marker_errors.sto     # IK errors
```

## 6. View Results in OpenSim

1. Open OpenSim GUI
2. File -> Open Model -> Select `kinematics/motion.osim`
3. File -> Load Motion -> Select `kinematics/motion.mot`
4. Click Play

## 7. FBX Export Requirements

To use `--export-fbx`, you need:

1. **Blender** installed (https://www.blender.org/download/)
2. **pyopensim** installed in Blender's Python:
   ```cmd
   "C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe" -m pip install pyopensim
   ```

The FBX file can be imported into Unity, Unreal Engine, Maya, or any 3D software.
