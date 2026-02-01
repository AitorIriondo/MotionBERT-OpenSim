# MotionBERT-OpenSim Pipeline Fixes Summary

## Date: 2026-02-01

## Problem
Converting MotionBERT H36M keypoints to OpenSim via Pose2Sim was producing severely distorted skeletons.

## Root Causes Found and Fixed

### 1. Bug in Pose2Sim's Markers_Coco17.xml (CRITICAL)
**Location:** `C:\Users\aitor\anaconda3\envs\rtmpose3d-opensim\lib\site-packages\Pose2Sim\OpenSim_Setup\Markers_Coco17.xml`

**Problem:** RHip and LKnee markers had incorrect positions:
- RHip was at (-0.005, -0.398, -0.001) - same as knee position!
- LKnee was at (-0.005, -0.398, -0.001) - same as RHip!

**Fix:** Created corrected marker file with symmetric positions:
```
LHip:  (-0.064, -0.081, -0.105)
RHip:  (-0.064, -0.081, +0.105)  # Mirrored from LHip
LKnee: (-0.005, -0.386, -0.005)  # Mirrored from RKnee
RKnee: (-0.005, -0.386, +0.005)
```

**Backup:** Original saved as `Markers_Coco17_ORIGINAL.xml`

### 2. Coordinate Transformation (src/trc_motionbert.py)
**Problem:** Body-frame rotation was producing incorrect orientations.

**Solution:** Simple axis transformation:
```python
# MotionBERT: X=lateral (left=higher), Y=down, Z=depth
# OpenSim: X=forward, Y=up, Z=right
transformed[..., 0] = -keypoints[..., 2]  # X_osim = -Z_mb (flip forward)
transformed[..., 1] = -keypoints[..., 1]  # Y_osim = -Y_mb (up)
transformed[..., 2] = -keypoints[..., 0]  # Z_osim = -X_mb (right)
```

### 3. H36M to COCO17 Marker Mapping (src/trc_motionbert.py)
**Problem:** Neck marker was placed at Thorax (chest level), causing head issues.

**Fix:** Updated COCO17_TO_H36M mapping:
```python
'Nose': 10,  # Head (top of head) - was 9 (Neck/Nose)
'Neck': 9,   # Neck/Nose (base of skull) - was 8 (Thorax)
```

### 4. Disabled Left/Right Consistency Fix
**File:** `run_motionbert_pipeline.py`
**Change:** `convert_motionbert_to_trc(..., fix_lr_swap=False)`

The automatic L/R flip was interfering with the coordinate transformation.

## Current Pipeline Configuration

### run_motionbert_pipeline.py
```python
INPUT_PKL = "input/aitor_garden_walk_poses.pkl"  # Your pickle file path
OUTPUT_DIR = Path("output/motionbert_opensim")
TARGET_HEIGHT = 1.75  # meters

convert_motionbert_to_trc(INPUT_PKL, str(trc_path), TARGET_HEIGHT, fix_lr_swap=False)
```

### Key Parameters in Pose2Sim config
```python
'pose_model': 'COCO_17',
'use_simple_model': True,
'use_augmentation': False,
```

## Results After Fixes
- Marker RMS error: ~0.09m (was ~0.19m)
- pelvis_list: ~1° (was -48°!)
- pelvis_rotation: ~-22° (was 59°)
- L5_S1_Lat_Bending: ~7° (was 90° at joint limit!)

## Files Modified
1. `src/trc_motionbert.py` - Coordinate transform and marker mapping
2. `run_motionbert_pipeline.py` - Disabled L/R fix
3. `Pose2Sim/OpenSim_Setup/Markers_Coco17.xml` - Fixed marker positions (in conda env)
4. `models/Markers_Coco17_fixed.xml` - Local copy of fixed markers

## Remaining Issue: Inconsistent Segment Lengths

**Problem confirmed:** MotionBERT outputs have highly variable bone lengths:
- Thigh segments: ~16-17% variation (CV)
- Shin segments: ~16-18% variation
- Total skeleton height: **23% variation**

**Cause:** Monocular 3D pose estimation cannot reliably estimate absolute depth.

**Solution needed:** Normalize skeleton to consistent bone lengths before TRC export:
1. Compute mean bone length for each segment across all frames
2. For each frame, scale joint positions to match mean bone lengths
3. This preserves joint angles while fixing segment lengths

**Implementation location:** `src/trc_motionbert.py` - add `normalize_bone_lengths()` function before `transform_motionbert_to_opensim()`

**Reference:** Similar to Pose2Sim's skeleton normalization or OpenSim's scaling approach.

## FIXED: Bone Length Normalization (2026-02-01)

Added `normalize_bone_lengths()` function in `src/trc_motionbert.py`:
- Computes mean bone length for each segment across all frames
- Adjusts each frame to match mean lengths while preserving joint angles
- Uses H36M skeleton hierarchy (pelvis as root, propagates outward)

**Results:**
- Segment length CV: 18.5% → **0.0%** (perfectly consistent)
- IK marker error now consistent across all frames (~0.08m RMS)

**Usage:**
```python
convert_motionbert_to_trc(pkl_path, output_path, target_height, fix_lr_swap=False, normalize_bones=True)
```

## FIXED: Synthetic Nose Position (2026-02-01)

**Problem:** H36M Head and Neck/Nose joints are only 22mm apart, causing extreme neck angles.

**Solution:** Create synthetic nose by projecting 10cm forward from head in facing direction:
1. Compute facing direction from shoulder line and spine
2. Project nose position: `head_pos + forward * 0.10m`

**Results:**
| Angle | Before | After |
|-------|--------|-------|
| neck_bending | -65.75° | **0.41°** |
| neck_rotation | 45.00° | **12.50°** |

**Implementation:** Modified `extract_coco17_markers()` in `src/trc_motionbert.py`
