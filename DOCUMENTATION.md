# MotionBERT-OpenSim: Complete Technical Documentation

**Version:** 1.0
**Date:** 2026-02-01
**Author:** Aitor Iriondo Pascual

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Full Pipeline Architecture](#2-full-pipeline-architecture)
3. [Coordinate Systems](#3-coordinate-systems)
4. [Skeleton Formats](#4-skeleton-formats)
5. [Data Flow and Transformations](#5-data-flow-and-transformations)
6. [Bugs Found and Fixed](#6-bugs-found-and-fixed)
7. [Performance Benchmarks](#7-performance-benchmarks)
8. [Code Reference](#8-code-reference)
9. [Configuration](#9-configuration)
10. [Usage Instructions](#10-usage-instructions)
11. [Troubleshooting](#11-troubleshooting)
12. [File Structure](#12-file-structure)

---

## 1. Project Overview

### 1.1 Purpose

MotionBERT-OpenSim is a pipeline that converts monocular video of human movement into OpenSim-compatible joint angle data (.mot files). This enables biomechanical analysis of human motion captured with a single consumer camera.

### 1.2 Key Features

- **Monocular 3D pose estimation** using MotionBERT transformer model
- **Automatic skeleton scaling** to match participant height
- **Bone length normalization** to fix monocular depth estimation errors
- **Synthetic marker generation** for anatomically correct head tracking
- **Direct integration with Pose2Sim** for OpenSim inverse kinematics

### 1.3 Input/Output

| Input | Output |
|-------|--------|
| Video file (MP4, AVI, etc.) | Scaled OpenSim model (.osim) |
| Participant height (meters) | Joint angle time series (.mot) |
| | Marker trajectories (.trc) |
| | IK error metrics (.sto) |

### 1.4 System Requirements

- **GPU:** NVIDIA GPU with CUDA support (tested on RTX 2080)
- **RAM:** 8GB minimum, 16GB recommended
- **Python:** 3.10+
- **Conda environment:** `motionbert-opensim`

### 1.5 Dependencies

```
numpy>=1.26
opencv-python>=4.8
torch>=2.5 (with CUDA)
torchvision>=0.20
ultralytics>=8.4 (YOLOv8)
Pose2Sim>=0.10
opensim>=4.5
easydict
tqdm
```

---

## 2. Full Pipeline Architecture

### 2.1 Pipeline Overview Diagram

```
                            FULL PIPELINE
    ================================================================

    INPUT: Video File (e.g., walking.mp4)
           |
           v
    +------------------+
    | 1. YOLOv8-Pose   |  Detect 2D keypoints in each frame
    |    (2D Detection)|  Output: COCO17 format (17 joints, x,y,confidence)
    +------------------+
           |
           v
    +------------------+
    | 2. COCO17->H36M  |  Convert keypoint format
    |    Conversion    |  Output: H36M format (17 joints, different order)
    +------------------+
           |
           v
    +------------------+
    | 3. MotionBERT    |  Lift 2D keypoints to 3D
    |    (2D->3D Lift) |  Output: H36M 3D (17 joints, x,y,z in camera coords)
    +------------------+
           |
           v
    +------------------+
    | 4. Bone Length   |  Fix monocular depth inconsistency
    |    Normalization |  Output: Consistent bone lengths across frames
    +------------------+
           |
           v
    +------------------+
    | 5. Coordinate    |  Transform to OpenSim axes
    |    Transform     |  Output: OpenSim coordinates (X=fwd, Y=up, Z=right)
    +------------------+
           |
           v
    +------------------+
    | 6. Marker        |  Map H36M joints to COCO17 markers
    |    Extraction    |  + Create synthetic nose marker
    +------------------+
           |
           v
    +------------------+
    | 7. TRC Export    |  Write marker trajectory file
    |                  |  Output: .trc file (14 markers, meters)
    +------------------+
           |
           v
    +------------------+
    | 8. Pose2Sim      |  Scale model + Inverse Kinematics
    |    Kinematics    |  Output: .osim (scaled model), .mot (joint angles)
    +------------------+
           |
           v
    OUTPUT: OpenSim Files
            - walk_scaled.osim (scaled musculoskeletal model)
            - walk.mot (joint angles over time)
            - walk.trc (marker trajectories)
```

### 2.2 Step-by-Step Explanation

#### Step 1: YOLOv8-Pose 2D Detection

**Location:** `process_video.py`

YOLOv8-Pose is a real-time 2D pose estimation model that detects human keypoints in each video frame.

- **Input:** Video frame (RGB image, e.g., 1920x1080)
- **Output:** 17 keypoints in COCO format with (x, y, confidence) per joint
- **Model:** `yolov8m-pose.pt` (medium variant, good speed/accuracy balance)

**COCO17 Joint Order:**
```
0: Nose
1: Left Eye
2: Right Eye
3: Left Ear
4: Right Ear
5: Left Shoulder
6: Right Shoulder
7: Left Elbow
8: Right Elbow
9: Left Wrist
10: Right Wrist
11: Left Hip
12: Right Hip
13: Left Knee
14: Right Knee
15: Left Ankle
16: Right Ankle
```

#### Step 2: COCO17 to H36M Conversion

**Location:** `utils/keypoint_converter.py`

Human3.6M (H36M) is the skeleton format used by MotionBERT. This step converts COCO17 keypoints to H36M format.

**Key differences:**
- Different joint ordering
- H36M has torso joints (Spine, Thorax) that COCO doesn't have
- H36M Pelvis is computed as midpoint of hips
- H36M Spine/Thorax are interpolated from hips and shoulders

**H36M Joint Order:**
```
0: Hip (Pelvis center)
1: RHip
2: RKnee
3: RFoot (Ankle)
4: LHip
5: LKnee
6: LFoot (Ankle)
7: Spine
8: Thorax
9: Neck/Nose
10: Head
11: LShoulder
12: LElbow
13: LWrist
14: RShoulder
15: RElbow
16: RWrist
```

#### Step 3: MotionBERT 2D-to-3D Lifting

**Location:** `MotionBERT/`

MotionBERT is a transformer-based model that lifts 2D keypoints to 3D. It uses temporal context (243 frames) to estimate depth.

- **Input:** Normalized 2D keypoints (x, y, confidence) in [-1, 1] range
- **Output:** 3D keypoints (x, y, z) in camera-relative coordinates
- **Model:** `FT_MB_lite_MB_ft_h36m_global_lite` (lightweight variant)
- **Checkpoint:** `checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin`

**MotionBERT Coordinate System:**
```
X = horizontal (left-right in image)
Y = vertical down (positive toward ground)
Z = depth (positive away from camera)
```

**Important:** MotionBERT uses test-time augmentation (flip augmentation) for better accuracy. The model runs twice (original + horizontally flipped) and averages the results.

#### Step 4: Bone Length Normalization

**Location:** `src/trc_motionbert.py` - `normalize_bone_lengths()`

Monocular 3D pose estimation cannot reliably estimate absolute depth, causing bone lengths to vary significantly between frames (up to 23% height variation observed).

**Problem:**
- Frame 1: Thigh length = 0.42m
- Frame 100: Thigh length = 0.38m
- Frame 500: Thigh length = 0.45m

**Solution:**
1. Compute mean bone length for each segment across all frames
2. For each frame, adjust joint positions to match mean bone lengths
3. Preserve joint angles while fixing segment proportions

**Algorithm:**
```python
# For each bone in skeleton hierarchy (parent -> child):
for frame in all_frames:
    direction = child_pos - parent_pos
    current_length = norm(direction)
    direction = direction / current_length  # Normalize
    child_pos = parent_pos + direction * mean_length  # Scale to mean
```

**H36M Skeleton Hierarchy:**
```
Pelvis (0)
├── RHip (1) -> RKnee (2) -> RAnkle (3)
├── LHip (4) -> LKnee (5) -> LAnkle (6)
└── Spine (7) -> Thorax (8)
                 ├── Neck (9) -> Head (10)
                 ├── LShoulder (11) -> LElbow (12) -> LWrist (13)
                 └── RShoulder (14) -> RElbow (15) -> RWrist (16)
```

**Results:**
- Before: Mean bone length CV = 18.5%
- After: Mean bone length CV = 0.0% (perfectly consistent)

#### Step 5: Coordinate Transformation

**Location:** `src/trc_motionbert.py` - `transform_motionbert_to_opensim()`

OpenSim uses a different coordinate system than MotionBERT. This step transforms the keypoints.

**MotionBERT Coordinates (camera-centric):**
```
X = horizontal (left has higher X when subject faces away)
Y = down (positive toward ground)
Z = depth (positive away from camera)
```

**OpenSim Coordinates (anatomical):**
```
X = anterior/forward (positive in walking direction)
Y = superior/up (positive toward head)
Z = lateral/right (positive toward subject's right side)
```

**Transformation (Simple Axis Mapping):**
```python
# Subject facing away from camera, walking in +Z direction
# But we want skeleton walking in +X direction in OpenSim

X_opensim = -Z_motionbert  # Forward (negated to flip walking direction)
Y_opensim = -Y_motionbert  # Up (negated because Y_mb points down)
Z_opensim = -X_motionbert  # Right (negated to preserve L/R after X flip)
```

**Additional Processing:**
1. Center pelvis at Z=0 (lateral centering)
2. Ground feet at Y=0 (put feet on floor)
3. Scale to target height (e.g., 1.75m)

#### Step 6: Marker Extraction (H36M to COCO17)

**Location:** `src/trc_motionbert.py` - `extract_coco17_markers()`

Pose2Sim expects COCO17 marker format (14 markers, no eyes/ears). This step extracts the relevant markers from H36M joints.

**COCO17 to H36M Mapping:**
```python
COCO17_TO_H36M = {
    'Nose': 10,       # H36M Head (synthetic, see below)
    'Neck': 9,        # H36M Neck/Nose (base of skull)
    'LShoulder': 11,
    'RShoulder': 14,
    'LElbow': 12,
    'RElbow': 15,
    'LWrist': 13,
    'RWrist': 16,
    'LHip': 4,
    'RHip': 1,
    'LKnee': 5,
    'RKnee': 2,
    'LAnkle': 6,
    'RAnkle': 3,
}
```

**Synthetic Nose Marker:**

H36M doesn't have a face marker. The Head joint (10) is at the top of the skull, and Neck/Nose (9) is at the base of the skull. These are only ~22mm apart, which caused extreme neck angles in IK.

**Solution:** Create a synthetic nose position by projecting 10cm forward from the head in the facing direction:

```python
# Compute facing direction from shoulder line and spine
lateral = l_shoulder - r_shoulder  # Points to subject's left
up = neck - thorax                  # Points up
forward = cross(lateral, up)        # Points forward

# Project nose 10cm forward from head
nose_pos = head_pos + forward * 0.10
```

**Results:**
- Before synthetic nose: neck_bending = -65.75 degrees
- After synthetic nose: neck_bending = 0.41 degrees

#### Step 7: TRC Export

**Location:** `src/trc_motionbert.py` - `export_trc()`

TRC (Track Row Column) is a standard format for marker trajectories used by OpenSim.

**TRC File Format:**
```
Line 1: PathFileType  4  (X/Y/Z)  filename.trc
Line 2: DataRate  CameraRate  NumFrames  NumMarkers  Units  ...
Line 3: Frame#  Time  Marker1      Marker2      ...
Line 4:              X1  Y1  Z1   X2  Y2  Z2   ...
Line 5: (empty)
Line 6+: frame_num  time  x1  y1  z1  x2  y2  z2  ...
```

**Example TRC Header:**
```
PathFileType	4	(X/Y/Z)	walk.trc
30.00	30.00	1136	14	m	30.00	1	1136
Frame#	Time	Nose			Neck			LShoulder	...
		X1	Y1	Z1	X2	Y2	Z2	X3	Y3	Z3	...

1	0.000000	-0.123456	1.654321	0.012345	...
2	0.033333	-0.123789	1.654654	0.012678	...
```

**Important:** Units must be meters (`m`), not millimeters.

#### Step 8: Pose2Sim Kinematics

**Location:** `run_motionbert_pipeline.py`

Pose2Sim uses OpenSim to perform model scaling and inverse kinematics.

**Scaling:**
1. Load generic model (`Model_Pose2Sim_simple.osim`)
2. Load marker positions from TRC file
3. Scale body segments to match marker distances
4. Save scaled model (`*_scaled.osim`)

**Inverse Kinematics (IK):**
1. Load scaled model
2. For each frame, find joint angles that minimize marker error
3. Save joint angles to MOT file

**Key Configuration:**
```python
config_dict = {
    'project': {
        'participant_height': 1.75,  # Target height
        'participant_mass': 70.0,
    },
    'pose': {
        'pose_model': 'COCO_17',  # Marker format
    },
    'kinematics': {
        'use_augmentation': False,  # Don't use marker augmentation
        'use_simple_model': True,   # Use simplified musculoskeletal model
        'right_left_symmetry': True,
    },
}
```

---

## 3. Coordinate Systems

### 3.1 MotionBERT Coordinate System

MotionBERT outputs keypoints in a camera-centric coordinate system:

```
        Camera View
        -----------
             ^ -Y (up in image)
             |
             |
    <--------+--------> +X (right in image)
             |
             | +Y (down in image)
             v

        +Z points away from camera (depth)
```

**Properties:**
- Origin: Camera optical center
- X-axis: Points right in the image
- Y-axis: Points DOWN in the image (standard image coordinates)
- Z-axis: Points away from camera (depth into scene)

### 3.2 OpenSim Coordinate System

OpenSim uses an anatomical coordinate system:

```
        OpenSim View (Standing Person)
        ------------------------------
             ^ +Y (superior/up)
             |
             |
    <--------+--------> +Z (lateral right)
             |
             | -Y (inferior/down)
             v

        +X points forward (anterior)
```

**Properties:**
- Origin: Ground plane, centered on pelvis
- X-axis: Anterior (forward, walking direction)
- Y-axis: Superior (up, toward head)
- Z-axis: Lateral (right, toward subject's right side)

### 3.3 Transformation Details

The transformation from MotionBERT to OpenSim coordinates depends on the camera setup. For a side-view video where the subject walks perpendicular to the camera:

**Observed MotionBERT output:**
- Subject walks in -Z direction (toward camera initially, then away)
- Subject's left side has higher X values
- Y increases downward

**Required transformation:**
```python
# To get subject walking in +X (OpenSim forward):
X_opensim = -Z_motionbert  # Flip Z to get correct walking direction
Y_opensim = -Y_motionbert  # Flip Y (down -> up)
Z_opensim = -X_motionbert  # Flip X to maintain correct L/R after X flip
```

**Why all axes are negated:**
1. `-Z_mb`: Subject was walking in -Z, we want +X forward
2. `-Y_mb`: MotionBERT Y points down, OpenSim Y points up
3. `-X_mb`: When we flip X->Z mapping, we need to negate to keep left/right correct

---

## 4. Skeleton Formats

### 4.1 COCO17 Format

COCO (Common Objects in Context) 17-keypoint format:

```
Index  Joint Name
-----  ----------
0      Nose
1      Left Eye
2      Right Eye
3      Left Ear
4      Right Ear
5      Left Shoulder
6      Right Shoulder
7      Left Elbow
8      Right Elbow
9      Left Wrist
10     Right Wrist
11     Left Hip
12     Right Hip
13     Left Knee
14     Right Knee
15     Left Ankle
16     Right Ankle
```

**Skeleton Connections:**
```
Nose -- LEye -- LEar
  |  \
  |   REye -- REar
  |
Neck (virtual, between shoulders)
  |
LShoulder ---- RShoulder
  |                |
LElbow          RElbow
  |                |
LWrist          RWrist
  |                |
LHip ---------- RHip
  |                |
LKnee           RKnee
  |                |
LAnkle          RAnkle
```

### 4.2 H36M (Human3.6M) Format

H36M 17-keypoint format used by MotionBERT:

```
Index  Joint Name     Description
-----  ----------     -----------
0      Hip            Pelvis center (midpoint of hips)
1      RHip           Right hip joint
2      RKnee          Right knee joint
3      RFoot          Right ankle
4      LHip           Left hip joint
5      LKnee          Left knee joint
6      LFoot          Left ankle
7      Spine          Lower spine (between pelvis and thorax)
8      Thorax         Chest (between shoulders)
9      Neck/Nose      Base of skull / upper neck
10     Head           Top of head
11     LShoulder      Left shoulder joint
12     LElbow         Left elbow joint
13     LWrist         Left wrist joint
14     RShoulder      Right shoulder joint
15     RElbow         Right elbow joint
16     RWrist         Right wrist joint
```

**Skeleton Hierarchy (Parent -> Child):**
```
Hip (0) [ROOT]
├── RHip (1)
│   └── RKnee (2)
│       └── RFoot (3)
├── LHip (4)
│   └── LKnee (5)
│       └── LFoot (6)
└── Spine (7)
    └── Thorax (8)
        ├── Neck/Nose (9)
        │   └── Head (10)
        ├── LShoulder (11)
        │   └── LElbow (12)
        │       └── LWrist (13)
        └── RShoulder (14)
            └── RElbow (15)
                └── RWrist (16)
```

### 4.3 Pose2Sim COCO17 Markers

Pose2Sim uses a subset of COCO17 (14 markers, no eyes/ears):

```
Index  Marker Name
-----  -----------
0      Nose
1      Neck
2      LShoulder
3      RShoulder
4      LElbow
5      RElbow
6      LWrist
7      RWrist
8      LHip
9      RHip
10     LKnee
11     RKnee
12     LAnkle
13     RAnkle
```

These markers are defined in `Markers_Coco17.xml` and placed on the OpenSim model.

---

## 5. Data Flow and Transformations

### 5.1 Data Shapes at Each Stage

| Stage | Shape | Units | Format |
|-------|-------|-------|--------|
| Video frame | (H, W, 3) | uint8 | BGR image |
| YOLOv8 keypoints | (17, 2) | pixels | COCO17 xy |
| YOLOv8 scores | (17,) | 0-1 | confidence |
| H36M 2D keypoints | (17, 2) | normalized | H36M xy |
| MotionBERT 3D | (T, 17, 3) | arbitrary | H36M xyz |
| Normalized 3D | (T, 17, 3) | arbitrary | H36M xyz (fixed bones) |
| OpenSim coords | (T, 17, 3) | meters | H36M xyz (OpenSim axes) |
| COCO17 markers | (T, 14, 3) | meters | Pose2Sim markers |
| TRC file | text | meters | Frame, Time, X1,Y1,Z1,... |

### 5.2 Transformation Pipeline Code

```python
# Step 1: Load video and detect 2D poses
for frame in video:
    keypoints_coco, scores = yolo_model.detect(frame)  # (17, 2), (17,)

# Step 2: Convert to H36M
keypoints_h36m = coco_to_h36m(keypoints_coco)  # (17, 2)

# Step 3: Lift to 3D
keypoints_3d = motionbert.lift(keypoints_h36m, scores)  # (T, 17, 3)

# Step 4: Normalize bone lengths
keypoints_3d = normalize_bone_lengths(keypoints_3d)  # (T, 17, 3)

# Step 5: Transform to OpenSim coordinates
keypoints_osim = transform_motionbert_to_opensim(keypoints_3d)  # (T, 17, 3)

# Step 6: Extract COCO17 markers
markers = extract_coco17_markers(keypoints_osim)  # (T, 14, 3)

# Step 7: Export TRC
export_trc(markers, output_path)

# Step 8: Run Pose2Sim IK
Pose2Sim.kinematics(config_dict)
```

---

## 6. Bugs Found and Fixed

### 6.1 CRITICAL: Pose2Sim Markers_Coco17.xml Bug

**Location:** `C:\Users\aitor\anaconda3\envs\rtmpose3d-opensim\lib\site-packages\Pose2Sim\OpenSim_Setup\Markers_Coco17.xml`

**Problem:** The original Pose2Sim marker file had incorrect positions for RHip and LKnee markers:

```xml
<!-- ORIGINAL (WRONG) -->
<Marker name="RHip">
    <location>-0.00541048 -0.397886 -0.000611877</location>  <!-- WRONG! Same as knee -->
</Marker>
<Marker name="LKnee">
    <location>-0.00541048 -0.397886 -0.000611877</location>  <!-- WRONG! Same as RHip -->
</Marker>
```

**Impact:** This caused severe skeleton distortion:
- pelvis_list: -48 degrees (should be ~0)
- L5_S1_Lat_Bending: 90 degrees (at joint limit!)
- Marker RMS error: ~0.19m (very high)

**Fix:** Created corrected marker positions with proper symmetry:

```xml
<!-- FIXED (CORRECT) -->
<Marker name="LHip">
    <location>-0.06392744445800781 -0.08134311294555664 -0.10540640790244724</location>
</Marker>
<Marker name="RHip">
    <location>-0.06392744445800781 -0.08134311294555664 0.10540640790244724</location>
    <!-- Note: Z is positive for right side (symmetric with LHip) -->
</Marker>
<Marker name="LKnee">
    <location>-0.005410484544778264 -0.3861321573680546 -0.005110696942696956</location>
    <!-- Note: Different position from hip (was same before!) -->
</Marker>
<Marker name="RKnee">
    <location>-0.005410484544778264 -0.3861321573680546 0.005110696942696956</location>
</Marker>
```

**Backup:** Original saved as `Markers_Coco17_ORIGINAL.xml`

### 6.2 Coordinate Transformation Issues

**Problem 1: Skeleton upside down**

Initial body-frame rotation gave inverted skeleton due to cross product order.

**Fix:** Changed cross product order in `compute_body_frame_rotation()`:
```python
# WRONG: body_z = np.cross(body_x, body_y)  # Gives down instead of up
# CORRECT:
body_z = np.cross(body_y, body_x)  # Gives up
```

**Problem 2: Hip and torso facing opposite directions**

Body-frame rotation was computing inconsistent orientations for different body parts.

**Fix:** Switched to simple axis transformation which is more robust:
```python
transformed[..., 0] = -keypoints[..., 2]  # X_osim = -Z_mb
transformed[..., 1] = -keypoints[..., 1]  # Y_osim = -Y_mb
transformed[..., 2] = -keypoints[..., 0]  # Z_osim = -X_mb
```

**Problem 3: Skeleton walking backwards**

Initial transform had subject walking in wrong direction.

**Fix:** Negated X output to flip walking direction:
```python
# Before: X_osim = Z_mb (walking backward)
# After:  X_osim = -Z_mb (walking forward)
```

**Problem 4: Left/right swapped after forward flip**

Changing X without adjusting Z broke the coordinate system handedness.

**Fix:** Keep Z negated to maintain correct L/R:
```python
Z_osim = -X_mb  # Keep negated (not X_mb)
```

### 6.3 H36M to COCO17 Marker Mapping Error

**Problem:** Neck marker was placed at Thorax (chest level) instead of actual neck.

**Original mapping:**
```python
'Nose': 9,   # Neck/Nose - WRONG (this is base of skull)
'Neck': 8,   # Thorax - WRONG (this is chest!)
```

**Fixed mapping:**
```python
'Nose': 10,  # Head (top of head) - closest to face
'Neck': 9,   # Neck/Nose (base of skull) - actual neck
```

### 6.4 Inconsistent Bone Lengths (Monocular Depth Issue)

**Problem:** MotionBERT monocular estimation produced highly variable bone lengths:
- Thigh segment CV: ~17%
- Shin segment CV: ~17%
- Total skeleton height variation: 23%

**Cause:** Monocular 3D pose estimation cannot reliably estimate absolute depth, causing "rubber skeleton" effect.

**Fix:** Implemented `normalize_bone_lengths()` function:
1. Compute mean bone length for each segment across all frames
2. Adjust each frame to match mean lengths while preserving joint angles
3. Process in skeleton hierarchy order (parents before children)

**Results:**
- Before: Mean bone length CV = 18.5%
- After: Mean bone length CV = 0.0%

### 6.5 Head/Neck Extreme Angles

**Problem:** H36M Head and Neck/Nose joints are only ~22mm apart, causing extreme neck angles in IK:
- neck_bending: -65.75 degrees (should be near 0)
- neck_rotation: 45.00 degrees (unrealistic)

**Cause:** The small distance between Head and Neck markers gave IK very little constraint on head orientation.

**Fix:** Created synthetic nose position by projecting 10cm forward from head:

```python
NOSE_FORWARD_OFFSET = 0.10  # 10cm

# Compute facing direction
lateral = l_shoulder - r_shoulder
up = neck - thorax
forward = np.cross(lateral, up)
forward = forward / np.linalg.norm(forward)

# Synthetic nose position
nose_pos = head_pos + forward * NOSE_FORWARD_OFFSET
```

**Results:**
| Angle | Before | After |
|-------|--------|-------|
| neck_bending | -65.75 deg | 0.41 deg |
| neck_rotation | 45.00 deg | 12.50 deg |
| neck_flexion | -28.65 deg | -24.73 deg |
| Nose-Neck distance | ~2cm | 11.2cm |

### 6.6 Left/Right Swap Interference

**Problem:** The automatic L/R consistency fix was interfering with the coordinate transformation, causing additional issues.

**Fix:** Disabled L/R fix when using simple transform:
```python
convert_motionbert_to_trc(pkl_path, output_path, target_height,
                          fix_lr_swap=False,  # Disabled
                          normalize_bones=True)
```

---

## 7. Performance Benchmarks

### 7.1 Test Configuration

- **GPU:** NVIDIA GeForce RTX 2080
- **CPU:** (used for OpenSim IK)
- **Video:** 1920x1080, 30 FPS, 1136 frames (37.87 seconds)
- **Participant height:** 1.75m

### 7.2 Timing Results

| Step | Component | Time | % of Total |
|------|-----------|------|------------|
| 1a | YOLOv8 model load | 6.57 s | 8.7% |
| 1b | YOLOv8 inference | 25.49 s | 33.7% |
| **1** | **YOLOv8 Total** | **32.07 s** | **42.3%** |
| 2 | COCO17 to H36M | 631 ms | 0.8% |
| 3a | MotionBERT load | 482 ms | 0.6% |
| 3b | MotionBERT inference | 891 ms | 1.2% |
| **3** | **MotionBERT Total** | **1.37 s** | **1.8%** |
| 4 | Bone normalization | 167 ms | 0.2% |
| 5 | Coordinate transform | 0.3 ms | 0.0% |
| 6 | Extract markers | 68 ms | 0.1% |
| 7 | TRC export | 52 ms | 0.1% |
| **8** | **Pose2Sim/OpenSim IK** | **7.93 s** | **10.5%** |
| | **TOTAL** | **1.3 min** | **100%** |

### 7.3 By Component

| Component | Time | % of Total |
|-----------|------|------------|
| VideoPoseEstimation (Steps 1-3) | 34.07 s | 45% |
| MotionBERT-OpenSim (Steps 4-8) | 8.22 s | 11% |
| (Model loading) | ~7 s | ~9% |

### 7.4 Throughput Metrics

| Stage | Throughput | Real-time Factor |
|-------|------------|------------------|
| VideoPoseEstimation | 33 FPS | 1.1x real-time |
| RTMPose3DOpenSim | 138 FPS | 4.6x real-time |
| **End-to-end** | **15 FPS** | **0.5x real-time** |

### 7.5 Bottleneck Analysis

1. **YOLOv8 2D detection** (42.3%) - Main bottleneck, runs at ~45 FPS on RTX 2080
2. **Pose2Sim/OpenSim IK** (10.5%) - Second bottleneck, CPU-bound
3. **Model loading** (~9%) - One-time cost, amortized over longer videos

### 7.6 Optimization Recommendations

To achieve real-time performance:
1. Use YOLOv8-nano instead of YOLOv8-medium (~2x faster, slightly less accurate)
2. Batch video frames for GPU processing
3. Use OpenSim's batch IK mode
4. Pre-load models (avoid loading time in pipeline)

---

## 8. Code Reference

### 8.1 Main Files

| File | Purpose |
|------|---------|
| `src/trc_motionbert.py` | Core conversion functions (H36M -> TRC) |
| `run_motionbert_pipeline.py` | Run conversion + Pose2Sim |
| `run_full_pipeline_timed.py` | Full video-to-OpenSim with timing |
| `run_pipeline_timed.py` | Pickle-to-OpenSim with timing |

### 8.2 Key Functions in `src/trc_motionbert.py`

```python
def load_motionbert_keypoints(pkl_path: str) -> Tuple[np.ndarray, float, dict]:
    """Load H36M keypoints from MotionBERT pickle file."""

def normalize_bone_lengths(keypoints: np.ndarray) -> np.ndarray:
    """Normalize skeleton to consistent bone lengths across frames."""

def transform_motionbert_to_opensim(keypoints: np.ndarray,
                                    target_height: float = 1.75,
                                    use_simple_transform: bool = True) -> np.ndarray:
    """Transform MotionBERT coordinates to OpenSim coordinate system."""

def extract_coco17_markers(h36m_keypoints: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Extract COCO17 markers from H36M keypoints (with synthetic nose)."""

def export_trc(marker_positions: np.ndarray,
               marker_names: List[str],
               output_path: str,
               fps: float = 30.0) -> str:
    """Export markers to TRC format for Pose2Sim."""

def convert_motionbert_to_trc(pkl_path: str,
                              output_path: str,
                              target_height: float = 1.75,
                              fix_lr_swap: bool = True,
                              normalize_bones: bool = True) -> str:
    """Full pipeline: Load pickle, transform, export TRC."""
```

### 8.3 Function Call Graph

```
convert_motionbert_to_trc()
├── load_motionbert_keypoints()
├── fix_left_right_consistency()  [optional]
├── normalize_bone_lengths()
├── transform_motionbert_to_opensim()
│   └── compute_body_frame_rotation()  [if not simple transform]
├── extract_coco17_markers()
└── export_trc()
```

---

## 9. Configuration

### 9.1 Pipeline Parameters

```python
# In run_motionbert_pipeline.py:

INPUT_PKL = "input/aitor_garden_walk_poses.pkl"  # Or path to your pickle file
OUTPUT_DIR = Path("output/motionbert_opensim")
TARGET_HEIGHT = 1.75  # meters

convert_motionbert_to_trc(
    pkl_path=INPUT_PKL,
    output_path=str(trc_path),
    target_height=TARGET_HEIGHT,
    fix_lr_swap=False,      # Disabled - interferes with transform
    normalize_bones=True    # Enabled - fixes monocular depth issues
)
```

### 9.2 Pose2Sim Configuration

```python
config_dict = {
    'project': {
        'project_dir': str(OUTPUT_DIR),
        'multi_person': False,
        'participant_height': TARGET_HEIGHT,
        'participant_mass': 70.0,
        'frame_rate': 'auto',
        'frame_range': 'all',
    },
    'pose': {
        'pose_model': 'COCO_17',  # Must match marker file
    },
    'markerAugmentation': {
        'feet_on_floor': False,
        'make_c3d': False,
    },
    'kinematics': {
        'use_augmentation': False,     # Don't use additional markers
        'use_simple_model': True,      # Use Model_Pose2Sim_simple.osim
        'right_left_symmetry': True,   # Assume symmetric body
        'default_height': TARGET_HEIGHT,
        'remove_individual_scaling_setup': False,
        'remove_individual_ik_setup': False,
        'fastest_frames_to_remove_percent': 0.1,
        'close_to_zero_speed_m': 0.2,
        'large_hip_knee_angles': 45,
        'trimmed_extrema_percent': 0.5,
    },
    'logging': {
        'use_custom_logging': False,
    },
}
```

### 9.3 OpenSim Model

**Model used:** `Model_Pose2Sim_simple.osim`

**Location:** `C:\Users\aitor\anaconda3\envs\rtmpose3d-opensim\lib\site-packages\Pose2Sim\OpenSim_Setup\Model_Pose2Sim_simple.osim`

**Properties:**
- 40 degrees of freedom (coordinates)
- Simplified musculoskeletal model (no muscles)
- Compatible with COCO17 markers

### 9.4 Marker Definition File

**File:** `Markers_Coco17.xml` (MODIFIED - see Bug 6.1)

**Location:** `C:\Users\aitor\anaconda3\envs\rtmpose3d-opensim\lib\site-packages\Pose2Sim\OpenSim_Setup\Markers_Coco17.xml`

---

## 10. Usage Instructions

### 10.1 Environment Setup

```bash
# Navigate to project directory and activate virtual environment
cd C:\MotionBERTOpenSim\MotionBERT-OpenSim
venv\Scripts\activate

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "import Pose2Sim; print('Pose2Sim: OK')"
```

### 10.2 Running the Full Pipeline (Video to OpenSim)

#### Command-Line Arguments

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--input` | `-i` | Yes | - | Input video path (MP4, AVI, etc.) |
| `--output` | `-o` | No | Auto | Output directory. If not specified, auto-generates as `output_YYYYMMDD_HHMMSS_videoname` |
| `--height` | - | No | 1.75 | Target participant height in meters |
| `--device` | `-d` | No | cuda:0 | Computation device (`cuda:0` for GPU, `cpu` for CPU) |
| `--filter` | `-f` | No | Off | Enable Butterworth low-pass filter to smooth keypoints and reduce jitter |
| `--filter-cutoff` | - | No | 6.0 | Filter cutoff frequency in Hz. Lower values = more smoothing. Range: 4-10 Hz recommended |
| `--no-timing` | - | No | Off | Disable timing output in console |

#### Usage Examples

```bash
cd C:\MotionBERTOpenSim\MotionBERT-OpenSim

# Basic usage (auto-generate output folder)
python run_full_pipeline.py --input video.mp4 --height 1.75

# Specify custom output folder
python run_full_pipeline.py --input video.mp4 --output results/my_output --height 1.75

# With Butterworth smoothing filter (recommended for cleaner motion)
python run_full_pipeline.py --input video.mp4 --height 1.75 --filter

# With custom filter cutoff (lower = more smoothing)
python run_full_pipeline.py --input video.mp4 --height 1.75 --filter --filter-cutoff 4.0

# Use CPU instead of GPU
python run_full_pipeline.py --input video.mp4 --height 1.75 --device cpu

# Full example with all options
python run_full_pipeline.py \
    --input input/walking_video.mp4 \
    --output output/walking_analysis \
    --height 1.65 \
    --device cuda:0 \
    --filter \
    --filter-cutoff 6.0
```

#### Filter Recommendations

| Movement Type | Recommended Cutoff | Notes |
|---------------|-------------------|-------|
| Slow walking | 4-6 Hz | More smoothing acceptable |
| Normal walking | 6 Hz | Default, good balance |
| Fast walking/jogging | 8-10 Hz | Preserve faster movements |
| Sports/dynamic | 10+ Hz | Minimal smoothing |

### 10.3 Running from Existing Pickle (MotionBERT Output to OpenSim)

```bash
cd E:\MotionBERT-OpenSim

python run_motionbert_pipeline.py
# Edit the script to change INPUT_PKL and TARGET_HEIGHT
```

Or use the timed version:
```bash
python run_pipeline_timed.py
```

### 10.4 Viewing Results in OpenSim

1. Open OpenSim GUI
2. File -> Open Model -> Select `output/*/kinematics/*_scaled.osim`
3. File -> Load Motion -> Select `output/*/kinematics/*.mot`
4. Click Play to view animation

### 10.5 Analyzing IK Errors

The IK marker errors are saved in `*_ik_marker_errors.sto`. Lower values indicate better fit.

```python
import pandas as pd
errors = pd.read_csv('output/kinematics/_ik_marker_errors.sto',
                     sep='\t', skiprows=6)
print(f"Mean marker error: {errors['marker_error_RMS'].mean():.4f} m")
```

---

## 11. Troubleshooting

### 11.1 Common Issues

**Issue: "ModuleNotFoundError: No module named 'ultralytics'"**
```bash
pip install ultralytics
```

**Issue: "ModuleNotFoundError: No module named 'Pose2Sim'"**
```bash
pip install pose2sim
```

**Issue: CUDA out of memory**
- Use `--device cpu` flag
- Or use smaller YOLOv8 model (yolov8n-pose.pt)

**Issue: Skeleton appears distorted/tilted**
- Check that `Markers_Coco17.xml` has the fixed marker positions
- Verify the fix with: compare LHip Z vs RHip Z (should be opposite signs)

**Issue: Extreme neck/head angles**
- Verify synthetic nose is being created (check TRC file, Nose should be ~10cm from Neck in X)

**Issue: Inconsistent scaling between frames**
- Ensure `normalize_bones=True` in `convert_motionbert_to_trc()`

**Issue: Skeleton walking backwards**
- The coordinate transform should use `-Z_mb` for `X_osim`

### 11.2 Validation Checks

**Check 1: Bone length consistency**
```python
# After normalization, all bone lengths should have 0% CV
# Look for output: "Mean bone length CV after: 0.0%"
```

**Check 2: Coordinate ranges in TRC**
```python
# Y values should all be positive (above ground)
# X values should span the walking distance
# Z values should be centered around 0
```

**Check 3: Marker error in IK**
```python
# RMS marker error should be < 0.1m for good fit
# If > 0.15m, there may be issues with markers or scaling
```

### 11.3 Debug Mode

Add print statements to see intermediate values:

```python
# In transform_motionbert_to_opensim():
print(f"  X range: [{transformed[..., 0].min():.3f}, {transformed[..., 0].max():.3f}]")
print(f"  Y range: [{transformed[..., 1].min():.3f}, {transformed[..., 1].max():.3f}]")
print(f"  Z range: [{transformed[..., 2].min():.3f}, {transformed[..., 2].max():.3f}]")
```

---

## 12. File Structure

### 12.1 Project Directory

```
E:\MotionBERT-OpenSim\
├── src/
│   ├── __init__.py
│   ├── trc_motionbert.py        # Main conversion code
│   ├── trc_pose2sim.py          # Alternative pipeline (HALPE_26)
│   ├── trc_export.py            # Rajagopal markers (unused)
│   ├── pose_extraction.py       # RTMW3D extraction (unused)
│   └── constants.py             # Constants (unused)
├── models/
│   ├── Model_Pose2Sim_simple.osim
│   ├── Markers_Coco17_fixed.xml # Local copy of fixed markers
│   └── ...
├── output/
│   ├── motionbert_opensim/
│   │   ├── walk.trc
│   │   ├── pose-3d/
│   │   │   └── walk.trc
│   │   └── kinematics/
│   │       ├── walk.mot          # Joint angles
│   │       ├── walk_scaled.osim  # Scaled model
│   │       ├── opensim_logs.txt
│   │       └── _ik_marker_errors.sto
│   └── full_pipeline_timed/
│       └── ...
├── run_motionbert_pipeline.py   # Main pipeline script
├── run_pipeline_timed.py        # Timed pipeline (pickle -> OpenSim)
├── run_full_pipeline_timed.py   # Full timed pipeline (video -> OpenSim)
├── CHANGES_SUMMARY.md           # Quick summary of changes
├── DOCUMENTATION.md             # This file
└── README.md
```

### 12.2 VideoPoseEstimation Directory

```
E:\MotionBERT-OpenSim\
├── input/
│   └── aitor_garden_walk.mp4    # Source video
├── output_fixed/
│   ├── aitor_garden_walk_poses.pkl  # MotionBERT output
│   └── aitor_garden_walk_poses.json
├── MotionBERT/
│   ├── checkpoint/
│   │   └── pose3d/
│   │       └── FT_MB_lite_MB_ft_h36m_global_lite/
│   │           └── best_epoch.bin
│   ├── configs/
│   │   └── pose3d/
│   │       └── MB_ft_h36m_global_lite.yaml
│   └── lib/
│       └── model/
│           └── DSTformer.py
├── utils/
│   └── keypoint_converter.py    # COCO17 <-> H36M conversion
└── process_video_fixed.py       # Video processing script
```

### 12.3 Pose2Sim Installation Directory

```
C:\Users\aitor\anaconda3\envs\rtmpose3d-opensim\lib\site-packages\Pose2Sim\
├── OpenSim_Setup/
│   ├── Model_Pose2Sim_simple.osim
│   ├── Markers_Coco17.xml           # MODIFIED (fixed)
│   ├── Markers_Coco17_ORIGINAL.xml  # Backup of original (buggy)
│   └── ...
└── ...
```

---

## Appendix A: Mathematical Details

### A.1 Bone Length Normalization Algorithm

Given keypoints `K` of shape `(T, J, 3)` where `T` = frames, `J` = joints:

```
For each bone (parent_idx, child_idx) in skeleton hierarchy:
    # Compute mean length across all frames
    lengths = []
    for t in 0..T:
        length = ||K[t, child] - K[t, parent]||
        lengths.append(length)
    mean_length = mean(lengths)

    # Normalize each frame
    for t in 0..T:
        direction = K[t, child] - K[t, parent]
        current_length = ||direction||
        if current_length > epsilon:
            direction = direction / current_length  # Unit vector
            K[t, child] = K[t, parent] + direction * mean_length
```

### A.2 Synthetic Nose Computation

```
For each frame t:
    # Get relevant joints
    l_shoulder = K[t, 11]  # LShoulder
    r_shoulder = K[t, 14]  # RShoulder
    thorax = K[t, 8]       # Thorax
    neck = K[t, 9]         # Neck
    head = K[t, 10]        # Head

    # Compute facing direction
    lateral = l_shoulder - r_shoulder      # Points to subject's left
    lateral = lateral / ||lateral||

    up = neck - thorax                     # Points up
    up = up / ||up||

    forward = cross(lateral, up)           # Points forward
    forward = forward / ||forward||

    # Synthetic nose position
    nose = head + forward * 0.10           # 10cm forward from head
```

### A.3 Coordinate Transformation Matrix

The transformation from MotionBERT (MB) to OpenSim (OS) coordinates can be expressed as a matrix:

```
[X_os]   [-1  0  0] [Z_mb]
[Y_os] = [ 0 -1  0] [Y_mb]
[Z_os]   [ 0  0 -1] [X_mb]

Or equivalently:
X_os = -Z_mb
Y_os = -Y_mb
Z_os = -X_mb
```

This is a combination of:
1. Axis permutation: (X,Y,Z)_mb -> (Z,Y,X)_os
2. Sign flip on all axes (180° rotation around the new Y axis + Y inversion)

---

## Appendix B: Output File Formats

### B.1 TRC File Format

```
PathFileType	4	(X/Y/Z)	walk.trc
DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames
30.00	30.00	1136	14	m	30.00	1	1136
Frame#	Time	Nose			Neck			LShoulder		...
		X1	Y1	Z1	X2	Y2	Z2	X3	Y3	Z3	...

1	0.000000	-0.115234	1.854321	0.012345	-0.098765	1.743210	0.011111	...
2	0.033333	-0.115567	1.854654	0.012678	-0.099098	1.743543	0.011444	...
...
```

### B.2 MOT File Format

```
Coordinates
version=1
nRows=1136
nColumns=41
inDegrees=yes
Units are S.I. units (seconds, meters, Newtons, ...)
Angles are in degrees.

time	pelvis_tilt	pelvis_list	pelvis_rotation	pelvis_tx	pelvis_ty	pelvis_tz	...
0.000000	0.123456	1.234567	-12.345678	0.012345	0.987654	0.001234	...
0.033333	0.123789	1.234890	-12.346011	0.012678	0.987987	0.001567	...
...
```

### B.3 Pickle File Format

```python
{
    'video_name': 'aitor_garden_walk',
    'video_path': 'E:/VideoPoseEstimation/input/aitor_garden_walk.mp4',
    'resolution': (1920, 1080),
    'fps': 30.0,
    'total_frames': 1136,
    'keypoints_2d_coco': np.ndarray,  # (1136, 17, 2)
    'keypoints_2d_h36m': np.ndarray,  # (1136, 17, 2)
    'keypoints_3d': np.ndarray,       # (1136, 17, 3)
    'scores': np.ndarray,             # (1136, 17)
    'joint_names': [...],             # H36M joint names
    'skeleton': [...],                # Skeleton connections
    'processing_time': 45.67,
}
```

---

## Appendix C: References

1. **MotionBERT:** Zhu et al., "MotionBERT: A Unified Perspective on Learning Human Motion Representations" (ICCV 2023)

2. **YOLOv8-Pose:** Ultralytics, https://github.com/ultralytics/ultralytics

3. **Pose2Sim:** Pagnon et al., "Pose2Sim: An open-source Python package for multiview markerless kinematics" (JOSS 2022)

4. **OpenSim:** Delp et al., "OpenSim: Open-Source Software to Create and Analyze Dynamic Simulations of Movement" (IEEE TBME 2007)

5. **Human3.6M Dataset:** Ionescu et al., "Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments" (TPAMI 2014)

6. **COCO Keypoint Format:** Lin et al., "Microsoft COCO: Common Objects in Context" (ECCV 2014)

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-01 | Initial comprehensive documentation |
| 1.1 | 2026-02-02 | Added forward lean correction, sliding window fix, auto output folders, reports |

---

## Appendix D: Updates from 2026-02-02 Session

### D.1 New Features Added

#### D.1.1 Auto-Generated Output Folder Names
Output directories are now automatically named as `output_YYYYMMDD_HHMMSS_videoname` when `--output` is not specified.

```bash
# Auto-generate output folder
python run_full_pipeline.py --input video.mp4 --height 1.75

# Custom output folder
python run_full_pipeline.py --input video.mp4 --output results/my_output --height 1.75
```

#### D.1.2 Processing Reports
Each run generates detailed reports:

**Text report (`processing_report.txt`):**
- Metadata (input video, height, device)
- Video information (resolution, FPS, frames, duration)
- Corrections applied (lean angle, scaling)
- Processing times breakdown with percentages
- Errors and warnings
- Output files list

**JSON report (`processing_report.json`):**
- Same data in machine-readable format for programmatic access

#### D.1.3 Forward Lean Correction
New step (4b) corrects systematic camera-facing tilt from monocular depth estimation:

```python
def correct_forward_lean(keypoints: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Correct systematic forward/backward lean from monocular depth estimation.

    Algorithm:
    1. Compute average spine direction across all frames
    2. Calculate lean angle from spine projection onto sagittal (YZ) plane
    3. Apply rotation correction around X-axis to verticalize spine
    """
```

Typical correction: ~16-17° for side-view walking videos.

### D.2 Bug Fixes

#### D.2.1 Sliding Window Motion Repetition
**Problem:** Last portion of video was repeated in the output motion.

**Cause:** Incorrect frame tracking in sliding window inference loop. When the final window was computed, the code was outputting more frames than needed.

**Fix:** Track `frames_collected` and only take exactly the frames needed to reach `n_frames`:

```python
# Track how many frames we've collected
frames_collected = 0

for start in range(0, n_frames, stride):
    # ... inference code ...

    if original_start == 0:
        # First window
        outputs.append(output[:keep_end])
        frames_collected = keep_end
    else:
        # Calculate how many frames we still need
        frames_needed = n_frames - frames_collected
        offset_in_window = frames_collected - start
        keep_count = min(frames_needed, clip_len - offset_in_window)
        outputs.append(output[offset_in_window:offset_in_window + keep_count])
        frames_collected += keep_count

        if frames_collected >= n_frames:
            break
```

#### D.2.2 Forward Lean Rotation Direction
**Problem:** Initial forward lean correction was applied in wrong direction, making lean worse.

**Fix:** Changed rotation to apply `lean_angle` directly (not `-lean_angle`) when spine Z component is negative.

### D.3 pyopensim Compatibility

Created `opensim_shim.py` to handle differences between `pyopensim` package and official `opensim`:

```python
# opensim_shim.py
import pyopensim
from pyopensim import *
from pyopensim.common import *
from pyopensim.simulation import *
from pyopensim.tools import *
from pyopensim.actuators import *
from pyopensim.analyses import *
from pyopensim.common import Logger, LogSink, StringLogSink
__version__ = pyopensim.__version__
```

The shim is loaded automatically if `import opensim` fails.

### D.4 Pre-Scaling Removal

**Previous behavior:** Step 5 scaled the skeleton based on `head_y - foot_y` before Pose2Sim.

**Problem:** For unusual poses (e.g., Pilates with person bent over), `head_y - foot_y` doesn't represent standing height, causing extreme scaling (7x observed).

**Solution:** Removed pre-scaling, delegating all scaling to Pose2Sim's `participant_height` parameter.

```python
# Note: We do NOT scale here - let Pose2Sim handle scaling
# This avoids issues with unusual poses where head-to-foot height
# doesn't reflect standing height
```

### D.5 Known Limitations

#### D.5.1 Unusual Pose Scaling
For videos with non-standard poses (e.g., Pilates, yoga, gymnastics), Pose2Sim's marker-based scaling may produce incorrect segment sizes for head and feet since there are no markers directly on these body parts.

The default Pose2Sim scaling works well for normal walking/standing videos but may have issues with:
- Extreme poses (inverted, bent over, etc.)
- Vertical/portrait videos
- People on elevated platforms

#### D.5.2 Head Wobble
Monocular depth estimation causes head position to fluctuate laterally. This manifests as head tilting side-to-side in the output motion. Potential solutions (not yet implemented):
- Temporal smoothing (Butterworth filter)
- Head position constraint relative to shoulders
- Spine-axis head constraint

### D.6 File Backups Created

| File | Backup Location | Purpose |
|------|-----------------|---------|
| `src/trc_motionbert.py` | `src/trc_motionbert.py.backup` | Before removing pre-scaling |
| `Pose2Sim/.../Scaling_Setup_Pose2Sim_Coco17.xml` | `.../Scaling_Setup_Pose2Sim_Coco17.xml.backup` | Original Pose2Sim scaling |
| `Pose2Sim/.../Markers_Coco17.xml` | `models/Markers_Coco17_fixed.xml` | Fixed marker positions |

### D.7 Test Videos

| Video | Height | Notes |
|-------|--------|-------|
| `louise_garden_walk.mp4` | 1.61m | Side-view walking, horizontal video, works well |
| `aitor_garden_walk.mp4` | - | Side-view walking, works well |
| `rebeca.mp4` | 1.65m | Pilates/stretching, vertical video, unusual poses - scaling challenges |

---

**End of Documentation**
