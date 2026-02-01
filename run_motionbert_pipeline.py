"""
Run the complete MotionBERT to OpenSim pipeline.

This script:
1. Converts MotionBERT H36M keypoints to TRC format
2. Runs Pose2Sim kinematics (scaling + IK)
3. Outputs the results to the output directory
"""
import os
import shutil
from pathlib import Path

# Configuration
INPUT_PKL = "E:/VideoPoseEstimation/output_fixed/aitor_garden_walk_poses.pkl"
OUTPUT_DIR = Path("output/motionbert_opensim")
TARGET_HEIGHT = 1.75  # meters

# Clean output directory
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Convert MotionBERT to TRC
print("=" * 60)
print("Step 1: Converting MotionBERT keypoints to TRC")
print("=" * 60)

from src.trc_motionbert import convert_motionbert_to_trc

trc_path = OUTPUT_DIR / "walk.trc"
convert_motionbert_to_trc(INPUT_PKL, str(trc_path), TARGET_HEIGHT, fix_lr_swap=False, normalize_bones=True)

print(f"\nTRC saved to: {trc_path}")

# Step 2: Set up Pose2Sim project structure
print("\n" + "=" * 60)
print("Step 2: Setting up Pose2Sim project")
print("=" * 60)

pose3d_dir = OUTPUT_DIR / "pose-3d"
kinematics_dir = OUTPUT_DIR / "kinematics"
pose3d_dir.mkdir(exist_ok=True)

# Copy TRC to pose-3d directory
trc_dest = pose3d_dir / trc_path.name
shutil.copy(trc_path, trc_dest)
print(f"TRC copied to: {trc_dest}")

# Step 3: Run Pose2Sim
print("\n" + "=" * 60)
print("Step 3: Running Pose2Sim kinematics")
print("=" * 60)

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
        'pose_model': 'COCO_17',
    },
    'markerAugmentation': {
        'feet_on_floor': False,
        'make_c3d': False,
    },
    'kinematics': {
        'use_augmentation': False,
        'use_simple_model': True,
        'right_left_symmetry': True,
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

from Pose2Sim import Pose2Sim as P2S
P2S.kinematics(config_dict)

# Step 4: Report results
print("\n" + "=" * 60)
print("Step 4: Results")
print("=" * 60)

mot_files = list(kinematics_dir.glob("*.mot"))
osim_files = list(kinematics_dir.glob("*_scaled.osim"))

print("\nOutput files:")
for f in kinematics_dir.iterdir():
    print(f"  {f.name}")

if mot_files:
    print(f"\n*** MOT file: {mot_files[0]} ***")

if osim_files:
    print(f"*** Scaled model: {osim_files[0]} ***")

print("\n" + "=" * 60)
print("DONE! Open the .osim file in OpenSim GUI, then load the .mot file")
print("=" * 60)
