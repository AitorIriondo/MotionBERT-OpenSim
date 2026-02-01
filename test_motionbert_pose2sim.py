"""Test MotionBERT TRC with Pose2Sim."""
import os
import shutil
from pathlib import Path

# Set up project structure for Pose2Sim
project_dir = Path('output/pose2sim_motionbert')
pose3d_dir = project_dir / 'pose-3d'
kinematics_dir = project_dir / 'kinematics'

# Create directories
pose3d_dir.mkdir(parents=True, exist_ok=True)
kinematics_dir.mkdir(parents=True, exist_ok=True)

# Copy TRC to pose-3d directory
trc_src = Path('output/aitor_garden_walk_motionbert.trc')
trc_dst = pose3d_dir / 'aitor_garden_walk_motionbert.trc'
shutil.copy(trc_src, trc_dst)
print(f"Copied TRC to {trc_dst}")

# Create Pose2Sim config
config_dict = {
    'project': {
        'project_dir': str(project_dir),
        'multi_person': False,
        'participant_height': 1.75,
        'participant_mass': 70.0,
        'frame_rate': 'auto',
        'frame_range': 'all',
    },
    'pose': {
        'pose_model': 'COCO_17',  # Using COCO_17 for H36M markers
    },
    'markerAugmentation': {
        'feet_on_floor': False,
        'make_c3d': False,
    },
    'kinematics': {
        'use_augmentation': False,
        'use_simple_model': True,
        'right_left_symmetry': True,
        'default_height': 1.75,
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

# Run Pose2Sim
print("\nRunning Pose2Sim kinematics...")
try:
    from Pose2Sim import Pose2Sim as P2S

    P2S.kinematics(config_dict)

    # Check output
    mot_files = list(kinematics_dir.glob('*.mot'))
    osim_files = list(kinematics_dir.glob('*_scaled.osim'))

    print(f"\nOutput files:")
    for f in mot_files:
        print(f"  MOT: {f}")
    for f in osim_files:
        print(f"  OSIM: {f}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
