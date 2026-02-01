"""
TRC export compatible with Pose2Sim pipeline.

Converts RTMPose3D keypoints to HALPE_26 format for use with:
- Pose2Sim.common.compute_height() for height computation
- Pose2Sim.markerAugmentation() for adding anatomical markers
- Pose2Sim.kinematics() for joint angle computation

HALPE_26 keypoint order (from Pose2Sim.skeletons):
  0: Nose
  5: LShoulder, 6: RShoulder
  7: LElbow, 8: RElbow
  9: LWrist, 10: RWrist
  11: LHip, 12: RHip
  13: LKnee, 14: RKnee
  15: LAnkle, 16: RAnkle
  17: Head, 18: Neck, 19: Hip
  20: LBigToe, 21: RBigToe
  22: LSmallToe, 23: RSmallToe
  24: LHeel, 25: RHeel

Reference: https://github.com/perfanalytics/pose2sim
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import json

from .constants import (
    RTMPOSE_NOSE,
    RTMPOSE_LEFT_SHOULDER, RTMPOSE_RIGHT_SHOULDER,
    RTMPOSE_LEFT_ELBOW, RTMPOSE_RIGHT_ELBOW,
    RTMPOSE_LEFT_WRIST, RTMPOSE_RIGHT_WRIST,
    RTMPOSE_LEFT_HIP, RTMPOSE_RIGHT_HIP,
    RTMPOSE_LEFT_KNEE, RTMPOSE_RIGHT_KNEE,
    RTMPOSE_LEFT_ANKLE, RTMPOSE_RIGHT_ANKLE,
    RTMPOSE_LEFT_BIG_TOE, RTMPOSE_RIGHT_BIG_TOE,
    RTMPOSE_LEFT_SMALL_TOE, RTMPOSE_RIGHT_SMALL_TOE,
    RTMPOSE_LEFT_HEEL, RTMPOSE_RIGHT_HEEL,
)


# HALPE_26 keypoint names in order (indices 0-25, but some are missing in pose estimation)
# We map RTMPose3D keypoints to these
HALPE_26_NAMES = [
    'Nose',       # 0
    None, None, None, None,  # 1-4: not used (eyes, ears)
    'LShoulder',  # 5
    'RShoulder',  # 6
    'LElbow',     # 7
    'RElbow',     # 8
    'LWrist',     # 9
    'RWrist',     # 10
    'LHip',       # 11
    'RHip',       # 12
    'LKnee',      # 13
    'RKnee',      # 14
    'LAnkle',     # 15
    'RAnkle',     # 16
    'Head',       # 17 (computed)
    'Neck',       # 18 (computed as mid-shoulder)
    'Hip',        # 19 (computed as mid-hip)
    'LBigToe',    # 20
    'RBigToe',    # 21
    'LSmallToe',  # 22
    'RSmallToe',  # 23
    'LHeel',      # 24
    'RHeel',      # 25
]

# RTMPose3D to HALPE_26 mapping
# Format: (halpe_idx, source_type, source_data)
# source_type: 'landmark', 'midpoint', 'computed'
RTMPOSE_TO_HALPE = [
    (0, 'landmark', RTMPOSE_NOSE),           # Nose
    (5, 'landmark', RTMPOSE_LEFT_SHOULDER),  # LShoulder
    (6, 'landmark', RTMPOSE_RIGHT_SHOULDER), # RShoulder
    (7, 'landmark', RTMPOSE_LEFT_ELBOW),     # LElbow
    (8, 'landmark', RTMPOSE_RIGHT_ELBOW),    # RElbow
    (9, 'landmark', RTMPOSE_LEFT_WRIST),     # LWrist
    (10, 'landmark', RTMPOSE_RIGHT_WRIST),   # RWrist
    (11, 'landmark', RTMPOSE_LEFT_HIP),      # LHip
    (12, 'landmark', RTMPOSE_RIGHT_HIP),     # RHip
    (13, 'landmark', RTMPOSE_LEFT_KNEE),     # LKnee
    (14, 'landmark', RTMPOSE_RIGHT_KNEE),    # RKnee
    (15, 'landmark', RTMPOSE_LEFT_ANKLE),    # LAnkle
    (16, 'landmark', RTMPOSE_RIGHT_ANKLE),   # RAnkle
    (17, 'computed', 'head'),                # Head (above nose)
    (18, 'midpoint', (RTMPOSE_LEFT_SHOULDER, RTMPOSE_RIGHT_SHOULDER)),  # Neck
    (19, 'midpoint', (RTMPOSE_LEFT_HIP, RTMPOSE_RIGHT_HIP)),            # Hip
    (20, 'landmark', RTMPOSE_LEFT_BIG_TOE),  # LBigToe
    (21, 'landmark', RTMPOSE_RIGHT_BIG_TOE), # RBigToe
    (22, 'landmark', RTMPOSE_LEFT_SMALL_TOE),  # LSmallToe
    (23, 'landmark', RTMPOSE_RIGHT_SMALL_TOE), # RSmallToe
    (24, 'landmark', RTMPOSE_LEFT_HEEL),     # LHeel
    (25, 'landmark', RTMPOSE_RIGHT_HEEL),    # RHeel
]


def _compute_head_position(keypoints: np.ndarray) -> np.ndarray:
    """
    Compute head position (top of head) from nose and neck.

    In OpenSim coords (Y-up), head is above nose by ~0.5 * neck-to-nose distance.
    """
    nose = keypoints[:, RTMPOSE_NOSE, :]
    neck = (keypoints[:, RTMPOSE_LEFT_SHOULDER, :] +
            keypoints[:, RTMPOSE_RIGHT_SHOULDER, :]) / 2

    # Direction from neck to nose
    neck_to_nose = nose - neck
    dist = np.linalg.norm(neck_to_nose, axis=1, keepdims=True)

    # Head is above nose in the Y direction (up in OpenSim)
    head = nose.copy()
    head[:, 1] += dist.flatten() * 0.5  # Y is up in OpenSim coords

    return head


def compute_halpe26_markers(
    keypoints: np.ndarray,
) -> Tuple[List[str], np.ndarray]:
    """
    Convert RTMPose3D keypoints to HALPE_26 format.

    Args:
        keypoints: (n_frames, 133, 3) array in OpenSim coordinates

    Returns:
        marker_names: List of HALPE_26 marker names (22 markers)
        marker_positions: (n_frames, 22, 3) array
    """
    n_frames = len(keypoints)
    marker_names = []
    marker_data = []

    for halpe_idx, source_type, source_data in RTMPOSE_TO_HALPE:
        name = HALPE_26_NAMES[halpe_idx]
        marker_names.append(name)

        if source_type == 'landmark':
            pos = keypoints[:, source_data, :]

        elif source_type == 'midpoint':
            idx1, idx2 = source_data
            pos = (keypoints[:, idx1, :] + keypoints[:, idx2, :]) / 2

        elif source_type == 'computed':
            if source_data == 'head':
                pos = _compute_head_position(keypoints)
            else:
                raise ValueError(f"Unknown computed marker: {source_data}")
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        marker_data.append(pos)

    markers = np.stack(marker_data, axis=1)
    return marker_names, markers


def write_trc_halpe26(
    marker_names: List[str],
    marker_positions: np.ndarray,
    output_path: str,
    fps: float = 30.0,
) -> str:
    """
    Write HALPE_26 markers to TRC file.

    Args:
        marker_names: List of marker names
        marker_positions: (n_frames, n_markers, 3) in meters
        output_path: Output file path
        fps: Frame rate

    Returns:
        Path to output file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames, n_markers, _ = marker_positions.shape

    # Keep in meters for Pose2Sim (OpenSim uses meters)
    # Note: TRC traditionally uses mm, but Pose2Sim's kinematics compares
    # TRC distances directly to model distances (in meters) without conversion
    positions_m = marker_positions

    with open(output_path, 'w') as f:
        # Line 1: File info
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{output_path.name}\n")

        # Line 2: Header labels
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")

        # Line 3: Header values (units = m for meters)
        f.write(f"{fps:.2f}\t{fps:.2f}\t{n_frames}\t{n_markers}\tm\t{fps:.2f}\t1\t{n_frames}\n")

        # Line 4: Marker names
        marker_header = "Frame#\tTime"
        for name in marker_names:
            marker_header += f"\t{name}\t\t"
        f.write(marker_header.rstrip() + "\n")

        # Line 5: Coordinate labels
        coord_header = "\t"
        for i in range(n_markers):
            coord_header += f"\tX{i+1}\tY{i+1}\tZ{i+1}"
        f.write(coord_header + "\n")

        # Line 6: Empty line
        f.write("\n")

        # Data lines
        for frame_idx in range(n_frames):
            time = frame_idx / fps
            line = f"{frame_idx + 1}\t{time:.6f}"
            for marker_idx in range(n_markers):
                x, y, z = positions_m[frame_idx, marker_idx, :]
                line += f"\t{x:.6f}\t{y:.6f}\t{z:.6f}"
            f.write(line + "\n")

    return str(output_path)


def export_trc_for_pose2sim(
    processed_json: str,
    output_path: str,
    real_height: float,
) -> Tuple[str, dict]:
    """
    Export processed keypoints to Pose2Sim-compatible TRC.

    Pipeline:
    1. Load processed keypoints (already in OpenSim coords)
    2. Convert to HALPE_26 format
    3. Compute height from markers using Pose2Sim function
    4. Scale to real height
    5. Offset to ground (feet at Y=0)
    6. Write TRC file

    Args:
        processed_json: Path to processed keypoints JSON
        output_path: Output TRC path
        real_height: Person's real height in meters

    Returns:
        Tuple of (output_path, info_dict)
    """
    # Import Pose2Sim functions
    try:
        from Pose2Sim.common import compute_height
        use_pose2sim_height = True
    except ImportError:
        print("  Warning: Pose2Sim not available, using simple height estimation")
        use_pose2sim_height = False

    # Load processed keypoints
    with open(processed_json, 'r') as f:
        data = json.load(f)

    keypoints = np.array(data['keypoints'])
    fps = data['metadata'].get('fps', 30.0)

    print(f"  Frames: {len(keypoints)}, FPS: {fps}")

    # Step 1: Convert to HALPE_26
    print("  Converting to HALPE_26 format...")
    marker_names, markers = compute_halpe26_markers(keypoints)
    print(f"  Generated {len(marker_names)} HALPE_26 markers")

    # Step 2: Compute height
    if use_pose2sim_height:
        print("  Computing height using Pose2Sim.common.compute_height()...")
        # Pose2Sim expects a DataFrame where each marker name appears 3 times (for X, Y, Z)
        # Access is done via Q_coords['MarkerName'] which returns a DataFrame with 3 columns
        import pandas as pd

        # Create DataFrame with columns: [Nose, Nose, Nose, LShoulder, LShoulder, LShoulder, ...]
        # Each marker name repeated 3 times for X, Y, Z
        n_frames = len(markers)
        data_dict = {}
        columns = []
        for i, name in enumerate(marker_names):
            # Add X, Y, Z as separate columns all with the same marker name
            columns.extend([name, name, name])

        # Create data array: flatten (n_frames, n_markers, 3) to (n_frames, n_markers*3)
        data_flat = markers.reshape(n_frames, -1)
        Q_coords = pd.DataFrame(data_flat, columns=columns)

        try:
            computed_height = compute_height(Q_coords, marker_names)
            print(f"  Computed height (Pose2Sim): {computed_height:.3f}m")
        except Exception as e:
            print(f"  Warning: Pose2Sim compute_height failed: {e}")
            print("  Falling back to simple height estimation...")
            use_pose2sim_height = False
    if not use_pose2sim_height:
        # Simple fallback: head to heel distance
        head_idx = marker_names.index('Head')
        lheel_idx = marker_names.index('LHeel')
        rheel_idx = marker_names.index('RHeel')

        head_y = markers[:, head_idx, 1]
        min_heel_y = np.minimum(markers[:, lheel_idx, 1], markers[:, rheel_idx, 1])
        heights = head_y - min_heel_y
        computed_height = float(np.median(heights))
        print(f"  Computed height (simple): {computed_height:.3f}m")

    # Step 3: Scale to real height
    scale_factor = real_height / computed_height
    print(f"  Scale factor: {scale_factor:.4f} (computed={computed_height:.3f}m -> real={real_height:.3f}m)")
    markers_scaled = markers * scale_factor

    # Step 4: Offset to ground
    foot_markers = ['LHeel', 'RHeel', 'LBigToe', 'RBigToe', 'LSmallToe', 'RSmallToe']
    foot_indices = [marker_names.index(m) for m in foot_markers if m in marker_names]

    foot_heights = markers_scaled[:, foot_indices, 1]  # Y is up
    min_foot_y = np.min(foot_heights)

    print(f"  Ground offset: {-min_foot_y*1000:.1f}mm")
    markers_grounded = markers_scaled.copy()
    markers_grounded[:, :, 1] -= min_foot_y

    # Step 5: Write TRC
    print(f"  Writing TRC file...")
    trc_path = write_trc_halpe26(marker_names, markers_grounded, output_path, fps)
    print(f"  Saved: {trc_path}")

    # Compute final height for verification
    head_idx = marker_names.index('Head')
    final_heights = markers_grounded[:, head_idx, 1]
    final_height = float(np.median(final_heights))
    print(f"  Final height (head to ground): {final_height:.3f}m")

    info = {
        'computed_height': computed_height,
        'real_height': real_height,
        'scale_factor': scale_factor,
        'final_height': final_height,
        'ground_offset': -min_foot_y,
        'n_markers': len(marker_names),
        'n_frames': len(keypoints),
        'marker_names': marker_names,
        'fps': fps,
    }

    return trc_path, info


def run_pose2sim_kinematics(
    trc_path: str,
    output_dir: str,
    participant_height: float,
    participant_mass: float = 70.0,
    use_augmentation: bool = True,
) -> dict:
    """
    Run Pose2Sim kinematics on a HALPE_26 TRC file.

    Creates a minimal Pose2Sim project structure and runs scaling + IK.

    Args:
        trc_path: Path to HALPE_26 TRC file
        output_dir: Directory for output files
        participant_height: Height in meters (for scaling)
        participant_mass: Mass in kg (default 70)
        use_augmentation: Whether to run marker augmentation first

    Returns:
        Dictionary with paths to output files
    """
    import os
    import shutil

    trc_path = Path(trc_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal Pose2Sim project structure
    project_dir = output_dir / 'pose2sim_project'
    pose_3d_dir = project_dir / 'pose-3d'
    pose_3d_dir.mkdir(parents=True, exist_ok=True)

    # Copy TRC to pose-3d folder
    trc_dest = pose_3d_dir / trc_path.name
    shutil.copy(trc_path, trc_dest)

    print(f"  Created Pose2Sim project at: {project_dir}")
    print(f"  TRC file: {trc_dest}")

    # Create config dictionary (Pose2Sim accepts dict directly)
    config_dict = {
        'project': {
            'project_dir': str(project_dir),
            'multi_person': False,
            'participant_height': participant_height,
            'participant_mass': participant_mass,
            'frame_rate': 'auto',
            'frame_range': 'all',
        },
        'pose': {
            'pose_model': 'HALPE_26',
        },
        'markerAugmentation': {
            'feet_on_floor': False,
            'make_c3d': False,
        },
        'kinematics': {
            'use_augmentation': use_augmentation,
            'use_simple_model': True,
            'right_left_symmetry': True,
            'default_height': participant_height,
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
    try:
        from Pose2Sim import Pose2Sim as P2S

        if use_augmentation:
            print("  Running Pose2Sim.markerAugmentation()...")
            P2S.markerAugmentation(config_dict)

        print("  Running Pose2Sim.kinematics()...")
        P2S.kinematics(config_dict)

        # Find output files
        kinematics_dir = project_dir / 'kinematics'
        mot_files = list(kinematics_dir.glob('*.mot'))
        osim_files = list(kinematics_dir.glob('*_scaled.osim'))

        results = {
            'project_dir': str(project_dir),
            'mot_files': [str(f) for f in mot_files],
            'osim_files': [str(f) for f in osim_files],
        }

        if mot_files:
            print(f"  Output MOT: {mot_files[0]}")
            # Copy to output directory
            for mot in mot_files:
                dest = output_dir / mot.name
                shutil.copy(mot, dest)
                print(f"  Copied to: {dest}")

        return results

    except Exception as e:
        print(f"  Error running Pose2Sim: {e}")
        raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Export to Pose2Sim-compatible TRC')
    parser.add_argument('processed_json', help='Processed keypoints JSON')
    parser.add_argument('--height', type=float, required=True, help='Real height in meters')
    parser.add_argument('-o', '--output', help='Output TRC path')
    parser.add_argument('--run-kinematics', action='store_true', help='Run Pose2Sim kinematics')

    args = parser.parse_args()

    if args.output is None:
        input_path = Path(args.processed_json)
        args.output = str(input_path.parent.parent / 'markers' / f"{input_path.stem.replace('_processed', '')}_halpe26.trc")

    trc_path, info = export_trc_for_pose2sim(
        args.processed_json,
        args.output,
        args.height,
    )

    print(f"\nExported to: {trc_path}")
    print(f"Scale factor: {info['scale_factor']:.4f}")
    print(f"Final height: {info['final_height']:.3f}m")

    if args.run_kinematics:
        print("\nRunning Pose2Sim kinematics...")
        output_dir = Path(args.output).parent
        results = run_pose2sim_kinematics(
            trc_path,
            str(output_dir),
            args.height,
        )
        print(f"\nKinematics complete!")
        print(f"MOT files: {results['mot_files']}")
