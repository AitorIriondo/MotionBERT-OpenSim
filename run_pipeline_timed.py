"""
Run the complete MotionBERT to OpenSim pipeline with timing.

This script times each step to give FPS expectations.
"""
import os
import shutil
import time
from pathlib import Path

# Configuration
INPUT_PKL = "E:/VideoPoseEstimation/output_fixed/aitor_garden_walk_poses.pkl"
OUTPUT_DIR = Path("output/motionbert_opensim")
TARGET_HEIGHT = 1.75  # meters

def format_time(seconds):
    """Format time in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        return f"{seconds/60:.1f} min"

def main():
    timings = {}

    print("=" * 70)
    print("RTMPose3DOpenSim Pipeline (Timed)")
    print("=" * 70)
    print(f"\nInput: {INPUT_PKL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Target height: {TARGET_HEIGHT}m")
    print()

    # Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: Load MotionBERT keypoints
    # =========================================================================
    print("-" * 70)
    print("STEP 1: Load MotionBERT H36M keypoints from pickle")
    print("-" * 70)

    t0 = time.perf_counter()

    import pickle
    import numpy as np

    with open(INPUT_PKL, 'rb') as f:
        data = pickle.load(f)

    keypoints = data['keypoints_3d']
    fps = data.get('fps', 30.0)
    n_frames = keypoints.shape[0]

    t1 = time.perf_counter()
    timings['1_load_pickle'] = t1 - t0

    print(f"  Shape: {keypoints.shape} (frames, joints, xyz)")
    print(f"  FPS: {fps}")
    print(f"  Duration: {n_frames/fps:.2f} seconds of video")
    print(f"  Time: {format_time(timings['1_load_pickle'])}")

    # =========================================================================
    # STEP 2: Normalize bone lengths
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: Normalize bone lengths (fix monocular depth estimation errors)")
    print("-" * 70)

    t0 = time.perf_counter()

    from src.trc_motionbert import normalize_bone_lengths
    keypoints = normalize_bone_lengths(keypoints)

    t1 = time.perf_counter()
    timings['2_normalize_bones'] = t1 - t0

    print(f"  Time: {format_time(timings['2_normalize_bones'])}")

    # =========================================================================
    # STEP 3: Transform to OpenSim coordinate system
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: Transform MotionBERT -> OpenSim coordinates")
    print("-" * 70)

    t0 = time.perf_counter()

    from src.trc_motionbert import transform_motionbert_to_opensim
    keypoints_osim = transform_motionbert_to_opensim(keypoints, TARGET_HEIGHT, use_simple_transform=True)

    t1 = time.perf_counter()
    timings['3_transform_coords'] = t1 - t0

    print(f"  X (forward) range: [{keypoints_osim[..., 0].min():.3f}, {keypoints_osim[..., 0].max():.3f}]")
    print(f"  Y (up) range: [{keypoints_osim[..., 1].min():.3f}, {keypoints_osim[..., 1].max():.3f}]")
    print(f"  Z (right) range: [{keypoints_osim[..., 2].min():.3f}, {keypoints_osim[..., 2].max():.3f}]")
    print(f"  Time: {format_time(timings['3_transform_coords'])}")

    # =========================================================================
    # STEP 4: Extract COCO17 markers (including synthetic nose)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: Extract COCO17 markers from H36M joints")
    print("-" * 70)

    t0 = time.perf_counter()

    from src.trc_motionbert import extract_coco17_markers, COCO17_MARKER_NAMES
    marker_names, marker_positions = extract_coco17_markers(keypoints_osim)

    t1 = time.perf_counter()
    timings['4_extract_markers'] = t1 - t0

    print(f"  Markers: {len(marker_names)}")
    print(f"  Names: {', '.join(marker_names)}")
    print(f"  Time: {format_time(timings['4_extract_markers'])}")

    # =========================================================================
    # STEP 5: Export to TRC format
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: Export TRC file")
    print("-" * 70)

    t0 = time.perf_counter()

    from src.trc_motionbert import export_trc
    trc_path = OUTPUT_DIR / "walk.trc"
    export_trc(marker_positions, marker_names, str(trc_path), fps)

    t1 = time.perf_counter()
    timings['5_export_trc'] = t1 - t0

    print(f"  Time: {format_time(timings['5_export_trc'])}")

    # =========================================================================
    # STEP 6: Setup Pose2Sim project structure
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 6: Setup Pose2Sim project structure")
    print("-" * 70)

    t0 = time.perf_counter()

    pose3d_dir = OUTPUT_DIR / "pose-3d"
    kinematics_dir = OUTPUT_DIR / "kinematics"
    pose3d_dir.mkdir(exist_ok=True)

    # Copy TRC to pose-3d directory
    trc_dest = pose3d_dir / trc_path.name
    shutil.copy(trc_path, trc_dest)

    t1 = time.perf_counter()
    timings['6_setup_pose2sim'] = t1 - t0

    print(f"  Created: {pose3d_dir}")
    print(f"  TRC copied to: {trc_dest}")
    print(f"  Time: {format_time(timings['6_setup_pose2sim'])}")

    # =========================================================================
    # STEP 7: Run Pose2Sim kinematics (scaling + IK)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 7: Run Pose2Sim kinematics (OpenSim scaling + IK)")
    print("-" * 70)

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

    t0 = time.perf_counter()

    from Pose2Sim import Pose2Sim as P2S
    P2S.kinematics(config_dict)

    t1 = time.perf_counter()
    timings['7_pose2sim_kinematics'] = t1 - t0

    print(f"  Time: {format_time(timings['7_pose2sim_kinematics'])}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)

    total_time = sum(timings.values())

    print(f"\n{'Step':<45} {'Time':>12} {'%':>8}")
    print("-" * 65)

    for step, t in timings.items():
        pct = (t / total_time) * 100
        print(f"  {step:<43} {format_time(t):>12} {pct:>7.1f}%")

    print("-" * 65)
    print(f"  {'TOTAL':<43} {format_time(total_time):>12} {'100.0%':>8}")

    # FPS calculations
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)

    # Pre-processing (steps 1-5)
    preprocessing_time = sum([
        timings['1_load_pickle'],
        timings['2_normalize_bones'],
        timings['3_transform_coords'],
        timings['4_extract_markers'],
        timings['5_export_trc'],
    ])

    # Pose2Sim time (step 7)
    pose2sim_time = timings['7_pose2sim_kinematics']

    print(f"\nVideo info:")
    print(f"  Frames: {n_frames}")
    print(f"  Original FPS: {fps:.1f}")
    print(f"  Duration: {n_frames/fps:.2f} seconds")

    print(f"\nPre-processing (pickle -> TRC):")
    print(f"  Total time: {format_time(preprocessing_time)}")
    print(f"  Throughput: {n_frames/preprocessing_time:.1f} frames/sec")
    print(f"  Real-time factor: {(n_frames/fps)/preprocessing_time:.1f}x")

    print(f"\nPose2Sim kinematics (TRC -> joint angles):")
    print(f"  Total time: {format_time(pose2sim_time)}")
    print(f"  Throughput: {n_frames/pose2sim_time:.1f} frames/sec")
    print(f"  Real-time factor: {(n_frames/fps)/pose2sim_time:.1f}x")

    print(f"\nEnd-to-end (pickle -> joint angles):")
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Throughput: {n_frames/total_time:.1f} frames/sec")
    print(f"  Real-time factor: {(n_frames/fps)/total_time:.1f}x")

    # =========================================================================
    # OUTPUT FILES
    # =========================================================================
    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    mot_files = list(kinematics_dir.glob("*.mot"))
    osim_files = list(kinematics_dir.glob("*_scaled.osim"))

    print("\nFiles created:")
    for f in OUTPUT_DIR.rglob("*"):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.relative_to(OUTPUT_DIR)} ({size_kb:.1f} KB)")

    if mot_files:
        print(f"\n*** Motion file: {mot_files[0]} ***")
    if osim_files:
        print(f"*** Scaled model: {osim_files[0]} ***")

    print("\n" + "=" * 70)
    print("DONE! Open the .osim file in OpenSim GUI, then load the .mot file")
    print("=" * 70)

if __name__ == "__main__":
    main()
