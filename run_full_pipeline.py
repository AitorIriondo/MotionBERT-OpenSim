"""
Full End-to-End Pipeline: Video -> OpenSim Joint Angles (Standalone)

This script runs the complete pipeline from raw video to OpenSim joint angles.

Pipeline Steps:
1. YOLOv8-Pose: 2D keypoint detection (COCO17 format)
2. COCO17 -> H36M: Keypoint format conversion
3. MotionBERT: 2D -> 3D lifting (H36M 17 joints)
4. Bone normalization: Fix monocular depth inconsistency
5. Coordinate transform: MotionBERT -> OpenSim axes
6. Marker extraction: H36M -> COCO17 (14 markers)
7. TRC export: Write marker trajectories
8. Pose2Sim/OpenSim: Scaling + Inverse Kinematics

Usage:
    python run_full_pipeline.py --input input/video.mp4 --output output/results --height 1.75
"""

import os
import sys
import shutil
import time
import pickle
from pathlib import Path
from typing import Dict
import argparse

import numpy as np
import cv2
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "MotionBERT"))


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds*1000000:.1f} us"
    elif seconds < 1:
        return f"{seconds*1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        return f"{seconds/60:.1f} min"


def run_pipeline(video_path: str, output_dir: str, target_height: float = 1.75,
                 device: str = 'cuda:0', timed: bool = True):
    """Run the full video to OpenSim pipeline."""

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    timings = {}

    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL PIPELINE: Video -> OpenSim Joint Angles")
    print("=" * 70)
    print(f"\nInput video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Target height: {target_height}m")
    print(f"Device: {device}")

    # Check CUDA
    if 'cuda' in device:
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: CUDA not available, using CPU")
            device = 'cpu'

    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps
    cap.release()

    print(f"\nVideo info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Frames: {n_frames}")
    print(f"  Duration: {duration:.2f} seconds")

    # =========================================================================
    # STEP 1: YOLOv8-Pose 2D Detection
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: YOLOv8-Pose 2D Keypoint Detection (COCO17)")
    print("=" * 70)

    t0 = time.perf_counter()

    from ultralytics import YOLO
    yolo_model = YOLO('yolov8m-pose.pt')
    yolo_device = 0 if 'cuda' in device and torch.cuda.is_available() else 'cpu'

    if timed:
        timings['1a_yolo_load'] = time.perf_counter() - t0
        print(f"  Model loaded: {format_time(timings['1a_yolo_load'])}")

    # Process video frames
    cap = cv2.VideoCapture(str(video_path))
    all_keypoints_coco = []
    all_scores = []

    t_infer = time.perf_counter()
    pbar = tqdm(total=n_frames, desc="  YOLOv8 detection")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, verbose=False, device=yolo_device)

        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            kpts_data = results[0].keypoints
            if hasattr(kpts_data, 'xy') and kpts_data.xy is not None:
                kpts = kpts_data.xy[0].cpu().numpy()
            else:
                kpts = kpts_data.data[0, :, :2].cpu().numpy()

            if hasattr(kpts_data, 'conf') and kpts_data.conf is not None:
                confs = kpts_data.conf[0].cpu().numpy()
            else:
                confs = kpts_data.data[0, :, 2].cpu().numpy()

            all_keypoints_coco.append(kpts)
            all_scores.append(confs)
        else:
            all_keypoints_coco.append(np.zeros((17, 2)))
            all_scores.append(np.zeros(17))

        pbar.update(1)

    pbar.close()
    cap.release()

    keypoints_coco = np.array(all_keypoints_coco)
    scores = np.array(all_scores)

    if timed:
        timings['1b_yolo_inference'] = time.perf_counter() - t_infer
        timings['1_yolo_total'] = time.perf_counter() - t0
        print(f"  Inference: {format_time(timings['1b_yolo_inference'])}")

    # =========================================================================
    # STEP 2: COCO17 -> H36M Conversion
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: COCO17 -> H36M Keypoint Conversion")
    print("=" * 70)

    t0 = time.perf_counter()

    from utils.keypoint_converter import COCOtoH36MConverter
    converter = COCOtoH36MConverter()

    keypoints_h36m = []
    scores_h36m = []

    for i in range(n_frames):
        h36m_kpts, h36m_scores = converter.convert(keypoints_coco[i], scores[i])
        keypoints_h36m.append(h36m_kpts)
        scores_h36m.append(h36m_scores)

    keypoints_h36m = np.array(keypoints_h36m)
    scores_h36m = np.array(scores_h36m)

    if timed:
        timings['2_coco_to_h36m'] = time.perf_counter() - t0
        print(f"  Converted {n_frames} frames: {format_time(timings['2_coco_to_h36m'])}")

    # =========================================================================
    # STEP 3: MotionBERT 2D -> 3D Lifting
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: MotionBERT 2D -> 3D Lifting")
    print("=" * 70)

    t0 = time.perf_counter()

    from functools import partial
    import torch.nn as nn
    from lib.model.DSTformer import DSTformer
    from lib.utils.tools import get_config

    # Load config
    config_path = PROJECT_ROOT / "MotionBERT" / "configs" / "pose3d" / "MB_ft_h36m_global_lite.yaml"
    args = get_config(str(config_path))

    # Create model
    model = DSTformer(
        dim_in=3,
        dim_out=3,
        dim_feat=args.dim_feat,
        dim_rep=args.dim_rep,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        maxlen=args.maxlen,
        num_joints=args.num_joints
    )

    # Load checkpoint
    ckpt_path = PROJECT_ROOT / "MotionBERT" / "checkpoint" / "pose3d" / "FT_MB_lite_MB_ft_h36m_global_lite" / "best_epoch.bin"
    checkpoint = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_pos']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)

    if 'cuda' in device and torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    if timed:
        timings['3a_motionbert_load'] = time.perf_counter() - t0
        print(f"  Model loaded: {format_time(timings['3a_motionbert_load'])}")

    # Prepare input
    kpts = keypoints_h36m.copy().astype(np.float32)
    confs = scores_h36m.copy().astype(np.float32)

    kpts[..., 0] = (kpts[..., 0] - width / 2) / (min(width, height) / 2)
    kpts[..., 1] = (kpts[..., 1] - height / 2) / (min(width, height) / 2)

    input_data = np.concatenate([kpts, confs[..., np.newaxis]], axis=-1)

    clip_len = args.maxlen
    if n_frames < clip_len:
        pad_len = clip_len - n_frames
        input_data = np.concatenate([
            input_data,
            np.tile(input_data[-1:], (pad_len, 1, 1))
        ], axis=0)

    # Inference
    t_infer = time.perf_counter()
    outputs = []
    stride = clip_len // 2

    def flip_data(data):
        flipped = data.clone()
        flipped[..., 0] = -flipped[..., 0]
        flipped[:, :, [1, 4], :] = flipped[:, :, [4, 1], :]
        flipped[:, :, [2, 5], :] = flipped[:, :, [5, 2], :]
        flipped[:, :, [3, 6], :] = flipped[:, :, [6, 3], :]
        flipped[:, :, [14, 11], :] = flipped[:, :, [11, 14], :]
        flipped[:, :, [15, 12], :] = flipped[:, :, [12, 15], :]
        flipped[:, :, [16, 13], :] = flipped[:, :, [13, 16], :]
        return flipped

    with torch.no_grad():
        for start in range(0, n_frames, stride):
            end = min(start + clip_len, n_frames)
            if end - start < clip_len:
                start = max(0, end - clip_len)

            clip = input_data[start:start + clip_len]
            clip_tensor = torch.from_numpy(clip).float().unsqueeze(0)

            if 'cuda' in device and torch.cuda.is_available():
                clip_tensor = clip_tensor.cuda()

            pred1 = model(clip_tensor)
            clip_flip = flip_data(clip_tensor)
            pred2_flip = model(clip_flip)
            pred2 = flip_data(pred2_flip)
            pred = (pred1 + pred2) / 2.0

            output = pred.cpu().numpy()[0]

            if start == 0:
                keep_end = min(stride + stride // 2, n_frames) if n_frames > clip_len else n_frames
                outputs.append(output[:keep_end])
            elif end >= n_frames:
                keep_start = clip_len - (n_frames - start)
                outputs.append(output[keep_start:])
                break
            else:
                keep_start = stride // 2
                keep_end = keep_start + stride
                outputs.append(output[keep_start:keep_end])

    keypoints_3d = np.concatenate(outputs, axis=0)[:n_frames]

    if timed:
        timings['3b_motionbert_inference'] = time.perf_counter() - t_infer
        timings['3_motionbert_total'] = time.perf_counter() - t0
        print(f"  Inference: {format_time(timings['3b_motionbert_inference'])}")

    # =========================================================================
    # STEP 4: Bone Length Normalization
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Bone Length Normalization")
    print("=" * 70)

    t0 = time.perf_counter()

    from trc_motionbert import normalize_bone_lengths
    keypoints_3d = normalize_bone_lengths(keypoints_3d)

    if timed:
        timings['4_normalize_bones'] = time.perf_counter() - t0
        print(f"  Time: {format_time(timings['4_normalize_bones'])}")

    # =========================================================================
    # STEP 5: Coordinate Transformation
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Coordinate Transform (MotionBERT -> OpenSim)")
    print("=" * 70)

    t0 = time.perf_counter()

    from trc_motionbert import transform_motionbert_to_opensim
    keypoints_osim = transform_motionbert_to_opensim(keypoints_3d, target_height, use_simple_transform=True)

    if timed:
        timings['5_transform_coords'] = time.perf_counter() - t0
        print(f"  Time: {format_time(timings['5_transform_coords'])}")

    # =========================================================================
    # STEP 6: Extract COCO17 Markers
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Extract COCO17 Markers")
    print("=" * 70)

    t0 = time.perf_counter()

    from trc_motionbert import extract_coco17_markers
    marker_names, marker_positions = extract_coco17_markers(keypoints_osim)

    if timed:
        timings['6_extract_markers'] = time.perf_counter() - t0
        print(f"  {len(marker_names)} markers extracted: {format_time(timings['6_extract_markers'])}")

    # =========================================================================
    # STEP 7: Export TRC File
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Export TRC File")
    print("=" * 70)

    t0 = time.perf_counter()

    from trc_motionbert import export_trc
    trc_path = output_dir / "motion.trc"
    export_trc(marker_positions, marker_names, str(trc_path), fps)

    pose3d_dir = output_dir / "pose-3d"
    pose3d_dir.mkdir(exist_ok=True)
    shutil.copy(trc_path, pose3d_dir / trc_path.name)

    if timed:
        timings['7_export_trc'] = time.perf_counter() - t0
        print(f"  Time: {format_time(timings['7_export_trc'])}")

    # =========================================================================
    # STEP 8: Pose2Sim/OpenSim (Scaling + IK)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Pose2Sim/OpenSim Scaling + Inverse Kinematics")
    print("=" * 70)

    t0 = time.perf_counter()

    config_dict = {
        'project': {
            'project_dir': str(output_dir),
            'multi_person': False,
            'participant_height': target_height,
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
            'default_height': target_height,
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

    if timed:
        timings['8_pose2sim_ik'] = time.perf_counter() - t0
        print(f"  Time: {format_time(timings['8_pose2sim_ik'])}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    if timed:
        print("\n" + "=" * 70)
        print("TIMING SUMMARY")
        print("=" * 70)

        total_time = sum(timings.values())

        print(f"\n{'Step':<50} {'Time':>12} {'%':>8}")
        print("-" * 70)

        for step, t in timings.items():
            pct = (t / total_time) * 100
            print(f"  {step:<48} {format_time(t):>12} {pct:>7.1f}%")

        print("-" * 70)
        print(f"  {'TOTAL':<48} {format_time(total_time):>12} {'100.0%':>8}")

        print(f"\nPerformance: {n_frames/total_time:.1f} FPS, {duration/total_time:.2f}x real-time")

    # Output files
    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    kinematics_dir = output_dir / "kinematics"
    if kinematics_dir.exists():
        for f in output_dir.rglob("*"):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  {f.relative_to(output_dir)} ({size_kb:.1f} KB)")

    mot_files = list(kinematics_dir.glob("*.mot")) if kinematics_dir.exists() else []
    osim_files = list(kinematics_dir.glob("*_scaled.osim")) if kinematics_dir.exists() else []

    if mot_files:
        print(f"\n*** Motion file: {mot_files[0]} ***")
    if osim_files:
        print(f"*** Scaled model: {osim_files[0]} ***")

    print("\n" + "=" * 70)
    print("DONE! Open the .osim file in OpenSim GUI, then load the .mot file")
    print("=" * 70)

    return timings


def main():
    parser = argparse.ArgumentParser(description='Full Pipeline: Video -> OpenSim')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default='output/pipeline_output', help='Output directory')
    parser.add_argument('--height', type=float, default=1.75, help='Target height in meters')
    parser.add_argument('--device', '-d', default='cuda:0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--no-timing', action='store_true', help='Disable timing output')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    run_pipeline(args.input, args.output, args.height, args.device, not args.no_timing)


if __name__ == "__main__":
    main()
