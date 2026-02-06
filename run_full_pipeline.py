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

# Shim for pyopensim -> opensim module aliasing
# pyopensim provides OpenSim bindings but separates API into submodules
# We need to expose everything at top level like the official opensim package
try:
    import opensim
except ImportError:
    # Use our custom shim that properly exposes all submodule contents
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.absolute()))
    import opensim_shim
    sys.modules['opensim'] = opensim_shim

import shutil
import time
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
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


def generate_output_dir_name(video_path: Path, base_dir: Path = None) -> Path:
    """Generate output directory name as output_YYYYMMDD_HHMMSS_videoname."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = video_path.stem  # filename without extension
    dir_name = f"output_{timestamp}_{video_name}"

    if base_dir is None:
        base_dir = video_path.parent

    return base_dir / dir_name


def generate_report(
    output_dir: Path,
    video_path: Path,
    video_info: Dict[str, Any],
    target_height: float,
    device: str,
    timings: Dict[str, float],
    corrections: Dict[str, Any],
    errors: List[str],
    warnings: List[str],
) -> Path:
    """Generate a detailed processing report."""

    report_path = output_dir / "processing_report.txt"
    json_path = output_dir / "processing_report.json"

    total_time = sum(timings.values()) if timings else 0
    n_frames = video_info.get('n_frames', 0)
    duration = video_info.get('duration', 0)

    # Text report (UTF-8 encoding for degree symbols etc.)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MOTIONBERT-OPENSIM PROCESSING REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Metadata
        f.write("METADATA\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Generated:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Input video:       {video_path}\n")
        f.write(f"  Output directory:  {output_dir}\n")
        f.write(f"  Target height:     {target_height} m\n")
        f.write(f"  Device:            {device}\n\n")

        # Video info
        f.write("VIDEO INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Resolution:        {video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}\n")
        f.write(f"  Frame rate:        {video_info.get('fps', 'N/A'):.2f} FPS\n")
        f.write(f"  Total frames:      {n_frames}\n")
        f.write(f"  Duration:          {duration:.2f} seconds\n\n")

        # Corrections applied
        f.write("CORRECTIONS APPLIED\n")
        f.write("-" * 70 + "\n")
        if corrections:
            if 'lean_correction' in corrections:
                lc = corrections['lean_correction']
                f.write(f"  Forward lean correction:\n")
                f.write(f"    Original spine direction:  [{lc.get('original_spine', 'N/A')}]\n")
                f.write(f"    Lean angle detected:       {lc.get('lean_angle_deg', 'N/A'):.1f}°\n")
                f.write(f"    Correction applied:        {lc.get('correction_deg', 'N/A'):.1f}° around X-axis\n")
                f.write(f"    Residual lean:             {lc.get('residual_deg', 'N/A'):.1f}°\n")
            if 'bone_normalization' in corrections:
                bn = corrections['bone_normalization']
                f.write(f"  Bone length normalization:\n")
                f.write(f"    CV before:                 {bn.get('cv_before', 'N/A'):.1f}%\n")
                f.write(f"    CV after:                  {bn.get('cv_after', 'N/A'):.1f}%\n")
            if 'scaling' in corrections:
                sc = corrections['scaling']
                f.write(f"  Height scaling:\n")
                f.write(f"    Original height:           {sc.get('original_height', 'N/A'):.3f} m\n")
                f.write(f"    Scale factor:              {sc.get('scale_factor', 'N/A'):.3f}\n")
                f.write(f"    Final height:              {sc.get('final_height', 'N/A'):.3f} m\n")
        else:
            f.write("  No correction data recorded\n")
        f.write("\n")

        # Processing times
        f.write("PROCESSING TIMES\n")
        f.write("-" * 70 + "\n")
        if timings:
            f.write(f"  {'Step':<45} {'Time':>12} {'%':>8}\n")
            f.write("  " + "-" * 65 + "\n")
            for step, t in timings.items():
                pct = (t / total_time) * 100 if total_time > 0 else 0
                f.write(f"  {step:<45} {format_time(t):>12} {pct:>7.1f}%\n")
            f.write("  " + "-" * 65 + "\n")
            f.write(f"  {'TOTAL':<45} {format_time(total_time):>12} {'100.0%':>8}\n\n")

            if n_frames > 0 and total_time > 0:
                f.write(f"  Processing speed:  {n_frames/total_time:.1f} FPS\n")
                f.write(f"  Real-time factor:  {duration/total_time:.2f}x\n\n")
        else:
            f.write("  No timing data recorded\n\n")

        # Errors and warnings
        f.write("ERRORS\n")
        f.write("-" * 70 + "\n")
        if errors:
            for err in errors:
                f.write(f"  [ERROR] {err}\n")
        else:
            f.write("  No errors\n")
        f.write("\n")

        f.write("WARNINGS\n")
        f.write("-" * 70 + "\n")
        if warnings:
            for warn in warnings:
                f.write(f"  [WARNING] {warn}\n")
        else:
            f.write("  No warnings\n")
        f.write("\n")

        # Output files
        f.write("OUTPUT FILES\n")
        f.write("-" * 70 + "\n")
        for fpath in sorted(output_dir.rglob("*")):
            if fpath.is_file() and fpath.name != "processing_report.txt" and fpath.name != "processing_report.json":
                size_kb = fpath.stat().st_size / 1024
                f.write(f"  {fpath.relative_to(output_dir)} ({size_kb:.1f} KB)\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # JSON report for programmatic access
    json_report = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "input_video": str(video_path),
            "output_directory": str(output_dir),
            "target_height": float(target_height),
            "device": device,
        },
        "video_info": convert_to_native(video_info),
        "corrections": convert_to_native(corrections),
        "timings": {k: round(float(v), 4) for k, v in timings.items()} if timings else {},
        "total_time": round(float(total_time), 4),
        "processing_fps": round(float(n_frames / total_time), 2) if total_time > 0 else 0,
        "realtime_factor": round(float(duration / total_time), 2) if total_time > 0 else 0,
        "errors": errors,
        "warnings": warnings,
    }

    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)

    return report_path


def run_pipeline(video_path: str, output_dir: str = None, target_height: float = 1.75,
                 device: str = 'cuda:0', timed: bool = True,
                 apply_filter: bool = False, filter_cutoff: float = 6.0,
                 export_fbx: bool = False, blender_path: str = None):
    """Run the full video to OpenSim pipeline.

    Args:
        video_path: Path to input video
        output_dir: Output directory (auto-generated if None)
        target_height: Target height in meters
        device: Computation device (cuda:0 or cpu)
        timed: Whether to record timing information
        apply_filter: Whether to apply Butterworth low-pass filter
        filter_cutoff: Filter cutoff frequency in Hz (default 6.0)
        export_fbx: Whether to export FBX using Blender
        blender_path: Path to Blender executable (auto-detected if None)
    """

    video_path = Path(video_path)

    # Auto-generate output directory name if not specified or if using default
    if output_dir is None or output_dir == 'output/pipeline_output':
        output_dir = generate_output_dir_name(video_path)
    else:
        output_dir = Path(output_dir)

    timings = {}
    corrections = {}
    errors = []
    warnings = []

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
    if apply_filter:
        print(f"Butterworth filter: ON (cutoff={filter_cutoff} Hz)")
    else:
        print(f"Butterworth filter: OFF")

    # Check CUDA
    if 'cuda' in device:
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            warnings.append("CUDA not available, falling back to CPU")
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

    # Store video info for report
    video_info = {
        'width': width,
        'height': height,
        'fps': fps,
        'n_frames': n_frames,
        'duration': duration,
    }

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
        # Track how many frames we've collected
        frames_collected = 0

        for start in range(0, n_frames, stride):
            end = min(start + clip_len, n_frames)
            original_start = start

            # If clip is shorter than clip_len, shift start back
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

            if original_start == 0:
                # First window: keep from start to middle of overlap region
                keep_end = min(stride + stride // 2, n_frames) if n_frames > clip_len else n_frames
                outputs.append(output[:keep_end])
                frames_collected = keep_end
            elif frames_collected >= n_frames:
                # Already have all frames, stop
                break
            else:
                # Calculate how many frames we still need
                frames_needed = n_frames - frames_collected

                # Calculate offset within this window for where our needed frames start
                # The window covers [start, start+clip_len), we need frames starting at frames_collected
                offset_in_window = frames_collected - start

                # Take only what we need
                keep_count = min(frames_needed, clip_len - offset_in_window)
                outputs.append(output[offset_in_window:offset_in_window + keep_count])
                frames_collected += keep_count

                if frames_collected >= n_frames:
                    break

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
    # STEP 4b: Forward Lean Correction
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4b: Forward Lean Correction")
    print("=" * 70)

    t0 = time.perf_counter()

    from trc_motionbert import correct_forward_lean
    keypoints_3d, lean_correction_info = correct_forward_lean(keypoints_3d)
    corrections['lean_correction'] = lean_correction_info

    if timed:
        timings['4b_lean_correction'] = time.perf_counter() - t0
        print(f"  Time: {format_time(timings['4b_lean_correction'])}")

    # =========================================================================
    # STEP 4c: Butterworth Smoothing (Optional)
    # =========================================================================
    if apply_filter:
        print("\n" + "=" * 70)
        print("STEP 4c: Butterworth Low-Pass Filter")
        print("=" * 70)

        t0 = time.perf_counter()

        from trc_motionbert import butterworth_filter_keypoints
        keypoints_3d, filter_info = butterworth_filter_keypoints(
            keypoints_3d, fps=fps, cutoff_freq=filter_cutoff, order=4
        )
        corrections['butterworth_filter'] = filter_info

        if timed:
            timings['4c_butterworth_filter'] = time.perf_counter() - t0
            print(f"  Time: {format_time(timings['4c_butterworth_filter'])}")

    # =========================================================================
    # STEP 5: Coordinate Transformation
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Coordinate Transform (MotionBERT -> OpenSim)")
    print("=" * 70)

    t0 = time.perf_counter()

    from trc_motionbert import transform_motionbert_to_opensim
    keypoints_osim, scaling_info = transform_motionbert_to_opensim(keypoints_3d, target_height, use_simple_transform=True)
    corrections['scaling'] = scaling_info

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
    osim_files = list(kinematics_dir.glob("*.osim")) if kinematics_dir.exists() else []

    if mot_files:
        print(f"\n*** Motion file: {mot_files[0]} ***")
    if osim_files:
        print(f"*** Scaled model: {osim_files[0]} ***")

    # Generate report
    report_path = generate_report(
        output_dir=output_dir,
        video_path=video_path,
        video_info=video_info,
        target_height=target_height,
        device=device,
        timings=timings,
        corrections=corrections,
        errors=errors,
        warnings=warnings,
    )
    print(f"\n*** Report: {report_path} ***")

    # =========================================================================
    # STEP 9 (Optional): FBX Export via Blender
    # =========================================================================
    fbx_path = None
    if export_fbx and mot_files and osim_files:
        print("\n" + "=" * 70)
        print("STEP 9: FBX Export (Blender)")
        print("=" * 70)

        t0 = time.perf_counter()

        # Find Blender
        if blender_path is None:
            # Try common locations
            blender_paths = [
                Path("C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"),
                Path("C:/Program Files/Blender Foundation/Blender 4.0/blender.exe"),
                Path("C:/Program Files/Blender Foundation/Blender 3.6/blender.exe"),
                Path("/Applications/Blender.app/Contents/MacOS/Blender"),
                Path("/usr/bin/blender"),
            ]
            for bp in blender_paths:
                if bp.exists():
                    blender_path = str(bp)
                    break

        if blender_path and Path(blender_path).exists():
            export_script = PROJECT_ROOT / "export_fbx_skely.py"
            skely_template = PROJECT_ROOT.parent / "Import_OS4_Patreon_Aitor_Skely.blend"
            fbx_path = output_dir / "motion.fbx"

            if not skely_template.exists():
                warnings.append(f"Skeleton template not found: {skely_template}")
                print(f"  WARNING: Skeleton template not found: {skely_template}")
                print(f"  Place Import_OS4_Patreon_Aitor_Skely.blend in {PROJECT_ROOT.parent}")
            else:
                import subprocess
                cmd = [
                    blender_path,
                    "--background",
                    str(skely_template),
                    "--python", str(export_script),
                    "--",
                    "--mot", str(mot_files[0]),
                    "--output", str(fbx_path),
                    "--fps", str(int(fps)),
                ]

                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                    if result.returncode == 0 and fbx_path.exists():
                        print(f"  FBX exported: {fbx_path}")
                        if timed:
                            timings['9_fbx_export'] = time.perf_counter() - t0
                            print(f"  Time: {format_time(timings['9_fbx_export'])}")
                    else:
                        errors.append(f"FBX export failed: {result.stderr}")
                        print(f"  ERROR: FBX export failed")
                        if result.stderr:
                            print(f"  {result.stderr[:500]}")
                except subprocess.TimeoutExpired:
                    errors.append("FBX export timed out")
                    print("  ERROR: FBX export timed out")
                except Exception as e:
                    errors.append(f"FBX export error: {e}")
                    print(f"  ERROR: {e}")
        else:
            warnings.append("Blender not found, skipping FBX export")
            print("  WARNING: Blender not found, skipping FBX export")
            print("  Install Blender from https://www.blender.org/download/")

    print("\n" + "=" * 70)
    print("DONE! Open the .osim file in OpenSim GUI, then load the .mot file")
    if fbx_path and fbx_path.exists():
        print(f"FBX file available: {fbx_path}")
    print("=" * 70)

    return timings, corrections, errors, warnings


def main():
    parser = argparse.ArgumentParser(
        description='Full Pipeline: Video -> OpenSim',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-generate output folder (output_YYYYMMDD_HHMMSS_videoname)
    python run_full_pipeline.py --input video.mp4 --height 1.75

    # Specify output folder
    python run_full_pipeline.py --input video.mp4 --output results/my_output --height 1.75
""")
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: auto-generated as output_YYYYMMDD_HHMMSS_videoname)')
    parser.add_argument('--height', type=float, default=1.75, help='Target height in meters')
    parser.add_argument('--device', '-d', default='cuda:0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--no-timing', action='store_true', help='Disable timing output')
    parser.add_argument('--filter', '-f', action='store_true',
                        help='Apply Butterworth low-pass filter to smooth keypoints (reduces jitter)')
    parser.add_argument('--filter-cutoff', type=float, default=6.0,
                        help='Filter cutoff frequency in Hz (default: 6.0, lower=smoother)')
    parser.add_argument('--export-fbx', action='store_true',
                        help='Export motion to FBX format using Blender (requires Blender installed)')
    parser.add_argument('--blender-path', default=None,
                        help='Path to Blender executable (auto-detected if not specified)')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    run_pipeline(args.input, args.output, args.height, args.device, not args.no_timing,
                 apply_filter=args.filter, filter_cutoff=args.filter_cutoff,
                 export_fbx=args.export_fbx, blender_path=args.blender_path)


if __name__ == "__main__":
    main()
