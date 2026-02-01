"""
Full End-to-End Pipeline: Video -> OpenSim Joint Angles (Timed)

This script runs the complete pipeline from raw video to OpenSim joint angles,
timing every step to provide performance metrics.

Pipeline Steps:
1. [VideoPoseEstimation] YOLOv8-Pose: 2D keypoint detection (COCO17 format)
2. [VideoPoseEstimation] COCO17 -> H36M: Keypoint format conversion
3. [VideoPoseEstimation] MotionBERT: 2D -> 3D lifting (H36M 17 joints)
4. [RTMPose3DOpenSim] Bone normalization: Fix monocular depth inconsistency
5. [RTMPose3DOpenSim] Coordinate transform: MotionBERT -> OpenSim axes
6. [RTMPose3DOpenSim] Marker extraction: H36M -> COCO17 (14 markers)
7. [RTMPose3DOpenSim] TRC export: Write marker trajectories
8. [RTMPose3DOpenSim] Pose2Sim/OpenSim: Scaling + Inverse Kinematics

Requirements:
- Run in 'body4d' conda environment (has GPU support + all dependencies)
- MotionBERT checkpoint must be downloaded
- Pose2Sim must be installed
"""

import os
import sys
import shutil
import time
import pickle
from pathlib import Path
from typing import Dict, Tuple
import argparse

import numpy as np
import cv2
import torch
from tqdm import tqdm

# Add paths
PROJECT_ROOT = Path(__file__).parent.absolute()
VIDEOPOSEESTIMATION_PATH = Path("E:/VideoPoseEstimation")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(VIDEOPOSEESTIMATION_PATH))
sys.path.insert(0, str(VIDEOPOSEESTIMATION_PATH / "MotionBERT"))


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds*1000000:.1f} µs"
    elif seconds < 1:
        return f"{seconds*1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        return f"{seconds/60:.1f} min"


class TimedPipeline:
    """Full pipeline with timing for each step."""

    def __init__(self, device: str = 'cuda:0'):
        self.device = device
        self.timings: Dict[str, float] = {}

        # Check CUDA
        if 'cuda' in device:
            if torch.cuda.is_available():
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print("WARNING: CUDA not available, using CPU")
                self.device = 'cpu'

    def run(self, video_path: str, output_dir: str, target_height: float = 1.75) -> Dict:
        """Run the full pipeline with timing."""

        video_path = Path(video_path)
        output_dir = Path(output_dir)

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
        print(f"Device: {self.device}")

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

        # =====================================================================
        # STEP 1: YOLOv8-Pose 2D Detection
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 1: YOLOv8-Pose 2D Keypoint Detection (COCO17)")
        print("=" * 70)

        t0 = time.perf_counter()

        from ultralytics import YOLO
        yolo_model = YOLO('yolov8m-pose.pt')
        yolo_device = 0 if 'cuda' in self.device and torch.cuda.is_available() else 'cpu'

        t_load = time.perf_counter()
        self.timings['1a_yolo_load'] = t_load - t0
        print(f"  Model loaded: {format_time(self.timings['1a_yolo_load'])}")

        # Process video frames
        cap = cv2.VideoCapture(str(video_path))
        all_keypoints_coco = []
        all_scores = []

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

        t1 = time.perf_counter()
        self.timings['1b_yolo_inference'] = t1 - t_load
        self.timings['1_yolo_total'] = t1 - t0
        print(f"  Inference: {format_time(self.timings['1b_yolo_inference'])}")
        print(f"  Total: {format_time(self.timings['1_yolo_total'])}")

        # =====================================================================
        # STEP 2: COCO17 -> H36M Conversion
        # =====================================================================
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

        t1 = time.perf_counter()
        self.timings['2_coco_to_h36m'] = t1 - t0
        print(f"  Converted {n_frames} frames: {format_time(self.timings['2_coco_to_h36m'])}")

        # =====================================================================
        # STEP 3: MotionBERT 2D -> 3D Lifting
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 3: MotionBERT 2D -> 3D Lifting")
        print("=" * 70)

        t0 = time.perf_counter()

        # Import MotionBERT
        from functools import partial
        import torch.nn as nn
        from lib.model.DSTformer import DSTformer
        from lib.utils.tools import get_config

        # Load config
        config_path = VIDEOPOSEESTIMATION_PATH / "MotionBERT" / "configs" / "pose3d" / "MB_ft_h36m_global_lite.yaml"
        args = get_config(str(config_path))

        # Create model
        model = DSTformer(
            dim_in=3,  # x, y, confidence
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
        ckpt_path = VIDEOPOSEESTIMATION_PATH / "MotionBERT" / "checkpoint" / "pose3d" / "FT_MB_lite_MB_ft_h36m_global_lite" / "best_epoch.bin"
        checkpoint = torch.load(str(ckpt_path), map_location='cpu')
        state_dict = checkpoint['model_pos']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)

        if 'cuda' in self.device and torch.cuda.is_available():
            model = model.cuda()
        model.eval()

        t_load = time.perf_counter()
        self.timings['3a_motionbert_load'] = t_load - t0
        print(f"  Model loaded: {format_time(self.timings['3a_motionbert_load'])}")

        # Prepare input
        kpts = keypoints_h36m.copy().astype(np.float32)
        confs = scores_h36m.copy().astype(np.float32)

        # Normalize to [-1, 1]
        kpts[..., 0] = (kpts[..., 0] - width / 2) / (min(width, height) / 2)
        kpts[..., 1] = (kpts[..., 1] - height / 2) / (min(width, height) / 2)

        input_data = np.concatenate([kpts, confs[..., np.newaxis]], axis=-1)

        # Pad if needed
        clip_len = args.maxlen  # 243
        if n_frames < clip_len:
            pad_len = clip_len - n_frames
            input_data = np.concatenate([
                input_data,
                np.tile(input_data[-1:], (pad_len, 1, 1))
            ], axis=0)

        # Inference
        t_infer_start = time.perf_counter()

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

                if 'cuda' in self.device and torch.cuda.is_available():
                    clip_tensor = clip_tensor.cuda()

                # With flip augmentation
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

        t1 = time.perf_counter()
        self.timings['3b_motionbert_inference'] = t1 - t_infer_start
        self.timings['3_motionbert_total'] = t1 - t0
        print(f"  Inference: {format_time(self.timings['3b_motionbert_inference'])}")
        print(f"  Total: {format_time(self.timings['3_motionbert_total'])}")

        # Save intermediate pickle (for debugging)
        pkl_path = output_dir / "poses_3d.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'keypoints_3d': keypoints_3d,
                'fps': fps,
                'video_name': video_path.stem,
            }, f)

        # =====================================================================
        # STEP 4: Bone Length Normalization
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 4: Bone Length Normalization")
        print("=" * 70)

        t0 = time.perf_counter()

        from trc_motionbert import normalize_bone_lengths
        keypoints_3d = normalize_bone_lengths(keypoints_3d)

        t1 = time.perf_counter()
        self.timings['4_normalize_bones'] = t1 - t0
        print(f"  Time: {format_time(self.timings['4_normalize_bones'])}")

        # =====================================================================
        # STEP 5: Coordinate Transformation (MotionBERT -> OpenSim)
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 5: Coordinate Transform (MotionBERT -> OpenSim)")
        print("=" * 70)

        t0 = time.perf_counter()

        from trc_motionbert import transform_motionbert_to_opensim
        keypoints_osim = transform_motionbert_to_opensim(keypoints_3d, target_height, use_simple_transform=True)

        t1 = time.perf_counter()
        self.timings['5_transform_coords'] = t1 - t0
        print(f"  Time: {format_time(self.timings['5_transform_coords'])}")

        # =====================================================================
        # STEP 6: Extract COCO17 Markers
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 6: Extract COCO17 Markers (including synthetic nose)")
        print("=" * 70)

        t0 = time.perf_counter()

        from trc_motionbert import extract_coco17_markers
        marker_names, marker_positions = extract_coco17_markers(keypoints_osim)

        t1 = time.perf_counter()
        self.timings['6_extract_markers'] = t1 - t0
        print(f"  {len(marker_names)} markers extracted")
        print(f"  Time: {format_time(self.timings['6_extract_markers'])}")

        # =====================================================================
        # STEP 7: Export TRC File
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 7: Export TRC File")
        print("=" * 70)

        t0 = time.perf_counter()

        from trc_motionbert import export_trc
        trc_path = output_dir / "motion.trc"
        export_trc(marker_positions, marker_names, str(trc_path), fps)

        # Setup Pose2Sim structure
        pose3d_dir = output_dir / "pose-3d"
        pose3d_dir.mkdir(exist_ok=True)
        shutil.copy(trc_path, pose3d_dir / trc_path.name)

        t1 = time.perf_counter()
        self.timings['7_export_trc'] = t1 - t0
        print(f"  Time: {format_time(self.timings['7_export_trc'])}")

        # =====================================================================
        # STEP 8: Pose2Sim/OpenSim (Scaling + IK)
        # =====================================================================
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

        t1 = time.perf_counter()
        self.timings['8_pose2sim_ik'] = t1 - t0
        print(f"  Time: {format_time(self.timings['8_pose2sim_ik'])}")

        # =====================================================================
        # SUMMARY
        # =====================================================================
        self._print_summary(n_frames, fps, duration, output_dir)

        return self.timings

    def _print_summary(self, n_frames: int, fps: float, duration: float, output_dir: Path):
        """Print timing summary and performance metrics."""

        print("\n" + "=" * 70)
        print("TIMING SUMMARY")
        print("=" * 70)

        total_time = sum(self.timings.values())

        # Group timings
        groups = {
            'VideoPoseEstimation': ['1_yolo_total', '2_coco_to_h36m', '3_motionbert_total'],
            'RTMPose3DOpenSim': ['4_normalize_bones', '5_transform_coords', '6_extract_markers', '7_export_trc', '8_pose2sim_ik'],
        }

        print(f"\n{'Step':<50} {'Time':>12} {'%':>8}")
        print("-" * 70)

        for step, t in self.timings.items():
            pct = (t / total_time) * 100
            print(f"  {step:<48} {format_time(t):>12} {pct:>7.1f}%")

        print("-" * 70)
        print(f"  {'TOTAL':<48} {format_time(total_time):>12} {'100.0%':>8}")

        # Group totals
        print("\n" + "=" * 70)
        print("BY COMPONENT")
        print("=" * 70)

        for group_name, steps in groups.items():
            group_time = sum(self.timings.get(s, 0) for s in steps)
            pct = (group_time / total_time) * 100
            print(f"  {group_name:<48} {format_time(group_time):>12} {pct:>7.1f}%")

        # Performance metrics
        print("\n" + "=" * 70)
        print("PERFORMANCE METRICS")
        print("=" * 70)

        print(f"\nVideo info:")
        print(f"  Frames: {n_frames}")
        print(f"  Original FPS: {fps:.1f}")
        print(f"  Duration: {duration:.2f} seconds")

        # By stage
        video_pose_time = sum(self.timings.get(s, 0) for s in groups['VideoPoseEstimation'])
        rtmpose_time = sum(self.timings.get(s, 0) for s in groups['RTMPose3DOpenSim'])

        print(f"\nVideoPoseEstimation (video -> H36M 3D):")
        print(f"  Total time: {format_time(video_pose_time)}")
        print(f"  Throughput: {n_frames/video_pose_time:.1f} frames/sec")
        print(f"  Real-time factor: {duration/video_pose_time:.2f}x")

        print(f"\nRTMPose3DOpenSim (H36M 3D -> OpenSim):")
        print(f"  Total time: {format_time(rtmpose_time)}")
        print(f"  Throughput: {n_frames/rtmpose_time:.1f} frames/sec")
        print(f"  Real-time factor: {duration/rtmpose_time:.2f}x")

        print(f"\nEnd-to-end (video -> joint angles):")
        print(f"  Total time: {format_time(total_time)}")
        print(f"  Throughput: {n_frames/total_time:.1f} frames/sec")
        print(f"  Real-time factor: {duration/total_time:.2f}x")

        # Bottleneck analysis
        print("\n" + "=" * 70)
        print("BOTTLENECK ANALYSIS")
        print("=" * 70)

        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 3 slowest steps:")
        for i, (step, t) in enumerate(sorted_timings[:3]):
            pct = (t / total_time) * 100
            print(f"  {i+1}. {step}: {format_time(t)} ({pct:.1f}%)")

        # Output files
        print("\n" + "=" * 70)
        print("OUTPUT FILES")
        print("=" * 70)

        kinematics_dir = output_dir / "kinematics"
        if kinematics_dir.exists():
            print("\nFiles created:")
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


def main():
    parser = argparse.ArgumentParser(description='Full Pipeline: Video -> OpenSim (Timed)')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default='output/full_pipeline', help='Output directory')
    parser.add_argument('--height', type=float, default=1.75, help='Target height in meters')
    parser.add_argument('--device', '-d', default='cuda:0', help='Device (cuda:0 or cpu)')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    pipeline = TimedPipeline(device=args.device)
    pipeline.run(args.input, args.output, args.height)


if __name__ == "__main__":
    main()
