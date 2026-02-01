"""
Video to 3D Pose Estimation Pipeline (Standalone)

This script processes a video file and extracts 3D poses using:
1. YOLOv8-Pose for 2D keypoint detection
2. COCO to H36M keypoint conversion
3. MotionBERT for 2D to 3D lifting

Output: Pickle file with H36M 3D keypoints ready for OpenSim conversion.

Usage:
    python process_video.py --input input/video.mp4 --output output/ --height 1.75
"""

import os
import sys
import argparse
import json
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import local modules
from utils.keypoint_converter import COCOtoH36MConverter


def check_checkpoint():
    """Check if the pose3d checkpoint exists."""
    ckpt_path = PROJECT_ROOT / "MotionBERT" / "checkpoint" / "pose3d" / "FT_MB_lite_MB_ft_h36m_global_lite" / "best_epoch.bin"
    if ckpt_path.exists():
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        print(f"Pose3D checkpoint found: {ckpt_path.name} ({size_mb:.1f} MB)")
        return str(ckpt_path)

    print("\n" + "="*60)
    print("MISSING: Pose3D checkpoint not found!")
    print("="*60)
    print(f"\nExpected: {ckpt_path}")
    print("\nPlease download from:")
    print("  https://1drv.ms/f/s!AvAdh0LSjEOlgT67igq_cIoYvO2y?e=bfEc73")
    print("\nLook for: FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin")
    print(f"Place in: {ckpt_path.parent}")
    print("="*60 + "\n")
    return None


class YOLOPoseDetector:
    """Pose detector using Ultralytics YOLOv8-Pose."""

    def __init__(self, device: str = 'cuda:0'):
        """Initialize pose detector."""
        self.device = device

        # Determine device
        if 'cuda' in device and torch.cuda.is_available():
            self.yolo_device = 0  # YOLO uses integer for GPU
            print(f"YOLOv8-Pose using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.yolo_device = 'cpu'
            print("YOLOv8-Pose using CPU")

        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8m-pose.pt')
            print("YOLOv8-Pose initialized successfully!")
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect 2D poses in an image.

        Returns:
            keypoints: (N, 17, 2) array of xy coordinates
            scores: (N, 17) array of confidence scores
        """
        results = self.model(image, verbose=False, device=self.yolo_device)

        all_keypoints = []
        all_scores = []

        for result in results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                kpts_data = result.keypoints

                if hasattr(kpts_data, 'xy') and kpts_data.xy is not None:
                    kpts = kpts_data.xy.cpu().numpy()
                elif hasattr(kpts_data, 'data') and kpts_data.data is not None:
                    kpts = kpts_data.data[:, :, :2].cpu().numpy()
                else:
                    continue

                if hasattr(kpts_data, 'conf') and kpts_data.conf is not None:
                    confs = kpts_data.conf.cpu().numpy()
                elif hasattr(kpts_data, 'data') and kpts_data.data is not None:
                    confs = kpts_data.data[:, :, 2].cpu().numpy()
                else:
                    confs = np.ones((kpts.shape[0], kpts.shape[1]))

                for i in range(len(kpts)):
                    all_keypoints.append(kpts[i])
                    all_scores.append(confs[i])

        if not all_keypoints:
            return np.zeros((0, 17, 2)), np.zeros((0, 17))

        return np.array(all_keypoints), np.array(all_scores)


class MotionBERTPose3D:
    """MotionBERT 3D pose lifter."""

    def __init__(self, device: str = 'cuda:0'):
        """Initialize MotionBERT pose3d model."""
        self.device = device
        self.model = None

        # Check for checkpoint
        ckpt_path = check_checkpoint()
        if ckpt_path is None:
            print("MotionBERT 3D lifting disabled - checkpoint missing")
            return

        # Config path
        config_path = PROJECT_ROOT / "MotionBERT" / "configs" / "pose3d" / "MB_ft_h36m_global_lite.yaml"

        self._load_model(ckpt_path, config_path)

    def _load_model(self, checkpoint_path: str, config_path: Path):
        """Load MotionBERT model with correct configuration."""
        try:
            # Add MotionBERT to path
            motionbert_path = PROJECT_ROOT / "MotionBERT"
            sys.path.insert(0, str(motionbert_path))

            from lib.model.DSTformer import DSTformer
            from lib.utils.tools import get_config

            # Load config
            if config_path.exists():
                args = get_config(str(config_path))
                print(f"Loaded config: {config_path.name}")
            else:
                # Fallback config for lite model
                from easydict import EasyDict as edict
                args = edict({
                    'dim_feat': 256,
                    'dim_rep': 512,
                    'depth': 5,
                    'num_heads': 8,
                    'mlp_ratio': 4,
                    'maxlen': 243,
                    'num_joints': 17,
                    'no_conf': False,
                    'flip': True,
                    'rootrel': False,
                })
                print("Using default lite config")

            # Store config for inference
            self.args = args
            self.clip_len = getattr(args, 'maxlen', 243)
            self.use_flip = getattr(args, 'flip', True)
            self.rootrel = getattr(args, 'rootrel', False)
            self.no_conf = getattr(args, 'no_conf', False)

            # Create model
            dim_in = 2 if self.no_conf else 3

            self.model = DSTformer(
                dim_in=dim_in,
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
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # The pose3d checkpoint uses 'model_pos' key
            if 'model_pos' in checkpoint:
                state_dict = checkpoint['model_pos']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Handle DataParallel prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v

            self.model.load_state_dict(new_state_dict, strict=True)

            # Move to device
            if 'cuda' in self.device and torch.cuda.is_available():
                self.model = self.model.cuda()
                print(f"Model on GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.model = self.model.to('cpu')
                print("Model on CPU")

            self.model.eval()
            print("MotionBERT pose3d model loaded successfully!")

        except Exception as e:
            print(f"Error loading MotionBERT: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def _flip_data(self, data):
        """Flip keypoints horizontally for test-time augmentation."""
        flipped = data.clone()
        flipped[..., 0] = -flipped[..., 0]

        # Swap left-right joints
        flipped[:, :, [1, 4], :] = flipped[:, :, [4, 1], :]
        flipped[:, :, [2, 5], :] = flipped[:, :, [5, 2], :]
        flipped[:, :, [3, 6], :] = flipped[:, :, [6, 3], :]
        flipped[:, :, [14, 11], :] = flipped[:, :, [11, 14], :]
        flipped[:, :, [15, 12], :] = flipped[:, :, [12, 15], :]
        flipped[:, :, [16, 13], :] = flipped[:, :, [13, 16], :]

        return flipped

    def lift(self, keypoints_2d: np.ndarray, scores: np.ndarray,
             image_width: int, image_height: int) -> np.ndarray:
        """
        Lift 2D keypoints to 3D using MotionBERT.

        Args:
            keypoints_2d: (T, 17, 2) array of xy coordinates in H36M format
            scores: (T, 17) array of confidence scores
            image_width, image_height: Original image dimensions

        Returns:
            keypoints_3d: (T, 17, 3) array of xyz coordinates
        """
        if self.model is None:
            print("Warning: MotionBERT not available")
            return np.concatenate([keypoints_2d, np.zeros_like(keypoints_2d[..., :1])], axis=-1)

        T = keypoints_2d.shape[0]

        # Normalize keypoints to [-1, 1] range
        kpts = keypoints_2d.copy().astype(np.float32)
        confs = scores.copy().astype(np.float32)

        kpts[..., 0] = (kpts[..., 0] - image_width / 2) / (min(image_width, image_height) / 2)
        kpts[..., 1] = (kpts[..., 1] - image_height / 2) / (min(image_width, image_height) / 2)

        # Combine keypoints with confidence
        if self.no_conf:
            input_data = kpts
        else:
            input_data = np.concatenate([kpts, confs[..., np.newaxis]], axis=-1)

        # Process in clips
        clip_len = self.clip_len
        outputs = []

        # Pad if needed
        if T < clip_len:
            pad_len = clip_len - T
            input_data = np.concatenate([
                input_data,
                np.tile(input_data[-1:], (pad_len, 1, 1))
            ], axis=0)

        stride = clip_len // 2

        with torch.no_grad():
            for start in range(0, T, stride):
                end = min(start + clip_len, T)
                if end - start < clip_len:
                    start = max(0, end - clip_len)

                clip = input_data[start:start + clip_len]
                clip_tensor = torch.from_numpy(clip).float().unsqueeze(0)

                if 'cuda' in self.device and torch.cuda.is_available():
                    clip_tensor = clip_tensor.cuda()

                # Forward pass with flip augmentation
                if self.use_flip:
                    pred1 = self.model(clip_tensor)
                    clip_flip = self._flip_data(clip_tensor)
                    pred2_flip = self.model(clip_flip)
                    pred2 = self._flip_data(pred2_flip)
                    pred = (pred1 + pred2) / 2.0
                else:
                    pred = self.model(clip_tensor)

                if self.rootrel:
                    pred[:, :, 0, :] = 0
                else:
                    pred[:, 0, 0, 2] = 0

                output = pred.cpu().numpy()[0]

                if start == 0:
                    keep_end = min(stride + stride // 2, T) if T > clip_len else T
                    outputs.append(output[:keep_end])
                elif end >= T:
                    keep_start = clip_len - (T - start)
                    outputs.append(output[keep_start:])
                    break
                else:
                    keep_start = stride // 2
                    keep_end = keep_start + stride
                    outputs.append(output[keep_start:keep_end])

        result = np.concatenate(outputs, axis=0)[:T]
        return result


class VideoPoseEstimationPipeline:
    """Complete video to 3D pose estimation pipeline."""

    def __init__(self, device: str = 'cuda:0'):
        """Initialize pipeline components."""
        self.device = device

        if 'cuda' in device:
            if torch.cuda.is_available():
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print("WARNING: CUDA not available, using CPU")
                self.device = 'cpu'

        print(f"\nUsing device: {self.device}")

        self.detector = YOLOPoseDetector(device=self.device)
        self.converter = COCOtoH36MConverter()
        self.lifter = MotionBERTPose3D(device=self.device)

    def process_video(self, video_path: str, output_dir: str,
                      person_idx: int = 0) -> Dict:
        """Process video and extract 3D poses."""
        start_time = time.time()

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nVideo: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")

        # Prepare output
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract 2D poses
        all_keypoints_2d = []
        all_keypoints_h36m = []
        all_scores = []

        print("\n[1/2] Extracting 2D poses with YOLOv8-Pose...")
        pbar = tqdm(total=total_frames, desc="Detection")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints, scores = self.detector.detect(frame)

            if len(keypoints) > 0:
                idx = min(person_idx, len(keypoints) - 1)
                kpts = keypoints[idx]
                scrs = scores[idx]

                h36m_kpts, h36m_scores = self.converter.convert(kpts, scrs)

                all_keypoints_2d.append(kpts)
                all_keypoints_h36m.append(h36m_kpts)
                all_scores.append(h36m_scores)
            else:
                all_keypoints_2d.append(np.zeros((17, 2)))
                all_keypoints_h36m.append(np.zeros((17, 2)))
                all_scores.append(np.zeros(17))

            pbar.update(1)

        pbar.close()
        cap.release()

        keypoints_2d = np.array(all_keypoints_2d)
        keypoints_h36m = np.array(all_keypoints_h36m)
        scores = np.array(all_scores)

        # Lift to 3D
        keypoints_3d = None
        if self.lifter.model is not None:
            print("\n[2/2] Lifting to 3D with MotionBERT...")
            lift_start = time.time()
            keypoints_3d = self.lifter.lift(keypoints_h36m, scores, width, height)
            lift_time = time.time() - lift_start
            print(f"3D lifting completed in {lift_time:.2f}s")
        else:
            print("\n[2/2] Skipping 3D lifting - model not available")

        total_time = time.time() - start_time

        # Save results
        video_name = Path(video_path).stem
        results = {
            'video_name': video_name,
            'video_path': str(video_path),
            'resolution': (width, height),
            'fps': fps,
            'total_frames': total_frames,
            'keypoints_2d_coco': keypoints_2d.tolist(),
            'keypoints_2d_h36m': keypoints_h36m.tolist(),
            'keypoints_3d': keypoints_3d.tolist() if keypoints_3d is not None else None,
            'scores': scores.tolist(),
            'joint_names': COCOtoH36MConverter.H36M_JOINT_NAMES,
            'skeleton': COCOtoH36MConverter.H36M_SKELETON,
            'processing_time': total_time,
        }

        # Save JSON
        json_path = output_dir / f"{video_name}_poses.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved JSON: {json_path}")

        # Save pickle (for OpenSim pipeline)
        pkl_path = output_dir / f"{video_name}_poses.pkl"
        pkl_data = {
            **results,
            'keypoints_2d_coco': keypoints_2d,
            'keypoints_2d_h36m': keypoints_h36m,
            'keypoints_3d': keypoints_3d,
            'scores': scores,
        }
        with open(pkl_path, 'wb') as f:
            pickle.dump(pkl_data, f)
        print(f"Saved pickle: {pkl_path}")

        # Print summary
        print(f"\n{'='*50}")
        print("Processing Complete!")
        print(f"{'='*50}")
        print(f"Frames processed: {total_frames}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {total_frames / total_time:.2f}")
        print(f"{'='*50}\n")

        return results


def main():
    parser = argparse.ArgumentParser(description='Video to 3D Pose Estimation')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--device', '-d', default='cuda:0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--person', '-p', type=int, default=0, help='Person index')
    parser.add_argument('--check-only', action='store_true', help='Only check checkpoint')

    args = parser.parse_args()

    if args.check_only:
        check_checkpoint()
        return

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    pipeline = VideoPoseEstimationPipeline(device=args.device)
    pipeline.process_video(
        video_path=args.input,
        output_dir=args.output,
        person_idx=args.person,
    )


if __name__ == "__main__":
    main()
