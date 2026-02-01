"""
Stage 1: Pose Extraction using rtmlib Wholebody3d.

Extracts 3D pose keypoints from video using rtmlib's RTMW3D model,
which provides native 3D whole-body pose estimation via ONNX inference.
Outputs 133 keypoints per frame in COCO-WholeBody format with 3D coordinates.

Note: RTMW3D outputs coordinates in model space, not real-world meters.
This module converts them to approximate real-world coordinates using
body proportions for scaling.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .constants import (
    TOTAL_KEYPOINTS,
    KEYPOINT_NAMES,
    COORDINATE_SYSTEMS,
    RTMPOSE_LEFT_HIP,
    RTMPOSE_RIGHT_HIP,
    RTMPOSE_NOSE,
    RTMPOSE_LEFT_ANKLE,
    RTMPOSE_RIGHT_ANKLE,
)

# Typical human body proportions for scaling
TYPICAL_PELVIS_WIDTH_M = 0.28  # Average pelvis width in meters
TYPICAL_HEIGHT_M = 1.7  # Average human height in meters


def extract_poses(
    video_path: str,
    output_dir: str,
    device: str = "cuda",
    person_idx: int = 0,
    mode: str = "balanced",
    show_progress: bool = True,
) -> str:
    """
    Extract 3D poses from video using rtmlib Wholebody3d (RTMW3D).

    Args:
        video_path: Path to input video file.
        output_dir: Directory to save output JSON.
        device: Device to run inference on ('cuda' or 'cpu').
        person_idx: Index of person to track (0 for first detected person).
        mode: Model mode - 'balanced' for good speed/accuracy tradeoff.
        show_progress: Whether to print progress updates.

    Returns:
        Path to output JSON file.
    """
    # Import rtmlib
    try:
        from rtmlib import Wholebody3d
    except ImportError:
        raise ImportError(
            "rtmlib with Wholebody3d not installed. Install with:\n"
            "pip install git+https://github.com/Tau-J/rtmlib.git"
        )

    # Validate video path
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine backend and device
    backend = "onnxruntime"
    # Normalize device format for rtmlib (expects 'cuda' or 'cpu', not 'cuda:0')
    rtmlib_device = "cuda" if "cuda" in device.lower() else "cpu"

    if show_progress:
        print(f"Loading RTMW3D model...")
        print(f"  Backend: {backend}")
        print(f"  Device: {rtmlib_device}")
        print(f"  Mode: {mode}")

    # Initialize model
    model = Wholebody3d(
        backend=backend,
        device=rtmlib_device,
        mode=mode,
    )

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if show_progress:
        print(f"Video: {video_path.name}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Frames: {total_frames}")
        print(f"Extracting 3D poses...")

    # Process frames
    all_keypoints_3d = []
    all_keypoints_2d = []
    all_scores = []
    all_bboxes = []
    frames_with_detection = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        # Wholebody3d returns: keypoints_3d, scores, keypoints_simcc, keypoints_2d
        result = model(frame)

        # Unpack results - format: (keypoints_3d, scores, keypoints_simcc, keypoints_2d)
        if len(result) >= 4:
            kps_3d_batch, scores_batch, _, kps_2d_batch = result[:4]
        else:
            # Fallback if format is different
            kps_3d_batch = result[0] if len(result) > 0 else []
            scores_batch = result[1] if len(result) > 1 else []
            kps_2d_batch = result[3] if len(result) > 3 else []

        # Check if person detected
        if len(kps_3d_batch) > person_idx:
            kps_3d = kps_3d_batch[person_idx]  # (133, 3)
            kps_2d = kps_2d_batch[person_idx] if len(kps_2d_batch) > person_idx else np.zeros((133, 2))
            scores = scores_batch[person_idx] if len(scores_batch) > person_idx else np.ones(133)

            # Estimate bounding box from 2D keypoints
            valid_mask = scores > 0.3
            if np.any(valid_mask):
                valid_kps = kps_2d[valid_mask]
                x_min, y_min = valid_kps.min(axis=0)
                x_max, y_max = valid_kps.max(axis=0)
                bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]
            else:
                bbox = None

            frames_with_detection += 1
        else:
            # No detection - fill with NaN
            kps_3d = np.full((TOTAL_KEYPOINTS, 3), np.nan)
            kps_2d = np.full((TOTAL_KEYPOINTS, 2), np.nan)
            scores = np.zeros(TOTAL_KEYPOINTS)
            bbox = None

        all_keypoints_3d.append(kps_3d)
        all_keypoints_2d.append(kps_2d)
        all_scores.append(scores)
        all_bboxes.append(bbox)

        frame_idx += 1
        if show_progress and frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()

    if show_progress:
        detection_rate = frames_with_detection / total_frames * 100 if total_frames > 0 else 0
        print(f"  Detection rate: {detection_rate:.1f}% ({frames_with_detection}/{total_frames})")

    # Convert to numpy arrays
    keypoints_3d_raw = np.array(all_keypoints_3d)  # (n_frames, 133, 3) - raw model coordinates
    keypoints_2d = np.array(all_keypoints_2d)      # (n_frames, 133, 2) - pixel coordinates
    scores = np.array(all_scores)                   # (n_frames, 133)

    if show_progress:
        print("  Converting to real-world coordinates...")

    # Convert RTMW3D output to real-world 3D coordinates
    # RTMW3D outputs:
    #   - keypoints_3d: X,Y in model input coordinates (288x384), Z is relative depth [-1, 1]
    #   - keypoints_2d: X,Y in original image pixel coordinates
    # We need to convert to meters using body proportions
    keypoints_3d = convert_to_world_coordinates(
        keypoints_3d_raw, keypoints_2d, scores, width, height
    )

    # Prepare output data
    output_data = {
        "metadata": {
            "source_video": str(video_path.absolute()),
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "model": "RTMW3D (rtmlib Wholebody3d)",
            "device": device,
            "mode": mode,
            "coordinate_system": COORDINATE_SYSTEMS["rtmpose3d"],
            "keypoint_count": TOTAL_KEYPOINTS,
            "frames_with_detection": frames_with_detection,
            "person_idx": person_idx,
        },
        "keypoint_names": {str(k): v for k, v in KEYPOINT_NAMES.items()},
        "keypoints_3d": keypoints_3d.tolist(),
        "keypoints_2d": keypoints_2d.tolist(),
        "scores": scores.tolist(),
        "bboxes": all_bboxes,
    }

    # Save output
    output_name = video_path.stem + "_rtmpose3d.json"
    output_path = output_dir / output_name

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    if show_progress:
        print(f"Saved keypoints to: {output_path}")

    return str(output_path)


def convert_to_world_coordinates(
    keypoints_3d_raw: np.ndarray,
    keypoints_2d: np.ndarray,
    scores: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Convert RTMW3D output to approximate real-world coordinates.

    RTMW3D outputs:
    - keypoints_3d: X,Y in model input space (288x384), Z is relative depth
    - keypoints_2d: X,Y in original image pixel coordinates

    This function converts to a consistent coordinate system in meters:
    - X: right (positive toward subject's right)
    - Y: depth (positive away from camera)
    - Z: up (positive upward)

    The scaling is based on typical human body proportions.

    Args:
        keypoints_3d_raw: Raw RTMW3D 3D keypoints (n_frames, 133, 3)
        keypoints_2d: 2D pixel keypoints (n_frames, 133, 2)
        scores: Confidence scores (n_frames, 133)
        image_width: Original image width
        image_height: Original image height

    Returns:
        Converted 3D keypoints in meters (n_frames, 133, 3)
    """
    n_frames = keypoints_3d_raw.shape[0]
    keypoints_3d = np.zeros_like(keypoints_3d_raw)

    # Calculate scale factor from pelvis width
    # Use median pelvis width across valid frames to get stable scale
    pelvis_widths_2d = []
    for frame_idx in range(n_frames):
        if scores[frame_idx, RTMPOSE_LEFT_HIP] > 0.3 and scores[frame_idx, RTMPOSE_RIGHT_HIP] > 0.3:
            l_hip_2d = keypoints_2d[frame_idx, RTMPOSE_LEFT_HIP]
            r_hip_2d = keypoints_2d[frame_idx, RTMPOSE_RIGHT_HIP]
            width_px = np.linalg.norm(l_hip_2d - r_hip_2d)
            if width_px > 10:  # Sanity check
                pelvis_widths_2d.append(width_px)

    if len(pelvis_widths_2d) > 0:
        median_pelvis_px = np.median(pelvis_widths_2d)
        # Scale factor: meters per pixel
        scale = TYPICAL_PELVIS_WIDTH_M / median_pelvis_px
    else:
        # Fallback: assume person takes up ~30% of image height = 1.7m
        scale = TYPICAL_HEIGHT_M / (0.3 * image_height)

    # Convert each frame
    for frame_idx in range(n_frames):
        # Get 2D pixel coordinates
        px_x = keypoints_2d[frame_idx, :, 0]  # pixels from left
        px_y = keypoints_2d[frame_idx, :, 1]  # pixels from top

        # Get relative depth from raw 3D output
        # Z range is approximately [-1, 1] where more negative = closer to camera
        rel_depth = keypoints_3d_raw[frame_idx, :, 2]

        # Convert to world coordinates:
        # X: horizontal position (subject's right is positive)
        #    When viewing subject from front, their right is on OUR left (smaller pixel X)
        #    So we negate: smaller pixel X -> positive world X
        world_x = -(px_x - image_width / 2) * scale

        # Y: depth (positive = away from camera)
        #    Scale relative depth to approximate meters
        #    Assume depth range of ~2m (person moving in depth)
        depth_scale = 1.0  # meters per unit of relative depth
        world_y = -rel_depth * depth_scale  # Negate so positive = away

        # Z: vertical position (up is positive)
        #    Image Y is from top, so negate and center
        world_z = -(px_y - image_height / 2) * scale

        keypoints_3d[frame_idx, :, 0] = world_x
        keypoints_3d[frame_idx, :, 1] = world_y
        keypoints_3d[frame_idx, :, 2] = world_z

    return keypoints_3d


def load_keypoints(json_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load keypoints from JSON file.

    Args:
        json_path: Path to keypoints JSON file.

    Returns:
        Tuple of (keypoints_3d, scores, metadata)
        - keypoints_3d: numpy array of shape (n_frames, 133, 3)
        - scores: numpy array of shape (n_frames, 133)
        - metadata: dict with video and extraction info
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    keypoints_3d = np.array(data["keypoints_3d"])
    scores = np.array(data["scores"])
    metadata = data["metadata"]

    return keypoints_3d, scores, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract 3D poses from video using RTMW3D")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "--output", "-o",
        default="output/keypoints",
        help="Output directory (default: output/keypoints)"
    )
    parser.add_argument(
        "--device", "-d",
        default="cuda",
        help="Device for inference: 'cuda' or 'cpu' (default: cuda)"
    )
    parser.add_argument(
        "--person", "-p",
        type=int,
        default=0,
        help="Person index to track (default: 0)"
    )
    parser.add_argument(
        "--mode", "-m",
        default="balanced",
        help="Model mode (default: balanced)"
    )

    args = parser.parse_args()

    output_path = extract_poses(
        args.video,
        args.output,
        device=args.device,
        person_idx=args.person,
        mode=args.mode,
    )
    print(f"\nOutput saved to: {output_path}")
