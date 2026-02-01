"""
Stage 2: Post-Processing of RTMPose3D keypoints.

Performs:
1. Gap filling (interpolation for missing/low-confidence keypoints)
2. Coordinate system transformation (RTMPose3D → OpenSim)
3. Optional smoothing filter
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .constants import (
    TOTAL_KEYPOINTS,
    RTMPOSE_LEFT_SHOULDER,
    RTMPOSE_RIGHT_SHOULDER,
    RTMPOSE_LEFT_HIP,
    RTMPOSE_RIGHT_HIP,
    COORDINATE_SYSTEMS,
)


def fill_gaps(
    keypoints: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill gaps in keypoint trajectories using linear interpolation.

    Args:
        keypoints: Keypoints array of shape (n_frames, n_keypoints, 3).
        scores: Confidence scores of shape (n_frames, n_keypoints).
        threshold: Minimum confidence threshold. Values below are interpolated.

    Returns:
        Tuple of (filled_keypoints, gap_mask) where gap_mask indicates filled frames.
    """
    n_frames, n_keypoints, n_coords = keypoints.shape
    filled = keypoints.copy()
    gap_mask = np.zeros((n_frames, n_keypoints), dtype=bool)

    for kp_idx in range(n_keypoints):
        # Identify low-confidence frames
        low_conf = scores[:, kp_idx] < threshold

        # Also treat NaN values as gaps
        nan_mask = np.isnan(keypoints[:, kp_idx, 0])
        gaps = low_conf | nan_mask

        if not np.any(gaps):
            continue

        # Get valid (non-gap) indices
        valid_mask = ~gaps
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 2:
            # Not enough valid points to interpolate
            # Fill with nearest valid value or leave as NaN
            if len(valid_indices) == 1:
                filled[gaps, kp_idx, :] = keypoints[valid_indices[0], kp_idx, :]
            continue

        # Interpolate each coordinate
        for coord in range(n_coords):
            valid_values = keypoints[valid_mask, kp_idx, coord]

            # Linear interpolation
            filled[:, kp_idx, coord] = np.interp(
                np.arange(n_frames),
                valid_indices,
                valid_values,
            )

        gap_mask[gaps, kp_idx] = True

    return filled, gap_mask


def apply_butterworth_filter(
    keypoints: np.ndarray,
    fps: float,
    cutoff_freq: float = 6.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply low-pass Butterworth filter to smooth keypoint trajectories.

    Args:
        keypoints: Keypoints array of shape (n_frames, n_keypoints, 3).
        fps: Video frame rate.
        cutoff_freq: Cutoff frequency in Hz.
        order: Filter order.

    Returns:
        Filtered keypoints array.
    """
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        print("Warning: scipy not installed. Skipping smoothing filter.")
        return keypoints

    # Design filter
    nyquist = fps / 2
    normalized_cutoff = cutoff_freq / nyquist

    if normalized_cutoff >= 1:
        print(f"Warning: Cutoff frequency {cutoff_freq}Hz too high for {fps}fps video. Skipping filter.")
        return keypoints

    b, a = butter(order, normalized_cutoff, btype="low")

    # Apply filter to each keypoint coordinate
    filtered = keypoints.copy()
    n_frames, n_keypoints, n_coords = keypoints.shape

    for kp_idx in range(n_keypoints):
        # Check for NaN values
        if np.any(np.isnan(keypoints[:, kp_idx, :])):
            continue

        for coord in range(n_coords):
            try:
                filtered[:, kp_idx, coord] = filtfilt(b, a, keypoints[:, kp_idx, coord])
            except ValueError:
                # Filter failed (e.g., too few samples)
                pass

    return filtered


def rtmpose3d_to_opensim(keypoints: np.ndarray) -> np.ndarray:
    """
    Transform RTMPose3D coordinates to OpenSim convention.

    After conversion in pose_extraction.py, our coordinate system is:
        X = right (positive toward subject's right)
        Y = depth (positive away from camera, i.e., into the scene)
        Z = up (positive upward)

    OpenSim coordinate system:
        X = anterior (positive forward, toward camera)
        Y = superior (positive upward)
        Z = lateral (positive toward subject's right)

    Since the subject typically faces the camera:
        X_opensim = -Y_rtm (forward = toward camera = negative depth)
        Y_opensim = Z_rtm (up stays up)
        Z_opensim = X_rtm (right stays right)

    Args:
        keypoints: Keypoints in RTMPose3D coordinates (n_frames, n_keypoints, 3).

    Returns:
        Keypoints in OpenSim coordinates.
    """
    transformed = np.zeros_like(keypoints)
    transformed[..., 0] = -keypoints[..., 1]  # X_opensim = -Y_rtm (forward = toward camera)
    transformed[..., 1] = keypoints[..., 2]   # Y_opensim = Z_rtm (up)
    transformed[..., 2] = keypoints[..., 0]   # Z_opensim = X_rtm (right)
    return transformed


def compute_body_frame_rotation(
    keypoints: np.ndarray,
    reference_frames: int = 10,
) -> np.ndarray:
    """
    Compute rotation matrix based on subject's actual body orientation.

    Uses pelvis orientation from first N frames to establish subject's
    forward direction, rather than assuming global axes.

    Args:
        keypoints: Keypoints array in RTMPose3D coordinates.
        reference_frames: Number of frames to average for orientation.

    Returns:
        3x3 rotation matrix to align subject with OpenSim axes.
    """
    n_frames = min(reference_frames, len(keypoints))

    # Average first N frames for stable orientation
    l_hip = np.nanmean(keypoints[:n_frames, RTMPOSE_LEFT_HIP, :], axis=0)
    r_hip = np.nanmean(keypoints[:n_frames, RTMPOSE_RIGHT_HIP, :], axis=0)
    l_shoulder = np.nanmean(keypoints[:n_frames, RTMPOSE_LEFT_SHOULDER, :], axis=0)
    r_shoulder = np.nanmean(keypoints[:n_frames, RTMPOSE_RIGHT_SHOULDER, :], axis=0)

    # Compute body coordinate system
    hip_center = (l_hip + r_hip) / 2
    shoulder_center = (l_shoulder + r_shoulder) / 2

    # Body axes in RTMPose3D coordinates
    body_x = r_hip - l_hip  # Lateral (left to right)
    body_z = shoulder_center - hip_center  # Vertical (up)
    body_y = np.cross(body_z, body_x)  # Forward (perpendicular, right-hand rule)

    # Normalize
    body_x = body_x / (np.linalg.norm(body_x) + 1e-8)
    body_y = body_y / (np.linalg.norm(body_y) + 1e-8)
    body_z = body_z / (np.linalg.norm(body_z) + 1e-8)

    # Construct rotation matrix
    # Maps body coordinates to OpenSim: X=forward, Y=up, Z=right
    R = np.array([
        body_y,  # OpenSim X (forward) = body forward
        body_z,  # OpenSim Y (up) = body up
        body_x,  # OpenSim Z (right) = body right
    ])

    return R


def apply_body_frame_rotation(
    keypoints: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """
    Apply rotation matrix to keypoints.

    Args:
        keypoints: Keypoints array (n_frames, n_keypoints, 3).
        rotation_matrix: 3x3 rotation matrix.

    Returns:
        Rotated keypoints array.
    """
    # Center around pelvis before rotation
    n_frames = len(keypoints)
    pelvis_center = (
        keypoints[:, RTMPOSE_LEFT_HIP, :] + keypoints[:, RTMPOSE_RIGHT_HIP, :]
    ) / 2

    # Translate to origin
    centered = keypoints - pelvis_center[:, np.newaxis, :]

    # Apply rotation: P_rotated = P @ R.T
    rotated = np.einsum("fkc,rc->fkr", centered, rotation_matrix)

    # Translate back (pelvis at original position, but rotated)
    # Actually, we want pelvis centered at origin for OpenSim
    return rotated


def scale_to_height(
    keypoints: np.ndarray,
    target_height: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Scale keypoints to target person height.

    Args:
        keypoints: Keypoints array (n_frames, n_keypoints, 3) in OpenSim coords.
                   OpenSim: X=forward, Y=up, Z=right
        target_height: Target person height in meters. If None, returns unscaled.

    Returns:
        Tuple of (scaled_keypoints, scale_factor).
    """
    if target_height is None:
        return keypoints, 1.0

    from .constants import (
        RTMPOSE_NOSE,
        RTMPOSE_LEFT_HEEL, RTMPOSE_RIGHT_HEEL,
        RTMPOSE_LEFT_BIG_TOE, RTMPOSE_RIGHT_BIG_TOE,
    )

    # Compute current height from head (nose) to lowest foot point
    # Average over all frames for stability
    nose_y = np.nanmean(keypoints[:, RTMPOSE_NOSE, 1])

    # Find minimum foot height (heels and toes)
    foot_indices = [RTMPOSE_LEFT_HEEL, RTMPOSE_RIGHT_HEEL,
                    RTMPOSE_LEFT_BIG_TOE, RTMPOSE_RIGHT_BIG_TOE]
    foot_heights = [np.nanmean(keypoints[:, idx, 1]) for idx in foot_indices]
    min_foot_y = np.min(foot_heights)

    # Current height is nose to lowest foot point
    current_height = nose_y - min_foot_y

    if current_height < 0.1:
        print("Warning: Could not estimate height. Skipping scaling.")
        return keypoints, 1.0

    scale_factor = target_height / current_height
    scaled = keypoints * scale_factor

    print(f"    Computed height: {current_height:.3f}m -> Target: {target_height:.3f}m (scale: {scale_factor:.4f})")

    return scaled, scale_factor


def offset_to_ground(keypoints: np.ndarray) -> np.ndarray:
    """
    Offset keypoints so lowest point is at ground level (Y=0 in OpenSim).

    Args:
        keypoints: Keypoints in OpenSim coordinates (n_frames, n_keypoints, 3).

    Returns:
        Offset keypoints with ground at Y=0.
    """
    # Find minimum Y value (vertical) across all frames and keypoints
    min_y = np.nanmin(keypoints[:, :, 1])

    # Offset so minimum is at 0
    offset = keypoints.copy()
    offset[:, :, 1] -= min_y

    return offset


def process_keypoints(
    input_path: str,
    output_dir: str,
    fill_gaps_threshold: float = 0.3,
    smooth_cutoff: Optional[float] = None,
    target_height: Optional[float] = None,
    use_body_frame: bool = True,
    show_progress: bool = True,
) -> str:
    """
    Process RTMPose3D keypoints: gap filling, transforms, smoothing.

    Args:
        input_path: Path to input keypoints JSON.
        output_dir: Directory to save processed output.
        fill_gaps_threshold: Confidence threshold for gap filling (0-1).
        smooth_cutoff: Butterworth filter cutoff frequency (Hz). None to skip.
        target_height: Target height for scaling (meters). None to skip.
        use_body_frame: Use body-relative rotation instead of global axes.
        show_progress: Print progress updates.

    Returns:
        Path to output JSON file.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print(f"Loading keypoints from: {input_path}")

    # Load input data
    with open(input_path, "r") as f:
        data = json.load(f)

    keypoints = np.array(data["keypoints_3d"])  # (n_frames, 133, 3)
    scores = np.array(data["scores"])            # (n_frames, 133)
    metadata = data["metadata"]
    fps = metadata["fps"]

    n_frames, n_keypoints, _ = keypoints.shape
    if show_progress:
        print(f"  Frames: {n_frames}, Keypoints: {n_keypoints}")

    # Step 1: Fill gaps
    if show_progress:
        print(f"  Filling gaps (threshold={fill_gaps_threshold})...")
    keypoints, gap_mask = fill_gaps(keypoints, scores, threshold=fill_gaps_threshold)
    gaps_filled = np.sum(gap_mask)
    if show_progress:
        print(f"    Filled {gaps_filled} keypoint-frames")

    # Step 2: Transform to OpenSim coordinates
    # We have two approaches:
    # A) Body-frame rotation: Compute rotation matrix from pelvis/shoulder orientation
    #    and apply it directly. This maps body axes to OpenSim axes.
    # B) Simple axis swap: Assume subject faces camera, apply fixed mapping.
    #
    # IMPORTANT: These are mutually exclusive - do NOT apply both!

    if use_body_frame:
        if show_progress:
            print("  Computing body-frame rotation...")
        rotation_matrix = compute_body_frame_rotation(keypoints)
        # Apply rotation: this directly maps to OpenSim coordinates
        # The rotation matrix rows are OpenSim axes in our coordinate system
        keypoints = apply_body_frame_rotation(keypoints, rotation_matrix)
        if show_progress:
            print("  Applied body-frame rotation to OpenSim coordinates")
    else:
        rotation_matrix = None
        # Fallback: simple axis swap (assumes subject faces camera)
        if show_progress:
            print("  Transforming to OpenSim coordinates (simple axis swap)...")
        keypoints = rtmpose3d_to_opensim(keypoints)

    # Step 4: Scale to target height (optional)
    scale_factor = 1.0
    if target_height is not None:
        if show_progress:
            print(f"  Scaling to height {target_height}m...")
        keypoints, scale_factor = scale_to_height(keypoints, target_height)

    # Step 5: Offset to ground
    if show_progress:
        print("  Offsetting to ground level...")
    keypoints = offset_to_ground(keypoints)

    # Step 6: Smoothing filter (optional)
    if smooth_cutoff is not None:
        if show_progress:
            print(f"  Applying Butterworth filter (cutoff={smooth_cutoff}Hz)...")
        keypoints = apply_butterworth_filter(keypoints, fps, cutoff_freq=smooth_cutoff)

    # Prepare output data
    output_data = {
        "metadata": {
            **metadata,
            "processing": {
                "gap_fill_threshold": fill_gaps_threshold,
                "gaps_filled": int(gaps_filled),
                "smooth_cutoff_hz": smooth_cutoff,
                "target_height_m": target_height,
                "scale_factor": scale_factor,
                "use_body_frame": use_body_frame,
                "coordinate_system": COORDINATE_SYSTEMS["opensim"],
            },
        },
        "keypoint_names": data.get("keypoint_names", {}),
        "keypoints": keypoints.tolist(),
        "scores": scores.tolist(),
    }

    # Add rotation matrix if computed
    if rotation_matrix is not None:
        output_data["metadata"]["processing"]["rotation_matrix"] = rotation_matrix.tolist()

    # Save output
    output_name = input_path.stem.replace("_rtmpose3d", "") + "_processed.json"
    output_path = output_dir / output_name

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    if show_progress:
        print(f"Saved processed keypoints to: {output_path}")

    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-process RTMPose3D keypoints")
    parser.add_argument("input", help="Path to input keypoints JSON")
    parser.add_argument(
        "--output", "-o",
        default="output/processed",
        help="Output directory (default: output/processed)"
    )
    parser.add_argument(
        "--gap-threshold", "-g",
        type=float,
        default=0.3,
        help="Gap fill confidence threshold (default: 0.3)"
    )
    parser.add_argument(
        "--smooth", "-s",
        type=float,
        default=None,
        help="Butterworth filter cutoff frequency in Hz (default: None)"
    )
    parser.add_argument(
        "--height", "-H",
        type=float,
        default=None,
        help="Target height in meters for scaling (default: None)"
    )
    parser.add_argument(
        "--no-body-frame",
        action="store_true",
        help="Disable body-frame rotation (use global axes)"
    )

    args = parser.parse_args()

    output_path = process_keypoints(
        args.input,
        args.output,
        fill_gaps_threshold=args.gap_threshold,
        smooth_cutoff=args.smooth,
        target_height=args.height,
        use_body_frame=not args.no_body_frame,
    )
    print(f"\nOutput saved to: {output_path}")
