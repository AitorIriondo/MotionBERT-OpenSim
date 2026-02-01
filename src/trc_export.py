"""
Stage 5: TRC Marker Export for OpenSim.

Converts processed keypoints to TRC format with anatomical markers
suitable for OpenSim Inverse Kinematics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import (
    MARKER_DEFINITIONS,
    RTMPOSE_LEFT_HIP,
    RTMPOSE_RIGHT_HIP,
    RTMPOSE_LEFT_SHOULDER,
    RTMPOSE_RIGHT_SHOULDER,
    RTMPOSE_LEFT_KNEE,
    RTMPOSE_RIGHT_KNEE,
)


def compute_marker(
    keypoints: np.ndarray,
    marker_def: Tuple,
) -> np.ndarray:
    """
    Compute marker position based on marker definition.

    Args:
        keypoints: Keypoints array (n_frames, n_keypoints, 3).
        marker_def: Tuple of (name, source_type, source_data).

    Returns:
        Marker positions array (n_frames, 3).
    """
    name, source_type, source_data = marker_def
    n_frames = keypoints.shape[0]

    if source_type == "landmark":
        # Direct keypoint mapping
        return keypoints[:, source_data, :]

    elif source_type == "midpoint":
        # Average of two keypoints
        kp1, kp2 = source_data
        return (keypoints[:, kp1, :] + keypoints[:, kp2, :]) / 2

    elif source_type == "offset":
        # Offset along a vector
        kp_from, kp_to, fraction = source_data
        direction = keypoints[:, kp_to, :] - keypoints[:, kp_from, :]
        return keypoints[:, kp_from, :] + fraction * direction

    elif source_type == "virtual":
        # Computed from pelvis geometry
        return compute_virtual_marker(keypoints, source_data)

    else:
        raise ValueError(f"Unknown marker source type: {source_type}")


def compute_virtual_marker(
    keypoints: np.ndarray,
    marker_type: str,
) -> np.ndarray:
    """
    Compute virtual markers from pelvis/trunk geometry.

    Args:
        keypoints: Keypoints array (n_frames, n_keypoints, 3) in OpenSim coords.
        marker_type: Type of virtual marker ('l_asis', 'r_asis', 'l_psis', 'r_psis').

    Returns:
        Virtual marker positions (n_frames, 3).
    """
    n_frames = keypoints.shape[0]

    # Get hip and shoulder positions
    l_hip = keypoints[:, RTMPOSE_LEFT_HIP, :]
    r_hip = keypoints[:, RTMPOSE_RIGHT_HIP, :]
    l_shoulder = keypoints[:, RTMPOSE_LEFT_SHOULDER, :]
    r_shoulder = keypoints[:, RTMPOSE_RIGHT_SHOULDER, :]

    hip_center = (l_hip + r_hip) / 2
    shoulder_center = (l_shoulder + r_shoulder) / 2

    # Build pelvis coordinate system (in OpenSim coords: X=forward, Y=up, Z=right)
    pelvis_z = r_hip - l_hip  # Lateral (left to right)
    pelvis_y = shoulder_center - hip_center  # Vertical (up)
    pelvis_x = np.cross(pelvis_y, pelvis_z)  # Forward (anterior)

    # Normalize axes
    pelvis_x = pelvis_x / (np.linalg.norm(pelvis_x, axis=1, keepdims=True) + 1e-8)
    pelvis_y = pelvis_y / (np.linalg.norm(pelvis_y, axis=1, keepdims=True) + 1e-8)
    pelvis_z = pelvis_z / (np.linalg.norm(pelvis_z, axis=1, keepdims=True) + 1e-8)

    # Pelvis width (for scaling offsets)
    pelvis_width = np.linalg.norm(r_hip - l_hip, axis=1, keepdims=True)

    # Compute virtual markers based on typical anatomical offsets
    # ASIS: Anterior Superior Iliac Spine (front of pelvis)
    # PSIS: Posterior Superior Iliac Spine (back of pelvis)

    if marker_type == "l_asis":
        # Left ASIS: forward and slightly left from left hip
        forward_offset = 0.10 * pelvis_width
        lateral_offset = -0.03 * pelvis_width  # Slightly more lateral
        return l_hip + forward_offset * pelvis_x + lateral_offset * pelvis_z

    elif marker_type == "r_asis":
        # Right ASIS: forward and slightly right from right hip
        forward_offset = 0.10 * pelvis_width
        lateral_offset = 0.03 * pelvis_width
        return r_hip + forward_offset * pelvis_x + lateral_offset * pelvis_z

    elif marker_type == "l_psis":
        # Left PSIS: backward from hip center, left side
        backward_offset = -0.10 * pelvis_width
        lateral_offset = -0.05 * pelvis_width
        return hip_center + backward_offset * pelvis_x + lateral_offset * pelvis_z

    elif marker_type == "r_psis":
        # Right PSIS: backward from hip center, right side
        backward_offset = -0.10 * pelvis_width
        lateral_offset = 0.05 * pelvis_width
        return hip_center + backward_offset * pelvis_x + lateral_offset * pelvis_z

    else:
        raise ValueError(f"Unknown virtual marker type: {marker_type}")


def compute_all_markers(
    keypoints: np.ndarray,
    marker_definitions: List[Tuple] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute all markers from keypoints.

    Args:
        keypoints: Keypoints array (n_frames, n_keypoints, 3).
        marker_definitions: List of marker definitions. Uses default if None.

    Returns:
        Dictionary mapping marker names to position arrays (n_frames, 3).
    """
    if marker_definitions is None:
        marker_definitions = MARKER_DEFINITIONS

    markers = {}
    for marker_def in marker_definitions:
        name = marker_def[0]
        markers[name] = compute_marker(keypoints, marker_def)

    return markers


def write_trc_file(
    output_path: str,
    markers: Dict[str, np.ndarray],
    fps: float,
    units: str = "mm",
) -> None:
    """
    Write markers to TRC file format.

    Args:
        output_path: Path to output TRC file.
        markers: Dictionary mapping marker names to positions (n_frames, 3).
        fps: Frame rate.
        units: Units for coordinates ('mm' or 'm').
    """
    marker_names = list(markers.keys())
    n_markers = len(marker_names)
    n_frames = len(next(iter(markers.values())))

    # Convert to mm if needed
    scale = 1000.0 if units == "mm" else 1.0

    with open(output_path, "w") as f:
        # Header line 1
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{Path(output_path).name}\n")

        # Header line 2
        f.write(f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps:.2f}\t{fps:.2f}\t{n_frames}\t{n_markers}\t{units}\t{fps:.2f}\t1\t{n_frames}\n")

        # Header line 3: marker names
        f.write("Frame#\tTime\t")
        for name in marker_names:
            f.write(f"{name}\t\t\t")
        f.write("\n")

        # Header line 4: coordinate labels
        f.write("\t\t")
        for i, name in enumerate(marker_names):
            f.write(f"X{i+1}\tY{i+1}\tZ{i+1}\t")
        f.write("\n")

        # Data lines
        for frame_idx in range(n_frames):
            time = frame_idx / fps
            f.write(f"{frame_idx + 1}\t{time:.6f}\t")

            for name in marker_names:
                pos = markers[name][frame_idx] * scale
                # Handle NaN values
                if np.any(np.isnan(pos)):
                    f.write("\t\t\t")
                else:
                    f.write(f"{pos[0]:.3f}\t{pos[1]:.3f}\t{pos[2]:.3f}\t")

            f.write("\n")


def keypoints_to_trc(
    input_path: str,
    output_dir: str,
    marker_definitions: List[Tuple] = None,
    units: str = "mm",
    show_progress: bool = True,
) -> str:
    """
    Convert processed keypoints to TRC marker file.

    Args:
        input_path: Path to processed keypoints JSON.
        output_dir: Directory to save TRC file.
        marker_definitions: Custom marker definitions. Uses default if None.
        units: Units for TRC file ('mm' or 'm').
        show_progress: Print progress updates.

    Returns:
        Path to output TRC file.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print(f"Loading processed keypoints from: {input_path}")

    # Load input data
    with open(input_path, "r") as f:
        data = json.load(f)

    keypoints = np.array(data["keypoints"])  # (n_frames, n_keypoints, 3)
    metadata = data["metadata"]
    fps = metadata["fps"]

    if show_progress:
        print(f"  Frames: {len(keypoints)}, FPS: {fps}")

    # Compute markers
    if show_progress:
        print("  Computing markers...")
    markers = compute_all_markers(keypoints, marker_definitions)

    if show_progress:
        print(f"  Generated {len(markers)} markers: {list(markers.keys())}")

    # Write TRC file
    output_name = input_path.stem.replace("_processed", "") + "_markers.trc"
    output_path = output_dir / output_name

    if show_progress:
        print(f"  Writing TRC file ({units})...")
    write_trc_file(str(output_path), markers, fps, units=units)

    if show_progress:
        print(f"Saved TRC file to: {output_path}")

    return str(output_path)


def validate_trc_file(trc_path: str) -> Dict:
    """
    Validate TRC file and return summary statistics.

    Args:
        trc_path: Path to TRC file.

    Returns:
        Dictionary with validation results.
    """
    with open(trc_path, "r") as f:
        lines = f.readlines()

    # Parse header
    header_line2 = lines[2].strip().split("\t")
    fps = float(header_line2[0])
    n_frames = int(header_line2[2])
    n_markers = int(header_line2[3])
    units = header_line2[4]

    # Parse marker names
    marker_line = lines[3].strip().split("\t")
    marker_names = [m for m in marker_line[2:] if m]  # Skip Frame# and Time

    # Count valid data points
    valid_points = 0
    nan_points = 0
    for line in lines[5:]:  # Skip header lines
        values = line.strip().split("\t")
        for i in range(2, len(values), 3):  # Skip Frame# and Time
            if i + 2 < len(values):
                try:
                    x, y, z = float(values[i]), float(values[i+1]), float(values[i+2])
                    if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                        valid_points += 1
                    else:
                        nan_points += 1
                except (ValueError, IndexError):
                    nan_points += 1

    return {
        "fps": fps,
        "n_frames": n_frames,
        "n_markers": n_markers,
        "units": units,
        "marker_names": marker_names,
        "valid_points": valid_points,
        "nan_points": nan_points,
        "valid_ratio": valid_points / (valid_points + nan_points) if (valid_points + nan_points) > 0 else 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert keypoints to TRC marker file")
    parser.add_argument("input", help="Path to processed keypoints JSON")
    parser.add_argument(
        "--output", "-o",
        default="output/markers",
        help="Output directory (default: output/markers)"
    )
    parser.add_argument(
        "--units", "-u",
        choices=["mm", "m"],
        default="mm",
        help="Units for TRC file (default: mm)"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate output TRC file"
    )

    args = parser.parse_args()

    output_path = keypoints_to_trc(
        args.input,
        args.output,
        units=args.units,
    )

    if args.validate:
        print("\nValidating TRC file...")
        stats = validate_trc_file(output_path)
        print(f"  FPS: {stats['fps']}")
        print(f"  Frames: {stats['n_frames']}")
        print(f"  Markers: {stats['n_markers']}")
        print(f"  Units: {stats['units']}")
        print(f"  Valid data ratio: {stats['valid_ratio']:.1%}")

    print(f"\nOutput saved to: {output_path}")
