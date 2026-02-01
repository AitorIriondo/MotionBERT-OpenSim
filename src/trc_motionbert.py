"""
TRC Export for MotionBERT H36M keypoints for use with Pose2Sim.

This module converts H36M 17-joint keypoints from MotionBERT to TRC format
compatible with Pose2Sim and OpenSim.

MotionBERT H36M joint order:
0: Hip (pelvis center)
1: RHip
2: RKnee
3: RFoot (right ankle)
4: LHip
5: LKnee
6: LFoot (left ankle)
7: Spine
8: Thorax
9: Neck/Nose
10: Head
11: LShoulder
12: LElbow
13: LWrist
14: RShoulder
15: RElbow
16: RWrist

Coordinate Systems:
- MotionBERT: X=right, Y=down, Z=forward (camera-centric)
- OpenSim: X=forward/anterior, Y=up/superior, Z=right/lateral

Transformation: X_osim = Z_mb, Y_osim = -Y_mb, Z_osim = X_mb
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, List, Optional


# H36M to COCO17 marker mapping
# COCO17 order: Nose, LEye, REye, LEar, REar, LShoulder, RShoulder,
#               LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, RKnee, LAnkle, RAnkle
# But Pose2Sim Markers_Coco17.xml only has 14 markers (no eyes/ears)

H36M_JOINT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
    'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow',
    'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]

# Mapping from Pose2Sim COCO17 marker names to H36M joint indices
# H36M joints: 8=Thorax (chest), 9=Neck/Nose (base of skull), 10=Head (top of head)
# COCO17 expects: Nose at face level, Neck at base of neck
COCO17_TO_H36M = {
    'Nose': 10,      # Head (top of head) - closest to face level
    'Neck': 9,       # Neck/Nose (base of skull) - actual neck position
    'LShoulder': 11,
    'RShoulder': 14,
    'LElbow': 12,
    'RElbow': 15,
    'LWrist': 13,
    'RWrist': 16,
    'LHip': 4,
    'RHip': 1,
    'LKnee': 5,
    'RKnee': 2,
    'LAnkle': 6,     # LFoot
    'RAnkle': 3,     # RFoot
}

# Order matching Markers_Coco17.xml
COCO17_MARKER_NAMES = [
    'Nose', 'Neck', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow',
    'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle'
]


def load_motionbert_keypoints(pkl_path: str) -> Tuple[np.ndarray, float, dict]:
    """
    Load MotionBERT H36M keypoints from pickle file.

    Args:
        pkl_path: Path to the poses pickle file

    Returns:
        keypoints_3d: Array of shape (n_frames, 17, 3)
        fps: Frame rate
        metadata: Additional metadata from pickle
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    keypoints_3d = data['keypoints_3d']
    fps = data.get('fps', 30.0)

    return keypoints_3d, fps, data


def fix_left_right_consistency(keypoints: np.ndarray) -> np.ndarray:
    """
    Fix left/right swapping issues in pose estimation.

    Monocular 3D pose estimation can sometimes swap left and right sides.
    This function detects and corrects these swaps to maintain consistency.

    For SIDE-VIEW video where subject walks in -X direction:
    - LShoulder should have HIGHER X than RShoulder (left is further from camera)
    - LHip should have HIGHER X than RHip

    H36M joint indices:
    0: Hip (center), 1: RHip, 2: RKnee, 3: RFoot, 4: LHip, 5: LKnee, 6: LFoot,
    7: Spine, 8: Thorax, 9: Neck/Nose, 10: Head, 11: LShoulder, 12: LElbow,
    13: LWrist, 14: RShoulder, 15: RElbow, 16: RWrist

    Args:
        keypoints: Array of shape (n_frames, 17, 3)

    Returns:
        Corrected keypoints with consistent left/right
    """
    n_frames = keypoints.shape[0]
    corrected = keypoints.copy()

    # Define left/right joint pairs (left_idx, right_idx)
    lr_pairs = [
        (4, 1),   # LHip, RHip
        (5, 2),   # LKnee, RKnee
        (6, 3),   # LFoot, RFoot
        (11, 14), # LShoulder, RShoulder
        (12, 15), # LElbow, RElbow
        (13, 16), # LWrist, RWrist
    ]

    print(f"  Left/right consistency check (side-view):")

    # For side-view where subject faces +Z (away from camera):
    # - Subject's LEFT side is at HIGHER X (further from camera in walking direction)
    # - Subject's RIGHT side is at LOWER X (closer to camera in walking direction)
    #
    # Use the cross product of shoulder vector with spine vector.
    # If the Z component is positive, the pose is in the expected orientation.
    lr_indicators = []

    for frame in range(n_frames):
        lshoulder = keypoints[frame, 11]  # LShoulder
        rshoulder = keypoints[frame, 14]  # RShoulder
        hip = keypoints[frame, 0]         # Hip center
        thorax = keypoints[frame, 8]      # Thorax

        # Shoulder vector: L to R
        shoulder_vec = rshoulder - lshoulder
        # Spine vector: hip to thorax
        spine_vec = thorax - hip

        # Cross product - Z component indicates facing direction
        cross = np.cross(shoulder_vec, spine_vec)
        lr_indicators.append(cross[2])  # Z component should be positive

    lr_indicators = np.array(lr_indicators)

    # Reference: Z component of cross product should be positive (facing +Z)
    # Use median of first 30 frames
    ref_sign = np.sign(np.median(lr_indicators[:min(30, n_frames)]))
    print(f"    Reference facing indicator sign: {ref_sign:.0f}")
    print(f"    Expected: +1 (facing +Z direction)")

    n_flipped = 0

    for frame in range(n_frames):
        curr_sign = np.sign(lr_indicators[frame])

        # If current sign differs from reference, flip L/R
        if curr_sign != 0 and curr_sign != ref_sign:
            for l_idx, r_idx in lr_pairs:
                temp = corrected[frame, l_idx].copy()
                corrected[frame, l_idx] = corrected[frame, r_idx]
                corrected[frame, r_idx] = temp
            n_flipped += 1

    print(f"    Flipped {n_flipped} frames to maintain consistency")

    return corrected


def normalize_bone_lengths(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize skeleton to have consistent bone lengths across all frames.

    MotionBERT monocular estimation produces variable bone lengths per frame.
    This function computes mean bone lengths and adjusts each frame to match,
    preserving joint angles while ensuring consistent segment proportions.

    H36M skeleton hierarchy (parent -> child):
        0 (Hip/Pelvis) is the root
        0 -> 1 (RHip), 0 -> 4 (LHip), 0 -> 7 (Spine)
        1 -> 2 (RKnee), 4 -> 5 (LKnee)
        2 -> 3 (RAnkle), 5 -> 6 (LAnkle)
        7 -> 8 (Thorax)
        8 -> 9 (Neck), 8 -> 11 (LShoulder), 8 -> 14 (RShoulder)
        9 -> 10 (Head)
        11 -> 12 (LElbow), 14 -> 15 (RElbow)
        12 -> 13 (LWrist), 15 -> 16 (RWrist)

    Args:
        keypoints: Array of shape (n_frames, 17, 3) in any coordinate system

    Returns:
        Normalized keypoints with consistent bone lengths
    """
    n_frames = keypoints.shape[0]

    # Define skeleton hierarchy: (parent_idx, child_idx)
    # Process order matters - parents must be processed before children
    bones = [
        # From pelvis
        (0, 1),   # Hip -> RHip
        (0, 4),   # Hip -> LHip
        (0, 7),   # Hip -> Spine
        # Legs
        (1, 2),   # RHip -> RKnee
        (4, 5),   # LHip -> LKnee
        (2, 3),   # RKnee -> RAnkle
        (5, 6),   # LKnee -> LAnkle
        # Torso
        (7, 8),   # Spine -> Thorax
        # Upper body
        (8, 9),   # Thorax -> Neck
        (8, 11),  # Thorax -> LShoulder
        (8, 14),  # Thorax -> RShoulder
        (9, 10),  # Neck -> Head
        # Arms
        (11, 12), # LShoulder -> LElbow
        (14, 15), # RShoulder -> RElbow
        (12, 13), # LElbow -> LWrist
        (15, 16), # RElbow -> RWrist
    ]

    # Compute mean bone lengths across all frames
    mean_lengths = {}
    for parent, child in bones:
        lengths = np.linalg.norm(keypoints[:, child, :] - keypoints[:, parent, :], axis=1)
        mean_lengths[(parent, child)] = np.mean(lengths)

    # Report variation before normalization
    print(f"  Normalizing bone lengths across {n_frames} frames...")
    total_cv = []
    for parent, child in bones:
        lengths = np.linalg.norm(keypoints[:, child, :] - keypoints[:, parent, :], axis=1)
        cv = np.std(lengths) / np.mean(lengths) * 100
        total_cv.append(cv)
    print(f"    Mean bone length CV before: {np.mean(total_cv):.1f}%")

    # Normalize each frame
    normalized = keypoints.copy()

    for frame in range(n_frames):
        # Start from root (pelvis at index 0) - keep it fixed
        # Then adjust each child joint to be at the correct distance from parent

        for parent, child in bones:
            # Current direction from parent to child
            direction = normalized[frame, child, :] - normalized[frame, parent, :]
            current_length = np.linalg.norm(direction)

            if current_length > 1e-6:  # Avoid division by zero
                # Normalize direction and scale to mean length
                direction = direction / current_length
                target_length = mean_lengths[(parent, child)]

                # Set child position at correct distance from parent
                normalized[frame, child, :] = normalized[frame, parent, :] + direction * target_length

    # Report variation after normalization
    total_cv_after = []
    for parent, child in bones:
        lengths = np.linalg.norm(normalized[:, child, :] - normalized[:, parent, :], axis=1)
        cv = np.std(lengths) / np.mean(lengths) * 100
        total_cv_after.append(cv)
    print(f"    Mean bone length CV after: {np.mean(total_cv_after):.1f}%")

    return normalized


def compute_body_frame_rotation(keypoints: np.ndarray, reference_frames: int = 10) -> np.ndarray:
    """
    Compute rotation matrix from body orientation in first N frames.

    This determines the subject's actual body orientation from hip and shoulder
    positions, rather than assuming a fixed camera view. This is the same approach
    used in KineticsToolkit for MediaPipe keypoints.

    H36M joint indices:
        LHip = 4, RHip = 1
        LShoulder = 11, RShoulder = 14

    Body axes (computed from anatomy):
        body_x = r_hip - l_hip  (lateral: left -> right)
        body_z = shoulder_center - hip_center  (vertical: down -> up)
        body_y = cross(body_z, body_x)  (forward: perpendicular to both)

    OpenSim coordinate convention:
        X = anterior (forward)
        Y = superior (up)
        Z = lateral (right)

    The rotation matrix R maps from input coordinates to OpenSim coordinates:
        R = [body_y,   # OpenSim X = our body forward
             body_z,   # OpenSim Y = our body up
             body_x]   # OpenSim Z = our body lateral

    Args:
        keypoints: Array of shape (n_frames, 17, 3) in MotionBERT coordinates
        reference_frames: Number of initial frames to average for stable orientation

    Returns:
        3x3 rotation matrix
    """
    n_ref = min(reference_frames, len(keypoints))

    # H36M joint indices
    L_HIP = 4
    R_HIP = 1
    L_SHOULDER = 11
    R_SHOULDER = 14

    # Average positions over reference frames for stability
    l_hip_avg = np.mean(keypoints[:n_ref, L_HIP, :], axis=0)
    r_hip_avg = np.mean(keypoints[:n_ref, R_HIP, :], axis=0)
    l_shoulder_avg = np.mean(keypoints[:n_ref, L_SHOULDER, :], axis=0)
    r_shoulder_avg = np.mean(keypoints[:n_ref, R_SHOULDER, :], axis=0)

    hip_center = (l_hip_avg + r_hip_avg) / 2
    shoulder_center = (l_shoulder_avg + r_shoulder_avg) / 2

    # Body X-axis (lateral): left hip to right hip
    body_x = r_hip_avg - l_hip_avg
    body_x = body_x / (np.linalg.norm(body_x) + 1e-8)

    # Body Z-axis (up): hip center to shoulder center
    body_z = shoulder_center - hip_center
    body_z = body_z / (np.linalg.norm(body_z) + 1e-8)

    # Body Y-axis (forward): perpendicular to X and Z (right-hand rule)
    # cross(right, up) = forward (NOT cross(up, right) which gives backward!)
    body_y = np.cross(body_x, body_z)
    body_y = body_y / (np.linalg.norm(body_y) + 1e-8)

    # Recompute Z to ensure orthogonality
    # cross(forward, right) = up (NOT cross(right, forward) which gives down!)
    body_z = np.cross(body_y, body_x)
    body_z = body_z / (np.linalg.norm(body_z) + 1e-8)

    # Rotation matrix: rows are OpenSim axes expressed in our coordinate system
    # OpenSim X (forward) = body_y direction in our coords
    # OpenSim Y (up) = body_z direction in our coords
    # OpenSim Z (right) = body_x direction in our coords
    R = np.array([
        body_y,  # OpenSim X = our body forward
        body_z,  # OpenSim Y = our body up
        body_x,  # OpenSim Z = our body lateral
    ])

    return R


def transform_motionbert_to_opensim(keypoints: np.ndarray, target_height: float = 1.75,
                                     use_simple_transform: bool = True) -> np.ndarray:
    """
    Transform MotionBERT coordinates to OpenSim coordinate system.

    MotionBERT coordinate system (camera-centric):
        X = horizontal (left has higher X when facing away from camera)
        Y = down (positive toward ground)
        Z = depth (positive away from camera)

    OpenSim coordinate convention:
        X = forward/anterior
        Y = up/superior
        Z = right/lateral (positive = subject's RIGHT side)

    Two modes:
        - Simple transform: Direct axis mapping assuming subject faces +Z (away from camera)
          X_osim = Z_mb, Y_osim = -Y_mb, Z_osim = -X_mb
        - Body-frame rotation: Compute rotation from pelvis/shoulder orientation

    Args:
        keypoints: Array of shape (n_frames, n_joints, 3) in MotionBERT coordinates
        target_height: Target height in meters for scaling
        use_simple_transform: If True, use simple axis mapping; if False, use body-frame rotation

    Returns:
        Transformed keypoints in OpenSim coordinates
    """
    n_frames = keypoints.shape[0]

    if use_simple_transform:
        # Simple axis transformation
        # MotionBERT: X=lateral (left=higher), Y=down, Z=depth (away=positive)
        # OpenSim: X=forward, Y=up, Z=right
        #
        # Base transform (subject facing away from camera, +Z):
        #   X_osim = Z_mb, Y_osim = -Y_mb, Z_osim = -X_mb
        #
        # But skeleton walks backward, so rotate 180° around Y:
        #   X' = -X, Z' = -Z (Y unchanged)
        #
        # Final transform:
        #   X_osim = -Z_mb, Y_osim = -Y_mb, Z_osim = -(-X_mb) = X_mb...
        #
        # Wait, that swaps L/R. Need to keep Z_osim = -X_mb and just flip X:
        # Actually for 180° Y rotation on the RESULT, not the input:
        #   X_osim = -Z_mb (flip forward direction)
        #   Z_osim = -X_mb (keep L/R correct, then flip for rotation) = X_mb? No...
        #
        # Let me think differently. The base transform gives correct L/R but wrong facing.
        # To flip facing 180° without swapping L/R, we negate X only:
        print(f"  Using simple axis transformation (flipped forward)")
        transformed = np.zeros_like(keypoints)
        transformed[..., 0] = -keypoints[..., 2]  # X_osim = -Z_mb (flip forward)
        transformed[..., 1] = -keypoints[..., 1]  # Y_osim = -Y_mb (up)
        transformed[..., 2] = -keypoints[..., 0]  # Z_osim = -X_mb (keep L/R correct)
    else:
        # Body-frame rotation approach
        R = compute_body_frame_rotation(keypoints, reference_frames=10)
        print(f"  Body-frame rotation matrix computed from first 10 frames")
        transformed = np.zeros_like(keypoints)
        for f in range(n_frames):
            transformed[f] = keypoints[f] @ R.T

    # Step 3: Center laterally (Z axis) at hip mean
    # H36M Hip center is joint 0
    hip_z = transformed[:, 0, 2].mean()
    transformed[:, :, 2] -= hip_z

    # Step 4: Put feet on ground (Y=0)
    # H36M: RFoot = 3, LFoot = 6
    foot_y_min = min(transformed[:, 3, 1].min(), transformed[:, 6, 1].min())
    transformed[:, :, 1] -= foot_y_min

    # Step 5: Scale to target height
    # H36M: Head = 10, RFoot = 3, LFoot = 6
    head_y = transformed[:, 10, 1].mean()
    foot_y = min(transformed[:, 3, 1].mean(), transformed[:, 6, 1].mean())
    current_height = head_y - foot_y

    print(f"  Current skeleton height: {current_height:.3f}m")

    if current_height > 0.1:  # Sanity check
        scale = target_height / current_height
        transformed *= scale
        print(f"  Scaling by {scale:.3f} to target height {target_height:.3f}m")

        # Re-ground feet after scaling
        foot_y_min = min(transformed[:, 3, 1].min(), transformed[:, 6, 1].min())
        transformed[:, :, 1] -= foot_y_min

        # Re-center Z after scaling
        hip_z = transformed[:, 0, 2].mean()
        transformed[:, :, 2] -= hip_z
    else:
        print(f"  Warning: Height too small ({current_height:.3f}m), not scaling")

    return transformed


def extract_coco17_markers(h36m_keypoints: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """
    Extract COCO17 markers from H36M keypoints.

    Note: H36M indices after transformation are still:
    0: Hip, 1: RHip, 2: RKnee, 3: RFoot, 4: LHip, 5: LKnee, 6: LFoot,
    7: Spine, 8: Thorax, 9: Neck/Nose, 10: Head, 11: LShoulder, 12: LElbow,
    13: LWrist, 14: RShoulder, 15: RElbow, 16: RWrist

    Special handling for 'Nose':
        H36M doesn't have a face marker. We create a synthetic nose position
        by projecting forward from the head in the facing direction.

    Args:
        h36m_keypoints: Array of shape (n_frames, 17, 3) in OpenSim coordinates

    Returns:
        marker_names: List of marker names
        marker_positions: Array of shape (n_frames, n_markers, 3)
    """
    n_frames = h36m_keypoints.shape[0]
    n_markers = len(COCO17_MARKER_NAMES)

    marker_positions = np.zeros((n_frames, n_markers, 3))

    # H36M indices
    HEAD = 10
    NECK = 9
    THORAX = 8
    L_SHOULDER = 11
    R_SHOULDER = 14

    # Distance to project nose forward from head (in meters)
    NOSE_FORWARD_OFFSET = 0.10  # 10cm forward

    for frame in range(n_frames):
        # Compute facing direction for this frame
        # Lateral direction: right shoulder to left shoulder
        l_shoulder = h36m_keypoints[frame, L_SHOULDER]
        r_shoulder = h36m_keypoints[frame, R_SHOULDER]
        lateral = l_shoulder - r_shoulder  # Points to subject's left
        lateral = lateral / (np.linalg.norm(lateral) + 1e-8)

        # Up direction: thorax to neck
        thorax = h36m_keypoints[frame, THORAX]
        neck = h36m_keypoints[frame, NECK]
        up = neck - thorax
        up = up / (np.linalg.norm(up) + 1e-8)

        # Forward direction: cross(lateral, up) - note: left × up = forward
        # In OpenSim: X=forward, so this should point in +X direction
        forward = np.cross(lateral, up)
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # Extract markers
        for i, name in enumerate(COCO17_MARKER_NAMES):
            if name == 'Nose':
                # Synthetic nose: head position + forward offset
                head_pos = h36m_keypoints[frame, HEAD]
                nose_pos = head_pos + forward * NOSE_FORWARD_OFFSET
                marker_positions[frame, i, :] = nose_pos
            else:
                h36m_idx = COCO17_TO_H36M[name]
                marker_positions[frame, i, :] = h36m_keypoints[frame, h36m_idx, :]

    return COCO17_MARKER_NAMES, marker_positions


def compute_height_from_markers(marker_positions: np.ndarray, marker_names: List[str]) -> float:
    """
    Compute approximate height from markers.

    Uses head to ankle distance with correction factor.

    Args:
        marker_positions: Array of shape (n_frames, n_markers, 3) in meters
        marker_names: List of marker names

    Returns:
        Estimated height in meters
    """
    # Find indices
    name_to_idx = {name: i for i, name in enumerate(marker_names)}

    nose_idx = name_to_idx.get('Nose')
    l_ankle_idx = name_to_idx.get('LAnkle')
    r_ankle_idx = name_to_idx.get('RAnkle')

    if nose_idx is None or l_ankle_idx is None or r_ankle_idx is None:
        print("Warning: Missing markers for height computation")
        return 1.75  # Default height

    # Average over all frames
    nose_y = marker_positions[:, nose_idx, 1].mean()
    l_ankle_y = marker_positions[:, l_ankle_idx, 1].mean()
    r_ankle_y = marker_positions[:, r_ankle_idx, 1].mean()

    ankle_y = (l_ankle_y + r_ankle_y) / 2

    # Height from nose to ankle, with factor for head above nose
    nose_to_ankle = nose_y - ankle_y

    # Add ~10% for top of head above nose
    estimated_height = nose_to_ankle * 1.1

    return estimated_height


def center_and_scale_markers(marker_positions: np.ndarray,
                             marker_names: List[str],
                             target_height: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Center markers and optionally scale to target height.

    Centers the pelvis region on Y=0 at the ground level.

    Args:
        marker_positions: Array of shape (n_frames, n_markers, 3) in meters
        marker_names: List of marker names
        target_height: Target height in meters (None to keep original scale)

    Returns:
        Centered and scaled marker positions
        Scale factor applied
    """
    name_to_idx = {name: i for i, name in enumerate(marker_names)}

    # Find ground level (minimum ankle Y)
    l_ankle_idx = name_to_idx.get('LAnkle')
    r_ankle_idx = name_to_idx.get('RAnkle')

    if l_ankle_idx is not None and r_ankle_idx is not None:
        l_ankle_y_min = marker_positions[:, l_ankle_idx, 1].min()
        r_ankle_y_min = marker_positions[:, r_ankle_idx, 1].min()
        ground_y = min(l_ankle_y_min, r_ankle_y_min)
    else:
        ground_y = marker_positions[..., 1].min()

    # Shift so ground is at Y=0
    marker_positions = marker_positions.copy()
    marker_positions[..., 1] -= ground_y

    # Scale if target height specified
    scale_factor = 1.0
    if target_height is not None:
        current_height = compute_height_from_markers(marker_positions, marker_names)
        if current_height > 0:
            scale_factor = target_height / current_height
            marker_positions *= scale_factor
            print(f"Scaled from {current_height:.3f}m to {target_height:.3f}m (factor: {scale_factor:.3f})")

    return marker_positions, scale_factor


def export_trc(marker_positions: np.ndarray,
               marker_names: List[str],
               output_path: str,
               fps: float = 30.0,
               data_name: str = "motionbert") -> str:
    """
    Export markers to TRC format for Pose2Sim.

    Args:
        marker_positions: Array of shape (n_frames, n_markers, 3) in METERS
        marker_names: List of marker names
        output_path: Output TRC file path
        fps: Frame rate
        data_name: Name for the data

    Returns:
        Path to output file
    """
    n_frames = marker_positions.shape[0]
    n_markers = len(marker_names)

    with open(output_path, 'w') as f:
        # Line 1: Header identifier
        f.write("PathFileType\t4\t(X/Y/Z)\t{}\n".format(output_path))

        # Line 2: TRC parameters
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps:.2f}\t{fps:.2f}\t{n_frames}\t{n_markers}\tm\t{fps:.2f}\t1\t{n_frames}\n")

        # Line 3: Column headers (Frame#, Time, then marker names with X/Y/Z)
        header_parts = ["Frame#", "Time"]
        for name in marker_names:
            header_parts.extend([name, "", ""])  # Name for X, empty for Y, Z
        f.write("\t".join(header_parts) + "\n")

        # Line 4: Coordinate labels (OpenSim expects X1, Y1, Z1, X2, Y2, Z2, etc.)
        coord_parts = ["", ""]  # Frame and Time have no sub-labels
        for i in range(n_markers):
            idx = i + 1  # 1-based indexing
            coord_parts.extend([f"X{idx}", f"Y{idx}", f"Z{idx}"])
        f.write("\t".join(coord_parts) + "\n")

        # Line 5: Empty line (for compatibility)
        f.write("\n")

        # Data lines
        for frame_idx in range(n_frames):
            time = frame_idx / fps
            parts = [str(frame_idx + 1), f"{time:.6f}"]

            for marker_idx in range(n_markers):
                x, y, z = marker_positions[frame_idx, marker_idx, :]
                parts.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])

            f.write("\t".join(parts) + "\n")

    print(f"Exported TRC to {output_path}")
    print(f"  Frames: {n_frames}, Markers: {n_markers}, FPS: {fps:.2f}")
    print(f"  Units: meters")

    return output_path


def convert_motionbert_to_trc(pkl_path: str,
                               output_path: str,
                               target_height: float = 1.75,
                               fix_lr_swap: bool = True,
                               normalize_bones: bool = True) -> str:
    """
    Full pipeline: Load MotionBERT pickle, transform, and export to TRC.

    Args:
        pkl_path: Path to MotionBERT poses pickle
        output_path: Output TRC path
        target_height: Target height in meters (default 1.75m)
        fix_lr_swap: Whether to fix left/right swapping issues (default True)
        normalize_bones: Whether to normalize bone lengths across frames (default True)

    Returns:
        Path to output TRC file
    """
    print(f"Loading MotionBERT keypoints from {pkl_path}")
    keypoints, fps, metadata = load_motionbert_keypoints(pkl_path)
    print(f"  Shape: {keypoints.shape}, FPS: {fps:.2f}")

    # Fix left/right swapping issues
    if fix_lr_swap:
        keypoints = fix_left_right_consistency(keypoints)

    # Normalize bone lengths to ensure consistent skeleton proportions
    if normalize_bones:
        keypoints = normalize_bone_lengths(keypoints)

    # Transform to OpenSim coordinates (includes centering, grounding, and scaling)
    print("Transforming to OpenSim coordinate system...")
    keypoints_osim = transform_motionbert_to_opensim(keypoints, target_height, use_simple_transform=True)

    # Check coordinate ranges after transformation
    print(f"  After transform:")
    print(f"    X (forward) range: [{keypoints_osim[..., 0].min():.3f}, {keypoints_osim[..., 0].max():.3f}]")
    print(f"    Y (up) range: [{keypoints_osim[..., 1].min():.3f}, {keypoints_osim[..., 1].max():.3f}]")
    print(f"    Z (right) range: [{keypoints_osim[..., 2].min():.3f}, {keypoints_osim[..., 2].max():.3f}]")

    # Extract COCO17 markers
    print("Extracting COCO17 markers...")
    marker_names, marker_positions = extract_coco17_markers(keypoints_osim)
    print(f"  {len(marker_names)} markers: {marker_names}")

    # Verify height
    final_height = compute_height_from_markers(marker_positions, marker_names)
    print(f"  Final height: {final_height:.3f}m")

    # Export TRC
    print(f"Exporting TRC to {output_path}")
    return export_trc(marker_positions, marker_names, output_path, fps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert MotionBERT H36M to TRC")
    parser.add_argument("input", help="Input pickle file from MotionBERT")
    parser.add_argument("output", help="Output TRC file path")
    parser.add_argument("--height", type=float, default=None,
                       help="Target height in meters (default: estimate from data)")

    args = parser.parse_args()

    convert_motionbert_to_trc(args.input, args.output, args.height)
