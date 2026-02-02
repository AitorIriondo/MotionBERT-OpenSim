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


def enforce_anthropometric_proportions(keypoints: np.ndarray) -> np.ndarray:
    """
    Enforce anatomically correct body proportions on the skeleton.

    Monocular 3D pose estimation often produces skeletons with incorrect segment
    proportions, particularly overestimating leg length relative to torso in the
    vertical direction. This function rescales leg segments to match standard
    human anthropometric ratios.

    The key insight is that in MotionBERT output:
    - Y axis is vertical (positive = down)
    - Leg bones are nearly 100% vertical (walking gait)
    - Torso bones have significant depth/lateral components

    We use VERTICAL distances (Y component) for proportion calculation, not
    Euclidean distances, to avoid distortion from forward lean.

    Standard human proportions (Winter, 2009; Drillis & Contini, 1966):
        - Torso (hip to shoulder): ~30% of height
        - Thigh (hip to knee): ~24.5% of height
        - Shin (knee to ankle): ~24.6% of height
        - Head above shoulders: ~12.5% of height (from shoulder to top of head)

    H36M joint indices:
        0: Hip, 1: RHip, 2: RKnee, 3: RAnkle, 4: LHip, 5: LKnee, 6: LAnkle,
        7: Spine, 8: Thorax, 9: Neck, 10: Head, 11: LShoulder, 12: LElbow,
        13: LWrist, 14: RShoulder, 15: RElbow, 16: RWrist

    Args:
        keypoints: Array of shape (n_frames, 17, 3) in MotionBERT coords (Y=down)

    Returns:
        Keypoints with corrected anthropometric proportions
    """
    n_frames = keypoints.shape[0]
    corrected = keypoints.copy()

    # Standard anthropometric ratios as fraction of total height
    RATIO_TORSO = 0.30      # Hip center to shoulder center (vertical)
    RATIO_HEAD = 0.125      # Shoulder to head top (vertical)
    RATIO_UPPER = RATIO_TORSO + RATIO_HEAD  # ~0.425 upper body
    RATIO_LEG = 0.49        # Total leg (thigh + shin)
    RATIO_THIGH = 0.245     # Hip joint to knee
    RATIO_SHIN = 0.246      # Knee to ankle

    # Compute VERTICAL heights (Y axis in MotionBERT, positive = down)
    # Hip center
    hip_y = keypoints[:, 0, 1].mean()

    # Shoulder center
    l_shoulder_y = keypoints[:, 11, 1].mean()
    r_shoulder_y = keypoints[:, 14, 1].mean()
    shoulder_y = (l_shoulder_y + r_shoulder_y) / 2

    # Head
    head_y = keypoints[:, 10, 1].mean()

    # Ankles
    l_ankle_y = keypoints[:, 6, 1].mean()
    r_ankle_y = keypoints[:, 3, 1].mean()
    ankle_y = (l_ankle_y + r_ankle_y) / 2

    # Vertical torso = shoulder - hip (negative because Y=down)
    vert_torso = hip_y - shoulder_y  # Positive value
    vert_head = shoulder_y - head_y   # Positive value (head above shoulder)
    vert_upper = vert_torso + vert_head

    # Vertical leg = ankle - hip (positive because ankle has higher Y)
    vert_leg = ankle_y - hip_y

    # Total vertical height
    vert_height = vert_upper + vert_leg

    # Estimate correct height from upper body (more reliable)
    # Upper body should be ~42.5% of height
    estimated_height = vert_upper / RATIO_UPPER

    # Target leg length from estimated height
    target_leg_vertical = RATIO_LEG * estimated_height

    # Scale factor for legs (based on vertical component)
    leg_scale = target_leg_vertical / vert_leg if vert_leg > 0.01 else 1.0

    print(f"  Enforcing anthropometric proportions (vertical analysis):")
    print(f"    Vertical upper body: {vert_upper:.3f}m ({vert_upper/vert_height*100:.1f}% of current height)")
    print(f"    Vertical leg: {vert_leg:.3f}m ({vert_leg/vert_height*100:.1f}% of current height)")
    print(f"    Estimated correct height: {estimated_height:.3f}m (from upper body)")
    print(f"    Target leg vertical: {target_leg_vertical:.3f}m ({RATIO_LEG*100:.1f}%)")
    print(f"    Leg scale factor: {leg_scale:.3f}")

    # Apply uniform leg scaling to maintain joint angles while fixing proportions
    # Scale both thigh and shin by the same factor
    for frame in range(n_frames):
        # Get hip positions (reference points for legs)
        r_hip = corrected[frame, 1]
        l_hip = corrected[frame, 4]

        # Right leg: scale knee and ankle positions relative to hip
        r_knee_dir = corrected[frame, 2] - r_hip
        r_shin_dir = corrected[frame, 3] - corrected[frame, 2]

        # Scale knee position
        corrected[frame, 2] = r_hip + r_knee_dir * leg_scale
        # Scale ankle position (relative to new knee)
        new_r_shin_dir = r_shin_dir * leg_scale
        corrected[frame, 3] = corrected[frame, 2] + new_r_shin_dir

        # Left leg: scale knee and ankle positions relative to hip
        l_knee_dir = corrected[frame, 5] - l_hip
        l_shin_dir = corrected[frame, 6] - corrected[frame, 5]

        # Scale knee position
        corrected[frame, 5] = l_hip + l_knee_dir * leg_scale
        # Scale ankle position (relative to new knee)
        new_l_shin_dir = l_shin_dir * leg_scale
        corrected[frame, 6] = corrected[frame, 5] + new_l_shin_dir

    # Report final proportions
    new_ankle_y = (corrected[:, 3, 1].mean() + corrected[:, 6, 1].mean()) / 2
    new_vert_leg = new_ankle_y - hip_y
    new_vert_height = vert_upper + new_vert_leg

    print(f"    After correction:")
    print(f"      New vertical height: {new_vert_height:.3f}m")
    print(f"      New vertical leg: {new_vert_leg:.3f}m ({new_vert_leg/new_vert_height*100:.1f}%)")
    print(f"      Upper body: {vert_upper:.3f}m ({vert_upper/new_vert_height*100:.1f}%)")

    return corrected


def correct_forward_lean(keypoints: np.ndarray) -> np.ndarray:
    """
    Correct systematic forward/backward lean from monocular depth estimation.

    Monocular 3D pose estimation often produces a consistent tilt toward or away
    from the camera due to depth ambiguity. This function corrects this by:
    1. Computing the average spine tilt (hip-to-shoulder vector) across all frames
    2. Computing the average ground plane tilt from ankle positions
    3. Applying a rotation to make the spine vertical and ground horizontal

    The correction assumes:
    - During normal walking, the average torso should be roughly vertical
    - The ground plane (defined by feet) should be horizontal

    H36M joint indices:
        0: Hip (pelvis center), 7: Spine, 8: Thorax
        3: RAnkle, 6: LAnkle
        11: LShoulder, 14: RShoulder

    MotionBERT coordinates:
        X = lateral (right positive)
        Y = vertical (down positive)
        Z = depth (forward/away from camera positive)

    Args:
        keypoints: Array of shape (n_frames, 17, 3) in MotionBERT coordinates

    Returns:
        Corrected keypoints with lean removed
    """
    n_frames = keypoints.shape[0]
    corrected = keypoints.copy()

    # H36M indices
    HIP = 0
    SPINE = 7
    THORAX = 8
    L_SHOULDER = 11
    R_SHOULDER = 14
    L_ANKLE = 6
    R_ANKLE = 3

    # === Method 1: Spine verticality ===
    # Compute average spine direction (hip to shoulder center)
    hip_positions = keypoints[:, HIP, :]
    l_shoulder = keypoints[:, L_SHOULDER, :]
    r_shoulder = keypoints[:, R_SHOULDER, :]
    shoulder_center = (l_shoulder + r_shoulder) / 2

    spine_vectors = shoulder_center - hip_positions  # Points upward (negative Y in MotionBERT)
    avg_spine = np.mean(spine_vectors, axis=0)
    avg_spine = avg_spine / (np.linalg.norm(avg_spine) + 1e-8)

    # The ideal spine direction should be purely vertical: [0, -1, 0] (up in MotionBERT coords)
    ideal_spine = np.array([0, -1, 0])

    # === Method 2: Ground plane from ankles ===
    # Compute average ankle positions to estimate ground plane tilt
    l_ankle = keypoints[:, L_ANKLE, :]
    r_ankle = keypoints[:, R_ANKLE, :]

    # Ground plane normal should point up [0, -1, 0]
    # We estimate tilt from the average ankle height difference in Z (depth) direction
    avg_l_ankle = np.mean(l_ankle, axis=0)
    avg_r_ankle = np.mean(r_ankle, axis=0)

    # If one ankle is consistently further in Z (depth), it suggests forward lean
    # Combined approach: use both spine and ground information

    # === Compute rotation to correct lean ===
    # We need to rotate avg_spine to align with ideal_spine
    # This is a rotation around the X-axis (lateral axis) primarily

    # Compute the forward lean angle from spine
    # Project spine onto YZ plane (sagittal plane)
    spine_yz = np.array([0, avg_spine[1], avg_spine[2]])
    spine_yz = spine_yz / (np.linalg.norm(spine_yz) + 1e-8)

    # Angle between spine_yz and ideal vertical [0, -1, 0]
    # In MotionBERT: Y down positive, Z forward positive
    # Spine points from hip toward shoulders (should be mostly negative Y = upward)
    # If Z is negative, spine tilts backward (toward camera)
    # If Z is positive, spine tilts forward (away from camera)
    # We want to rotate the spine to have Z=0 (purely vertical)
    lean_angle_spine = np.arctan2(spine_yz[2], -spine_yz[1])

    # Compute lean from ankle depth difference (as secondary check)
    # If both ankles have similar Y but different Z during stance, suggests tilt
    ankle_z_diff = avg_r_ankle[2] - avg_l_ankle[2]  # Lateral difference in depth
    ankle_y_mean = (avg_l_ankle[1] + avg_r_ankle[1]) / 2

    # Weight the spine-based correction more heavily (it's more reliable)
    lean_angle = lean_angle_spine

    print(f"  Correcting forward lean:")
    print(f"    Average spine direction: [{avg_spine[0]:.3f}, {avg_spine[1]:.3f}, {avg_spine[2]:.3f}]")
    print(f"    Lean angle (from spine): {np.degrees(lean_angle_spine):.1f}°")
    print(f"    Applying correction: {np.degrees(lean_angle):.1f}° rotation around X-axis")

    # Create rotation matrix around X-axis to correct the lean
    # We rotate BY lean_angle (not negative) to bring spine Z component to zero
    # Rotation around X-axis: rotates Y toward Z with positive angle
    cos_a = np.cos(lean_angle)
    sin_a = np.sin(lean_angle)
    R_x = np.array([
        [1, 0, 0],
        [0, cos_a, -sin_a],
        [0, sin_a, cos_a]
    ])

    # Apply rotation to all keypoints
    # First, center on hip, rotate, then restore position
    for frame in range(n_frames):
        hip_pos = corrected[frame, HIP, :].copy()

        # Translate to origin (hip-centered)
        for j in range(17):
            corrected[frame, j, :] -= hip_pos

        # Apply rotation
        for j in range(17):
            corrected[frame, j, :] = R_x @ corrected[frame, j, :]

        # Translate back
        for j in range(17):
            corrected[frame, j, :] += hip_pos

    # Verify correction
    new_spine_vectors = (corrected[:, L_SHOULDER, :] + corrected[:, R_SHOULDER, :]) / 2 - corrected[:, HIP, :]
    new_avg_spine = np.mean(new_spine_vectors, axis=0)
    new_avg_spine = new_avg_spine / (np.linalg.norm(new_avg_spine) + 1e-8)
    new_lean = np.arctan2(new_avg_spine[2], -new_avg_spine[1])

    print(f"    After correction:")
    print(f"      New spine direction: [{new_avg_spine[0]:.3f}, {new_avg_spine[1]:.3f}, {new_avg_spine[2]:.3f}]")
    print(f"      Residual lean: {np.degrees(new_lean):.1f}°")

    return corrected


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
