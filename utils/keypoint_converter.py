"""
Keypoint Format Converter: COCO to Human3.6M (H36M)

COCO Format (17 keypoints):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

H36M Format (17 keypoints):
    0: Hip (pelvis), 1: RHip, 2: RKnee, 3: RFoot, 4: LHip, 5: LKnee, 6: LFoot,
    7: Spine, 8: Thorax, 9: Neck/Nose, 10: Head, 11: LShoulder, 12: LElbow,
    13: LWrist, 14: RShoulder, 15: RElbow, 16: RWrist
"""

import numpy as np
from typing import Union, Optional, Tuple


class COCOtoH36MConverter:
    """
    Converts COCO keypoint format to Human3.6M format.

    The conversion handles joints that don't have direct correspondence
    by computing derived positions (e.g., pelvis = midpoint of hips).
    """

    # COCO joint indices
    COCO_NOSE = 0
    COCO_LEFT_EYE = 1
    COCO_RIGHT_EYE = 2
    COCO_LEFT_EAR = 3
    COCO_RIGHT_EAR = 4
    COCO_LEFT_SHOULDER = 5
    COCO_RIGHT_SHOULDER = 6
    COCO_LEFT_ELBOW = 7
    COCO_RIGHT_ELBOW = 8
    COCO_LEFT_WRIST = 9
    COCO_RIGHT_WRIST = 10
    COCO_LEFT_HIP = 11
    COCO_RIGHT_HIP = 12
    COCO_LEFT_KNEE = 13
    COCO_RIGHT_KNEE = 14
    COCO_LEFT_ANKLE = 15
    COCO_RIGHT_ANKLE = 16

    # H36M joint indices
    H36M_HIP = 0
    H36M_RHIP = 1
    H36M_RKNEE = 2
    H36M_RFOOT = 3
    H36M_LHIP = 4
    H36M_LKNEE = 5
    H36M_LFOOT = 6
    H36M_SPINE = 7
    H36M_THORAX = 8
    H36M_NECK_NOSE = 9
    H36M_HEAD = 10
    H36M_LSHOULDER = 11
    H36M_LELBOW = 12
    H36M_LWRIST = 13
    H36M_RSHOULDER = 14
    H36M_RELBOW = 15
    H36M_RWRIST = 16

    # H36M joint names for reference
    H36M_JOINT_NAMES = [
        'Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
        'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow',
        'LWrist', 'RShoulder', 'RElbow', 'RWrist'
    ]

    # COCO joint names for reference
    COCO_JOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # Skeleton connections for visualization (H36M format)
    H36M_SKELETON = [
        (0, 1), (1, 2), (2, 3),      # Right leg
        (0, 4), (4, 5), (5, 6),      # Left leg
        (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
        (8, 11), (11, 12), (12, 13), # Left arm
        (8, 14), (14, 15), (15, 16)  # Right arm
    ]

    def __init__(self, spine_ratio: float = 0.5, thorax_ratio: float = 0.75):
        """
        Initialize the converter.

        Args:
            spine_ratio: Interpolation ratio for spine position between hip and thorax (0-1)
            thorax_ratio: Interpolation ratio for thorax between hip and shoulder center (0-1)
        """
        self.spine_ratio = spine_ratio
        self.thorax_ratio = thorax_ratio

    def convert(self,
                coco_keypoints: np.ndarray,
                scores: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert COCO keypoints to H36M format.

        Args:
            coco_keypoints: Array of shape (17, 2) or (17, 3) for single frame,
                           or (N, 17, 2) or (N, 17, 3) for batch/sequence
            scores: Optional confidence scores, shape (17,) or (N, 17)

        Returns:
            Tuple of (h36m_keypoints, h36m_scores)
            - h36m_keypoints: Array of shape (17, 2/3) or (N, 17, 2/3)
            - h36m_scores: Confidence scores for H36M joints, or None if not provided
        """
        coco_keypoints = np.array(coco_keypoints)

        # Handle single frame vs batch
        single_frame = coco_keypoints.ndim == 2
        if single_frame:
            coco_keypoints = coco_keypoints[np.newaxis, ...]
            if scores is not None:
                scores = np.array(scores)[np.newaxis, ...]

        n_frames = coco_keypoints.shape[0]
        n_dims = coco_keypoints.shape[-1]  # 2 for 2D, 3 for 3D

        # Initialize H36M keypoints
        h36m_keypoints = np.zeros((n_frames, 17, n_dims), dtype=np.float32)
        h36m_scores = np.zeros((n_frames, 17), dtype=np.float32) if scores is not None else None

        for i in range(n_frames):
            kpts = coco_keypoints[i]

            # Direct mappings
            h36m_keypoints[i, self.H36M_RHIP] = kpts[self.COCO_RIGHT_HIP]
            h36m_keypoints[i, self.H36M_RKNEE] = kpts[self.COCO_RIGHT_KNEE]
            h36m_keypoints[i, self.H36M_RFOOT] = kpts[self.COCO_RIGHT_ANKLE]
            h36m_keypoints[i, self.H36M_LHIP] = kpts[self.COCO_LEFT_HIP]
            h36m_keypoints[i, self.H36M_LKNEE] = kpts[self.COCO_LEFT_KNEE]
            h36m_keypoints[i, self.H36M_LFOOT] = kpts[self.COCO_LEFT_ANKLE]
            h36m_keypoints[i, self.H36M_LSHOULDER] = kpts[self.COCO_LEFT_SHOULDER]
            h36m_keypoints[i, self.H36M_LELBOW] = kpts[self.COCO_LEFT_ELBOW]
            h36m_keypoints[i, self.H36M_LWRIST] = kpts[self.COCO_LEFT_WRIST]
            h36m_keypoints[i, self.H36M_RSHOULDER] = kpts[self.COCO_RIGHT_SHOULDER]
            h36m_keypoints[i, self.H36M_RELBOW] = kpts[self.COCO_RIGHT_ELBOW]
            h36m_keypoints[i, self.H36M_RWRIST] = kpts[self.COCO_RIGHT_WRIST]
            h36m_keypoints[i, self.H36M_NECK_NOSE] = kpts[self.COCO_NOSE]

            # Derived joints
            # Hip (pelvis) = midpoint of left and right hips
            h36m_keypoints[i, self.H36M_HIP] = (
                kpts[self.COCO_LEFT_HIP] + kpts[self.COCO_RIGHT_HIP]
            ) / 2.0

            # Shoulder center (for spine computation)
            shoulder_center = (
                kpts[self.COCO_LEFT_SHOULDER] + kpts[self.COCO_RIGHT_SHOULDER]
            ) / 2.0

            # Thorax = interpolated position between hip center and shoulder center
            hip_center = h36m_keypoints[i, self.H36M_HIP]
            h36m_keypoints[i, self.H36M_THORAX] = (
                hip_center * (1 - self.thorax_ratio) + shoulder_center * self.thorax_ratio
            )

            # Spine = interpolated between hip and thorax
            h36m_keypoints[i, self.H36M_SPINE] = (
                hip_center * (1 - self.spine_ratio) +
                h36m_keypoints[i, self.H36M_THORAX] * self.spine_ratio
            )

            # Head = midpoint of ears (or eyes if ears not available)
            # Use eyes/ears average for head position (above nose)
            head_pts = []
            for idx in [self.COCO_LEFT_EAR, self.COCO_RIGHT_EAR,
                       self.COCO_LEFT_EYE, self.COCO_RIGHT_EYE]:
                if np.any(kpts[idx] != 0):
                    head_pts.append(kpts[idx])

            if head_pts:
                h36m_keypoints[i, self.H36M_HEAD] = np.mean(head_pts, axis=0)
            else:
                # Fallback: use nose with offset
                h36m_keypoints[i, self.H36M_HEAD] = kpts[self.COCO_NOSE]

            # Handle confidence scores
            if scores is not None:
                s = scores[i]
                # Direct score mappings
                h36m_scores[i, self.H36M_RHIP] = s[self.COCO_RIGHT_HIP]
                h36m_scores[i, self.H36M_RKNEE] = s[self.COCO_RIGHT_KNEE]
                h36m_scores[i, self.H36M_RFOOT] = s[self.COCO_RIGHT_ANKLE]
                h36m_scores[i, self.H36M_LHIP] = s[self.COCO_LEFT_HIP]
                h36m_scores[i, self.H36M_LKNEE] = s[self.COCO_LEFT_KNEE]
                h36m_scores[i, self.H36M_LFOOT] = s[self.COCO_LEFT_ANKLE]
                h36m_scores[i, self.H36M_LSHOULDER] = s[self.COCO_LEFT_SHOULDER]
                h36m_scores[i, self.H36M_LELBOW] = s[self.COCO_LEFT_ELBOW]
                h36m_scores[i, self.H36M_LWRIST] = s[self.COCO_LEFT_WRIST]
                h36m_scores[i, self.H36M_RSHOULDER] = s[self.COCO_RIGHT_SHOULDER]
                h36m_scores[i, self.H36M_RELBOW] = s[self.COCO_RIGHT_ELBOW]
                h36m_scores[i, self.H36M_RWRIST] = s[self.COCO_RIGHT_WRIST]
                h36m_scores[i, self.H36M_NECK_NOSE] = s[self.COCO_NOSE]

                # Derived joint scores (minimum of contributing joints)
                h36m_scores[i, self.H36M_HIP] = min(s[self.COCO_LEFT_HIP], s[self.COCO_RIGHT_HIP])
                h36m_scores[i, self.H36M_SPINE] = h36m_scores[i, self.H36M_HIP]
                h36m_scores[i, self.H36M_THORAX] = min(
                    h36m_scores[i, self.H36M_HIP],
                    min(s[self.COCO_LEFT_SHOULDER], s[self.COCO_RIGHT_SHOULDER])
                )
                h36m_scores[i, self.H36M_HEAD] = min(
                    s[self.COCO_LEFT_EYE], s[self.COCO_RIGHT_EYE],
                    s[self.COCO_LEFT_EAR], s[self.COCO_RIGHT_EAR]
                ) if all(s[[self.COCO_LEFT_EYE, self.COCO_RIGHT_EYE]] > 0) else s[self.COCO_NOSE]

        if single_frame:
            h36m_keypoints = h36m_keypoints[0]
            if h36m_scores is not None:
                h36m_scores = h36m_scores[0]

        return h36m_keypoints, h36m_scores

    @staticmethod
    def normalize_keypoints(keypoints: np.ndarray,
                           image_width: int,
                           image_height: int,
                           center_at_hip: bool = True) -> np.ndarray:
        """
        Normalize keypoints for MotionBERT input.

        Args:
            keypoints: H36M format keypoints, shape (T, 17, 2)
            image_width: Original image width
            image_height: Original image height
            center_at_hip: Whether to center keypoints at hip joint

        Returns:
            Normalized keypoints in range [-1, 1]
        """
        keypoints = keypoints.copy().astype(np.float32)

        # Normalize to [0, 1]
        keypoints[..., 0] /= image_width
        keypoints[..., 1] /= image_height

        # Shift to [-0.5, 0.5]
        keypoints -= 0.5

        # Scale to [-1, 1]
        keypoints *= 2.0

        if center_at_hip:
            # Center at hip for each frame
            hip_pos = keypoints[:, 0:1, :]  # Shape: (T, 1, 2)
            keypoints = keypoints - hip_pos

        return keypoints

    @classmethod
    def get_skeleton_connections(cls) -> list:
        """Return skeleton connection pairs for visualization."""
        return cls.H36M_SKELETON

    @classmethod
    def get_joint_names(cls) -> list:
        """Return H36M joint names."""
        return cls.H36M_JOINT_NAMES


if __name__ == "__main__":
    # Test the converter
    np.random.seed(42)
    coco_kpts = np.random.rand(17, 2) * 100 + 50
    coco_scores = np.random.rand(17)

    converter = COCOtoH36MConverter()
    h36m_kpts, h36m_scores = converter.convert(coco_kpts, coco_scores)

    print("Input COCO keypoints shape:", coco_kpts.shape)
    print("Output H36M keypoints shape:", h36m_kpts.shape)
