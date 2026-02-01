# RTMPose3DOpenSim - Video to OpenSim pipeline using RTMPose3D
#
# Pipeline stages:
#   1. Pose Extraction (RTMPose3D with GPU)
#   2. Post-Processing (gap filling, coordinate transforms)
#   5. TRC Export (OpenSim marker format)
#   6. OpenSim IK (Inverse Kinematics)

from .constants import *
from .pose_extraction import extract_poses
from .post_processing import process_keypoints
from .trc_export import keypoints_to_trc
from .opensim_ik import run_inverse_kinematics

__version__ = "1.0.0"
__all__ = [
    "extract_poses",
    "process_keypoints",
    "keypoints_to_trc",
    "run_inverse_kinematics",
]
