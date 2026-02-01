"""
RTMPose3D keypoint indices and mappings for OpenSim integration.

RTMPose3D uses COCO-WholeBody format with 133 keypoints:
- Body: 0-16 (17 keypoints, COCO format)
- Feet: 17-22 (6 keypoints)
- Face: 23-90 (68 keypoints)
- Left Hand: 91-111 (21 keypoints)
- Right Hand: 112-132 (21 keypoints)

Coordinate System:
- X: Right (positive toward right)
- Y: Depth (positive away from camera)
- Z: Up (positive upward)
- Units: Meters
"""

# =============================================================================
# RTMPose3D Body Keypoints (COCO format, indices 0-16)
# =============================================================================

RTMPOSE_NOSE = 0
RTMPOSE_LEFT_EYE = 1
RTMPOSE_RIGHT_EYE = 2
RTMPOSE_LEFT_EAR = 3
RTMPOSE_RIGHT_EAR = 4
RTMPOSE_LEFT_SHOULDER = 5
RTMPOSE_RIGHT_SHOULDER = 6
RTMPOSE_LEFT_ELBOW = 7
RTMPOSE_RIGHT_ELBOW = 8
RTMPOSE_LEFT_WRIST = 9
RTMPOSE_RIGHT_WRIST = 10
RTMPOSE_LEFT_HIP = 11
RTMPOSE_RIGHT_HIP = 12
RTMPOSE_LEFT_KNEE = 13
RTMPOSE_RIGHT_KNEE = 14
RTMPOSE_LEFT_ANKLE = 15
RTMPOSE_RIGHT_ANKLE = 16

# =============================================================================
# RTMPose3D Feet Keypoints (indices 17-22)
# =============================================================================

RTMPOSE_LEFT_BIG_TOE = 17
RTMPOSE_LEFT_SMALL_TOE = 18
RTMPOSE_LEFT_HEEL = 19
RTMPOSE_RIGHT_BIG_TOE = 20
RTMPOSE_RIGHT_SMALL_TOE = 21
RTMPOSE_RIGHT_HEEL = 22

# =============================================================================
# RTMPose3D Face Keypoints (indices 23-90)
# Not used for OpenSim, but defined for reference
# =============================================================================

FACE_KEYPOINT_START = 23
FACE_KEYPOINT_END = 90
FACE_KEYPOINT_COUNT = 68

# =============================================================================
# RTMPose3D Hand Keypoints (indices 91-132)
# =============================================================================

LEFT_HAND_BASE = 91    # Left hand: indices 91-111 (21 keypoints)
RIGHT_HAND_BASE = 112  # Right hand: indices 112-132 (21 keypoints)
HAND_KEYPOINT_COUNT = 21

# Hand landmark offsets (relative to base index)
HAND_WRIST = 0
HAND_THUMB_CMC = 1
HAND_THUMB_MCP = 2
HAND_THUMB_IP = 3
HAND_THUMB_TIP = 4
HAND_INDEX_MCP = 5
HAND_INDEX_PIP = 6
HAND_INDEX_DIP = 7
HAND_INDEX_TIP = 8
HAND_MIDDLE_MCP = 9
HAND_MIDDLE_PIP = 10
HAND_MIDDLE_DIP = 11
HAND_MIDDLE_TIP = 12
HAND_RING_MCP = 13
HAND_RING_PIP = 14
HAND_RING_DIP = 15
HAND_RING_TIP = 16
HAND_PINKY_MCP = 17
HAND_PINKY_PIP = 18
HAND_PINKY_DIP = 19
HAND_PINKY_TIP = 20

# =============================================================================
# Total Keypoint Count
# =============================================================================

TOTAL_KEYPOINTS = 133
BODY_KEYPOINTS = 17
FEET_KEYPOINTS = 6
FACE_KEYPOINTS = 68
HAND_KEYPOINTS = 42  # 21 per hand

# =============================================================================
# Keypoints used for OpenSim marker export
# =============================================================================

# Body keypoints needed for biomechanical analysis
OPENSIM_BODY_KEYPOINTS = [
    RTMPOSE_NOSE,
    RTMPOSE_LEFT_EYE,
    RTMPOSE_RIGHT_EYE,
    RTMPOSE_LEFT_EAR,
    RTMPOSE_RIGHT_EAR,
    RTMPOSE_LEFT_SHOULDER,
    RTMPOSE_RIGHT_SHOULDER,
    RTMPOSE_LEFT_ELBOW,
    RTMPOSE_RIGHT_ELBOW,
    RTMPOSE_LEFT_WRIST,
    RTMPOSE_RIGHT_WRIST,
    RTMPOSE_LEFT_HIP,
    RTMPOSE_RIGHT_HIP,
    RTMPOSE_LEFT_KNEE,
    RTMPOSE_RIGHT_KNEE,
    RTMPOSE_LEFT_ANKLE,
    RTMPOSE_RIGHT_ANKLE,
]

# Feet keypoints for ground contact
OPENSIM_FEET_KEYPOINTS = [
    RTMPOSE_LEFT_BIG_TOE,
    RTMPOSE_LEFT_SMALL_TOE,
    RTMPOSE_LEFT_HEEL,
    RTMPOSE_RIGHT_BIG_TOE,
    RTMPOSE_RIGHT_SMALL_TOE,
    RTMPOSE_RIGHT_HEEL,
]

# =============================================================================
# Keypoint Names (for JSON output and debugging)
# =============================================================================

KEYPOINT_NAMES = {
    # Body
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
    # Feet
    17: "left_big_toe",
    18: "left_small_toe",
    19: "left_heel",
    20: "right_big_toe",
    21: "right_small_toe",
    22: "right_heel",
}

# =============================================================================
# Skeleton Connections (for visualization)
# =============================================================================

SKELETON_CONNECTIONS = [
    # Head
    (RTMPOSE_NOSE, RTMPOSE_LEFT_EYE),
    (RTMPOSE_NOSE, RTMPOSE_RIGHT_EYE),
    (RTMPOSE_LEFT_EYE, RTMPOSE_LEFT_EAR),
    (RTMPOSE_RIGHT_EYE, RTMPOSE_RIGHT_EAR),
    # Torso
    (RTMPOSE_LEFT_SHOULDER, RTMPOSE_RIGHT_SHOULDER),
    (RTMPOSE_LEFT_SHOULDER, RTMPOSE_LEFT_HIP),
    (RTMPOSE_RIGHT_SHOULDER, RTMPOSE_RIGHT_HIP),
    (RTMPOSE_LEFT_HIP, RTMPOSE_RIGHT_HIP),
    # Left arm
    (RTMPOSE_LEFT_SHOULDER, RTMPOSE_LEFT_ELBOW),
    (RTMPOSE_LEFT_ELBOW, RTMPOSE_LEFT_WRIST),
    # Right arm
    (RTMPOSE_RIGHT_SHOULDER, RTMPOSE_RIGHT_ELBOW),
    (RTMPOSE_RIGHT_ELBOW, RTMPOSE_RIGHT_WRIST),
    # Left leg
    (RTMPOSE_LEFT_HIP, RTMPOSE_LEFT_KNEE),
    (RTMPOSE_LEFT_KNEE, RTMPOSE_LEFT_ANKLE),
    (RTMPOSE_LEFT_ANKLE, RTMPOSE_LEFT_HEEL),
    (RTMPOSE_LEFT_ANKLE, RTMPOSE_LEFT_BIG_TOE),
    (RTMPOSE_LEFT_BIG_TOE, RTMPOSE_LEFT_SMALL_TOE),
    # Right leg
    (RTMPOSE_RIGHT_HIP, RTMPOSE_RIGHT_KNEE),
    (RTMPOSE_RIGHT_KNEE, RTMPOSE_RIGHT_ANKLE),
    (RTMPOSE_RIGHT_ANKLE, RTMPOSE_RIGHT_HEEL),
    (RTMPOSE_RIGHT_ANKLE, RTMPOSE_RIGHT_BIG_TOE),
    (RTMPOSE_RIGHT_BIG_TOE, RTMPOSE_RIGHT_SMALL_TOE),
]

# =============================================================================
# OpenSim Marker Definitions
# =============================================================================

# Marker definitions for TRC export
# Format: (marker_name, source_type, source_data)
# source_type: 'landmark' (direct keypoint), 'midpoint' (average of two),
#              'offset' (fraction along vector), 'virtual' (computed from pelvis)

MARKER_DEFINITIONS = [
    # Joint centers (direct landmarks)
    ("LHJC", "landmark", RTMPOSE_LEFT_HIP),       # Left Hip Joint Center
    ("RHJC", "landmark", RTMPOSE_RIGHT_HIP),      # Right Hip Joint Center
    ("LKJC", "landmark", RTMPOSE_LEFT_KNEE),      # Left Knee Joint Center
    ("RKJC", "landmark", RTMPOSE_RIGHT_KNEE),     # Right Knee Joint Center
    ("LAJC", "landmark", RTMPOSE_LEFT_ANKLE),     # Left Ankle Joint Center
    ("RAJC", "landmark", RTMPOSE_RIGHT_ANKLE),    # Right Ankle Joint Center
    ("LSJC", "landmark", RTMPOSE_LEFT_SHOULDER),  # Left Shoulder Joint Center
    ("RSJC", "landmark", RTMPOSE_RIGHT_SHOULDER), # Right Shoulder Joint Center
    ("LEJC", "landmark", RTMPOSE_LEFT_ELBOW),     # Left Elbow Joint Center
    ("REJC", "landmark", RTMPOSE_RIGHT_ELBOW),    # Right Elbow Joint Center
    ("LWJC", "landmark", RTMPOSE_LEFT_WRIST),     # Left Wrist Joint Center
    ("RWJC", "landmark", RTMPOSE_RIGHT_WRIST),    # Right Wrist Joint Center

    # Feet markers
    ("LHEE", "landmark", RTMPOSE_LEFT_HEEL),      # Left Heel
    ("RHEE", "landmark", RTMPOSE_RIGHT_HEEL),     # Right Heel
    ("LTOE", "landmark", RTMPOSE_LEFT_BIG_TOE),   # Left Toe
    ("RTOE", "landmark", RTMPOSE_RIGHT_BIG_TOE),  # Right Toe

    # Midpoint markers
    ("CLAV", "midpoint", (RTMPOSE_LEFT_SHOULDER, RTMPOSE_RIGHT_SHOULDER)),  # Clavicle (sternum)
    ("PELV", "midpoint", (RTMPOSE_LEFT_HIP, RTMPOSE_RIGHT_HIP)),            # Pelvis center

    # Virtual markers (computed from pelvis geometry)
    ("LASI", "virtual", "l_asis"),   # Left Anterior Superior Iliac Spine
    ("RASI", "virtual", "r_asis"),   # Right Anterior Superior Iliac Spine
    ("LPSI", "virtual", "l_psis"),   # Left Posterior Superior Iliac Spine
    ("RPSI", "virtual", "r_psis"),   # Right Posterior Superior Iliac Spine

    # Offset markers (lateral condyles)
    ("LLFC", "offset", (RTMPOSE_LEFT_KNEE, RTMPOSE_LEFT_HIP, 0.05)),   # Left Lateral Femoral Condyle
    ("RLFC", "offset", (RTMPOSE_RIGHT_KNEE, RTMPOSE_RIGHT_HIP, 0.05)), # Right Lateral Femoral Condyle
    ("LLMC", "offset", (RTMPOSE_LEFT_KNEE, RTMPOSE_RIGHT_KNEE, 0.05)), # Left Medial Femoral Condyle
    ("RLMC", "offset", (RTMPOSE_RIGHT_KNEE, RTMPOSE_LEFT_KNEE, 0.05)), # Right Medial Femoral Condyle

    # Head marker
    ("HEAD", "landmark", RTMPOSE_NOSE),  # Head (approximated by nose)
]

# =============================================================================
# Rajagopal Model Marker Mapping
# =============================================================================

# Mapping from our TRC markers to Rajagopal2023.osim markers
# Format: (osim_marker_name, trc_marker_name, weight)
RAJAGOPAL_MARKER_MAPPING = [
    # Pelvis markers (high weight - anchor the model)
    ("L.ASIS", "LASI", 10.0),
    ("R.ASIS", "RASI", 10.0),
    ("L.PSIS", "LPSI", 10.0),
    ("R.PSIS", "RPSI", 10.0),

    # Hip joint centers
    ("L.Hip", "LHJC", 5.0),
    ("R.Hip", "RHJC", 5.0),

    # Knee markers
    ("L.Knee", "LKJC", 5.0),
    ("R.Knee", "RKJC", 5.0),
    ("L.Knee.Lat", "LLFC", 3.0),
    ("R.Knee.Lat", "RLFC", 3.0),

    # Ankle markers
    ("L.Ankle", "LAJC", 5.0),
    ("R.Ankle", "RAJC", 5.0),

    # Foot markers
    ("L.Heel", "LHEE", 3.0),
    ("R.Heel", "RHEE", 3.0),
    ("L.Toe", "LTOE", 3.0),
    ("R.Toe", "RTOE", 3.0),

    # Upper body markers
    ("Sternum", "CLAV", 5.0),
    ("L.Shoulder", "LSJC", 3.0),
    ("R.Shoulder", "RSJC", 3.0),
    ("L.Elbow", "LEJC", 3.0),
    ("R.Elbow", "REJC", 3.0),
    ("L.Wrist", "LWJC", 2.0),
    ("R.Wrist", "RWJC", 2.0),
]

# =============================================================================
# Coordinate System Information
# =============================================================================

COORDINATE_SYSTEMS = {
    "rtmpose3d": {
        "description": "RTMPose3D camera-relative coordinates",
        "x": "right (positive toward right)",
        "y": "depth (positive away from camera)",
        "z": "up (positive upward)",
        "units": "meters",
    },
    "opensim": {
        "description": "OpenSim anatomical coordinates",
        "x": "anterior (positive forward)",
        "y": "superior (positive upward)",
        "z": "lateral (positive toward right)",
        "units": "meters (converted to mm for TRC)",
    },
}
