"""Test Pose2Sim compute_height function."""
import numpy as np
import json

# Load processed keypoints
with open('output/processed/aitor_garden_walk_processed.json', 'r') as f:
    data = json.load(f)

keypoints = np.array(data['keypoints'])
print(f"Keypoints shape: {keypoints.shape}")

# Simple HALPE_26 conversion (just a few markers for testing)
from src.trc_pose2sim import compute_halpe26_markers

marker_names, markers = compute_halpe26_markers(keypoints)
print(f"Markers shape: {markers.shape}")
print(f"Marker names: {marker_names}")

# Test compute_height
print("\nTrying Pose2Sim compute_height...")
try:
    from Pose2Sim.common import compute_height
    height = compute_height(markers, marker_names)
    print(f"Computed height: {height}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
