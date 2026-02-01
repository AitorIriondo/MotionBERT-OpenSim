"""Test TRC generation without L/R correction."""
from src.trc_motionbert import load_motionbert_keypoints, transform_motionbert_to_opensim, extract_coco17_markers, export_trc
import numpy as np

# Load without L/R fix
keypoints, fps, _ = load_motionbert_keypoints('E:/VideoPoseEstimation/output_fixed/aitor_garden_walk_poses.pkl')
print(f'Loaded {keypoints.shape[0]} frames')

# Transform WITHOUT L/R fix
keypoints_osim = transform_motionbert_to_opensim(keypoints, 1.75)

# Check ranges
print(f'After transform:')
print(f'  X range: [{keypoints_osim[..., 0].min():.3f}, {keypoints_osim[..., 0].max():.3f}]')
print(f'  Y range: [{keypoints_osim[..., 1].min():.3f}, {keypoints_osim[..., 1].max():.3f}]')
print(f'  Z range: [{keypoints_osim[..., 2].min():.3f}, {keypoints_osim[..., 2].max():.3f}]')

# Extract and export
marker_names, marker_positions = extract_coco17_markers(keypoints_osim)
export_trc(marker_positions, marker_names, 'output/no_lr_fix.trc', fps)
print('Done!')
