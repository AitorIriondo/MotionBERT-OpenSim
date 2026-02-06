# Claude Memory - FBX Export Project

## Current Status: WORKING
FBX export now uses the expert-provided `metarig_skely` skeleton with verified rotation mappings.

## Key Files
**Export Script (current, working):**
- `export_fbx_skely.py` - FBX export using metarig_skely skeleton (WORKING)

**Skeleton Template:**
- `C:\MotionBERTOpenSim\Import_OS4_Patreon_Aitor_Skely.blend` - Expert's skeleton with 29 bones + 18 mesh body parts

**Old/Deprecated Scripts (Y-Bot approach, abandoned):**
- `export_fbx_ybot.py` - Y-Bot export (broken due to arm bone orientation differences)
- `export_fbx_stick.py` - Stick figure export
- `export_fbx.py` - Original pyopensim-based export
- `debug_ybot_pose_v*.py` - Debug iterations

**Reference:**
- `import_mot_to_blender.py` - Working Endorfina metarig import (reference implementation)
- Expert code from `Import_OS4_Patreon_Aitor.blend` - Verified rotation mappings

## Expert-Verified Rotation Mappings (Endorfina/metarig_skely)

All bones use Euler XYZ rotation mode. Values from .mot file (in degrees) converted to radians.

| Bone | Blender Euler XYZ | Signs |
|------|------------------|-------|
| spine (pelvis) | (-pelvis_tilt, pelvis_rotation, pelvis_list) | (-,+,+) |
| spine location | (-pelvis_tz, pelvis_ty, pelvis_tx) | |
| thigh.R | (-hip_flexion_r, -hip_rotation_r, -hip_adduction_r) | (-,-,-) |
| shin.R | (+knee_angle_r, 0, 0) | (+) NOTE: POSITIVE |
| foot.R | (-ankle_angle_r, 0, 0) | (-) |
| thigh.L | (-hip_flexion_l, +hip_rotation_l, +hip_adduction_l) | (-,+,+) |
| shin.L | (+knee_angle_l, 0, 0) | (+) |
| foot.L | (-ankle_angle_l, 0, 0) | (-) |
| spine.001 (lumbar) | (-lumbar_extension, lumbar_rotation, lumbar_bending) | (-,+,+) |
| spine.002 (thorax) | (-thorax_extension, thorax_rotation, thorax_bending) | (-,+,+) |
| upper_arm.R | (-arm_flex_r, -arm_rot_r, -arm_add_r) | (-,-,-) |
| forearm.R | (-elbow_flex_r, 0, 0) | (-) |
| upper_arm.L | (-arm_flex_l, +arm_rot_l, +arm_add_l) | (-,+,+) |
| forearm.L | (-elbow_flex_l, 0, 0) | (-) |

Pattern: Right side has all negative signs. Left side flips rotation and adduction/bending to positive.

## Column Name Mapping
Our .mot files use different column names than the expert's:
- `L5_S1_Flex_Ext` = expert's `lumbar_extension`
- `L5_S1_Lat_Bending` = expert's `lumbar_bending`
- `L5_S1_axial_rotation` = expert's `lumbar_rotation`
- `neck_flexion` = expert's `thorax_extension`
- `neck_bending` = expert's `thorax_bending`
- `neck_rotation` = expert's `thorax_rotation`

## Why Y-Bot Failed
The Y-Bot (Mixamo) skeleton has fundamentally different arm bone orientations:
- Y-Bot arms point HORIZONTAL (T-pose), bone roll = +/-90deg
- metarig_skely/Endorfina arms point DOWN, bone roll = 0deg
- Matrix-based coordinate transforms were mathematically correct but Blender's Euler rotation mode handling
  (QUATERNION default on Y-Bot) prevented proper application
- Using the expert's skeleton with matching bone orientations eliminates the problem entirely

## Pipeline Integration
`run_full_pipeline.py` updated:
- `--export-fbx` flag triggers FBX export using `export_fbx_skely.py`
- Requires `Import_OS4_Patreon_Aitor_Skely.blend` in the project root directory
- Command: `blender --background template.blend --python export_fbx_skely.py -- --mot motion.mot --output motion.fbx`

## Blender Command
```bash
"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background "C:\MotionBERTOpenSim\Import_OS4_Patreon_Aitor_Skely.blend" --python export_fbx_skely.py -- --mot motion.mot --output motion.fbx
```

## Test Data
- Motion: `C:\MotionBERTOpenSim\output_20260203_161244_aitor_garden_walk\kinematics\motion.mot`
- Template: `C:\MotionBERTOpenSim\Import_OS4_Patreon_Aitor_Skely.blend`
- Verification renders: `skely_ortho_front_f*.png`, `skely_ortho_side_f*.png`
