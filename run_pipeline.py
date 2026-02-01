#!/usr/bin/env python3
"""
RTMPose3DOpenSim - Video to OpenSim pipeline using RTMPose3D.

Pipeline stages:
    1. Pose Extraction - RTMPose3D GPU inference
    2. Post-Processing - Gap filling, coordinate transforms
    5. TRC Export - OpenSim marker format
    6. OpenSim IK - Inverse Kinematics

Usage:
    python run_pipeline.py input/video.mp4 --export-trc --run-ik
    python run_pipeline.py input/video.mp4 --height 1.75 --smooth 6.0
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def main():
    parser = argparse.ArgumentParser(
        description="RTMPose3DOpenSim: Video to OpenSim pipeline using RTMPose3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with IK
    python run_pipeline.py input/walk.mp4 --export-trc --run-ik

    # With height scaling and smoothing
    python run_pipeline.py input/walk.mp4 --height 1.75 --smooth 6.0 --run-ik

    # Skip extraction (reuse existing keypoints)
    python run_pipeline.py input/walk.mp4 --skip-extraction --run-ik

    # CPU-only mode
    python run_pipeline.py input/walk.mp4 --device cpu --export-trc
        """
    )

    # Input
    parser.add_argument(
        "video",
        help="Path to input video file"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory (default: output)"
    )

    # Device
    parser.add_argument(
        "--device", "-d",
        default="cuda",
        help="Device for RTMW3D inference: 'cuda' or 'cpu' (default: cuda)"
    )

    # Person tracking
    parser.add_argument(
        "--person", "-p",
        type=int,
        default=0,
        help="Person index to track in multi-person scenes (default: 0)"
    )

    # Processing options
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for gap filling (default: 0.3)"
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=None,
        help="Butterworth filter cutoff frequency in Hz (default: None)"
    )
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="Target height in meters for scaling (default: None)"
    )
    parser.add_argument(
        "--no-body-frame",
        action="store_true",
        help="Disable body-frame rotation (use global axes)"
    )

    # Pipeline control
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip pose extraction (use existing keypoints)"
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip post-processing (use existing processed data)"
    )

    # Export options
    parser.add_argument(
        "--export-trc",
        action="store_true",
        help="Export TRC marker file (Rajagopal format)"
    )
    parser.add_argument(
        "--export-trc-pose2sim",
        action="store_true",
        help="Export TRC with HALPE_26 names for Pose2Sim pipeline"
    )
    parser.add_argument(
        "--run-ik",
        action="store_true",
        help="Run OpenSim Inverse Kinematics"
    )
    parser.add_argument(
        "--run-pose2sim",
        action="store_true",
        help="Run Pose2Sim markerAugmentation and kinematics"
    )

    # OpenSim model
    parser.add_argument(
        "--model",
        default="models/Rajagopal2023.osim",
        help="Path to OpenSim model file (default: models/Rajagopal2023.osim)"
    )

    args = parser.parse_args()

    # Validate video path
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Setup output directories
    output_dir = Path(args.output)
    keypoints_dir = output_dir / "keypoints"
    processed_dir = output_dir / "processed"
    markers_dir = output_dir / "markers"
    ik_dir = output_dir / "ik"

    # Derive file paths
    video_stem = video_path.stem
    keypoints_json = keypoints_dir / f"{video_stem}_rtmpose3d.json"
    processed_json = processed_dir / f"{video_stem}_processed.json"
    trc_file = markers_dir / f"{video_stem}_markers.trc"
    trc_halpe26_file = markers_dir / f"{video_stem}_halpe26.trc"
    mot_file = ik_dir / f"{video_stem}_ik.mot"

    # Validate height for Pose2Sim export
    if (args.export_trc_pose2sim or args.run_pose2sim) and args.height is None:
        print("Error: --height is required for --export-trc-pose2sim or --run-pose2sim")
        print("Example: --export-trc-pose2sim --height 1.68")
        sys.exit(1)

    print("=" * 70)
    print("RTMPose3DOpenSim Pipeline")
    print("=" * 70)
    print(f"Input: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print()

    # =========================================================================
    # Stage 1: Pose Extraction
    # =========================================================================
    if not args.skip_extraction:
        print("=" * 70)
        print("Stage 1: Pose Extraction (RTMPose3D)")
        print("=" * 70)

        from src.pose_extraction import extract_poses

        keypoints_json = extract_poses(
            str(video_path),
            str(keypoints_dir),
            device=args.device,
            person_idx=args.person,
        )
        print()
    else:
        print("Skipping Stage 1 (using existing keypoints)")
        if not keypoints_json.exists():
            print(f"Error: Keypoints file not found: {keypoints_json}")
            sys.exit(1)
        keypoints_json = str(keypoints_json)
        print()

    # =========================================================================
    # Stage 2: Post-Processing
    # =========================================================================
    if not args.skip_processing:
        print("=" * 70)
        print("Stage 2: Post-Processing")
        print("=" * 70)

        from src.post_processing import process_keypoints

        processed_json = process_keypoints(
            keypoints_json,
            str(processed_dir),
            fill_gaps_threshold=args.gap_threshold,
            smooth_cutoff=args.smooth,
            target_height=args.height,
            use_body_frame=not args.no_body_frame,
        )
        print()
    else:
        print("Skipping Stage 2 (using existing processed data)")
        if not processed_json.exists():
            print(f"Error: Processed file not found: {processed_json}")
            sys.exit(1)
        processed_json = str(processed_json)
        print()

    # =========================================================================
    # Stage 5: TRC Export
    # =========================================================================
    if args.export_trc or args.run_ik:
        print("=" * 70)
        print("Stage 5: TRC Marker Export (Rajagopal)")
        print("=" * 70)

        from src.trc_export import keypoints_to_trc

        trc_file = keypoints_to_trc(
            processed_json,
            str(markers_dir),
        )
        print()

    # =========================================================================
    # Stage 5b: TRC Export for Pose2Sim (HALPE_26 format)
    # =========================================================================
    if args.export_trc_pose2sim or args.run_pose2sim:
        print("=" * 70)
        print("Stage 5b: TRC Export for Pose2Sim (HALPE_26)")
        print("=" * 70)

        from src.trc_pose2sim import export_trc_for_pose2sim

        trc_halpe26_file, trc_info = export_trc_for_pose2sim(
            processed_json,
            str(trc_halpe26_file),
            real_height=args.height,
        )
        print()

    # =========================================================================
    # Stage 6: OpenSim IK (traditional)
    # =========================================================================
    if args.run_ik:
        print("=" * 70)
        print("Stage 6: OpenSim Inverse Kinematics")
        print("=" * 70)

        # Check model file
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            print("Copy Rajagopal2023.osim to models/ directory")
            sys.exit(1)

        from src.opensim_ik import run_inverse_kinematics

        ik_outputs = run_inverse_kinematics(
            str(model_path),
            trc_file,
            str(ik_dir),
        )
        print()

    # =========================================================================
    # Stage 7: Pose2Sim markerAugmentation + kinematics
    # =========================================================================
    if args.run_pose2sim:
        print("=" * 70)
        print("Stage 7: Pose2Sim Pipeline")
        print("=" * 70)

        try:
            from src.trc_pose2sim import run_pose2sim_kinematics

            pose2sim_results = run_pose2sim_kinematics(
                trc_halpe26_file,
                str(ik_dir),
                participant_height=args.height,
                participant_mass=70.0,
                use_augmentation=False,  # Skip augmentation for now
            )
            print()

        except ImportError as e:
            print(f"  Error: Pose2Sim not available: {e}")
            print("  Install with: pip install pose2sim")
        except Exception as e:
            print(f"  Error running Pose2Sim: {e}")
            import traceback
            traceback.print_exc()

        print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print()
    print("Output files:")

    if not args.skip_extraction:
        print(f"  Keypoints: {keypoints_json}")

    if not args.skip_processing:
        print(f"  Processed: {processed_json}")

    if args.export_trc or args.run_ik:
        print(f"  TRC:       {trc_file}")

    if args.export_trc_pose2sim or args.run_pose2sim:
        print(f"  TRC (HALPE_26): {trc_halpe26_file}")

    if args.run_ik:
        print(f"  IK MOT:    {ik_outputs['mot']}")

    print()


if __name__ == "__main__":
    main()
