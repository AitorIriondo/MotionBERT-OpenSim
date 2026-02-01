"""
Stage 6: OpenSim Inverse Kinematics.

Runs OpenSim's IK solver on TRC marker data to compute joint angles
that best fit the experimental markers using the Rajagopal musculoskeletal model.

Requirements:
    - OpenSim Python API: conda install opensim-org::opensim
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import opensim as osim
    OPENSIM_AVAILABLE = True
except ImportError:
    OPENSIM_AVAILABLE = False
    print("Warning: OpenSim Python API not available. Install with:")
    print("  conda install opensim-org::opensim")

from .constants import RAJAGOPAL_MARKER_MAPPING


def create_marker_set_file(
    marker_mapping: List[Tuple[str, str, float]],
    output_path: str,
) -> str:
    """
    Create an OpenSim MarkerSet XML file for IK.

    Args:
        marker_mapping: List of (osim_marker, trc_marker, weight) tuples.
        output_path: Path to save the marker set file.

    Returns:
        Path to created file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build XML content
    xml_content = '''<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <IKTaskSet name="rtmpose3d_markers">
        <objects>
'''

    for osim_name, trc_name, weight in marker_mapping:
        xml_content += f'''            <IKMarkerTask name="{osim_name}">
                <apply>true</apply>
                <weight>{weight}</weight>
            </IKMarkerTask>
'''

    xml_content += '''        </objects>
    </IKTaskSet>
</OpenSimDocument>
'''

    with open(output_path, "w") as f:
        f.write(xml_content)

    return str(output_path)


def create_ik_setup_file(
    model_file: str,
    marker_file: str,
    output_motion_file: str,
    time_range: Tuple[float, float],
    output_path: str,
    marker_set_file: Optional[str] = None,
    accuracy: float = 1e-5,
) -> str:
    """
    Create an OpenSim IK setup XML file.

    Args:
        model_file: Path to .osim model file.
        marker_file: Path to .trc marker file.
        output_motion_file: Path for output .mot file.
        time_range: (start_time, end_time) in seconds.
        output_path: Path to save the setup file.
        marker_set_file: Optional path to marker task set file.
        accuracy: IK solver accuracy.

    Returns:
        Path to created file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dir = Path(output_motion_file).parent

    xml_content = f'''<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <InverseKinematicsTool name="ik_tool">
        <model_file>{model_file}</model_file>
        <constraint_weight>20</constraint_weight>
        <accuracy>{accuracy}</accuracy>
        <marker_file>{marker_file}</marker_file>
        <coordinate_file></coordinate_file>
        <time_range>{time_range[0]} {time_range[1]}</time_range>
        <output_motion_file>{output_motion_file}</output_motion_file>
        <report_errors>true</report_errors>
        <report_marker_locations>false</report_marker_locations>
        <results_directory>{results_dir}</results_directory>
'''

    if marker_set_file:
        xml_content += f'''        <IKTaskSet file="{marker_set_file}"/>
'''

    xml_content += '''    </InverseKinematicsTool>
</OpenSimDocument>
'''

    with open(output_path, "w") as f:
        f.write(xml_content)

    return str(output_path)


def get_trc_time_range(trc_file: str) -> Tuple[float, float]:
    """
    Read time range from TRC file header.

    Args:
        trc_file: Path to TRC file.

    Returns:
        Tuple of (start_time, end_time) in seconds.
    """
    with open(trc_file, "r") as f:
        lines = f.readlines()

    # Parse header line 3 (0-indexed line 2)
    header_values = lines[2].split("\t")
    data_rate = float(header_values[0])
    num_frames = int(header_values[2])

    return (0.0, (num_frames - 1) / data_rate)


def run_inverse_kinematics(
    model_file: str,
    trc_file: str,
    output_dir: str,
    time_range: Optional[Tuple[float, float]] = None,
    show_progress: bool = True,
) -> Dict[str, str]:
    """
    Run OpenSim Inverse Kinematics on marker data.

    Args:
        model_file: Path to OpenSim model (.osim).
        trc_file: Path to marker data (.trc).
        output_dir: Directory for output files.
        time_range: Optional (start, end) time in seconds. If None, uses full trial.
        show_progress: Print progress updates.

    Returns:
        Dictionary with output file paths.
    """
    if not OPENSIM_AVAILABLE:
        raise RuntimeError(
            "OpenSim Python API not available. Install with:\n"
            "  conda install opensim-org::opensim"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trc_path = Path(trc_file)
    stem = trc_path.stem.replace("_markers", "")

    # Output paths
    output_mot = output_dir / f"{stem}_ik.mot"
    setup_file = output_dir / f"{stem}_ik_setup.xml"
    marker_set_file = output_dir / f"{stem}_ik_markers.xml"

    if show_progress:
        print("Running OpenSim Inverse Kinematics...")
        print(f"  Model: {model_file}")
        print(f"  Markers: {trc_file}")

    # Load model
    model = osim.Model(model_file)
    model.initSystem()

    # Get time range from TRC file if not specified
    if time_range is None:
        time_range = get_trc_time_range(trc_file)

    if show_progress:
        print(f"  Time range: {time_range[0]:.3f} - {time_range[1]:.3f} s")

    # Create marker set file with weights
    create_marker_set_file(RAJAGOPAL_MARKER_MAPPING, str(marker_set_file))

    # Also save setup file for reference
    create_ik_setup_file(
        model_file=model_file,
        marker_file=trc_file,
        output_motion_file=str(output_mot),
        time_range=time_range,
        output_path=str(setup_file),
        marker_set_file=str(marker_set_file),
    )

    # Setup IK tool
    ik_tool = osim.InverseKinematicsTool()
    ik_tool.setModel(model)
    ik_tool.setMarkerDataFileName(str(trc_file))
    ik_tool.setStartTime(time_range[0])
    ik_tool.setEndTime(time_range[1])
    ik_tool.setOutputMotionFileName(str(output_mot))
    ik_tool.setResultsDir(str(output_dir))

    # Run IK
    if show_progress:
        print("  Running IK solver...")

    try:
        ik_tool.run()
        if show_progress:
            print(f"  Output: {output_mot}")
    except Exception as e:
        print(f"  IK failed: {e}")
        raise

    return {
        "mot": str(output_mot),
        "setup": str(setup_file),
        "marker_set": str(marker_set_file),
    }


def run_ik_from_processed(
    processed_json: str,
    model_file: str,
    output_dir: str,
    show_progress: bool = True,
) -> Dict[str, str]:
    """
    Complete pipeline: processed keypoints -> TRC -> OpenSim IK -> MOT.

    Args:
        processed_json: Path to processed keypoints JSON.
        model_file: Path to OpenSim model (.osim).
        output_dir: Directory for all output files.
        show_progress: Print progress updates.

    Returns:
        Dictionary with output file paths.
    """
    from .trc_export import keypoints_to_trc

    output_dir = Path(output_dir)
    markers_dir = output_dir / "markers"
    ik_dir = output_dir / "ik"

    # Step 1: Convert keypoints to TRC
    if show_progress:
        print("=" * 60)
        print("Step 1: Convert keypoints to TRC markers")
        print("=" * 60)

    trc_file = keypoints_to_trc(
        processed_json,
        str(markers_dir),
        show_progress=show_progress,
    )

    if show_progress:
        print()

    # Step 2: Run IK
    if show_progress:
        print("=" * 60)
        print("Step 2: Run OpenSim Inverse Kinematics")
        print("=" * 60)

    outputs = run_inverse_kinematics(
        model_file=model_file,
        trc_file=trc_file,
        output_dir=str(ik_dir),
        show_progress=show_progress,
    )

    outputs["trc"] = trc_file

    return outputs


def validate_mot_file(mot_path: str) -> Dict:
    """
    Validate MOT file and return summary statistics.

    Args:
        mot_path: Path to MOT file.

    Returns:
        Dictionary with validation results.
    """
    with open(mot_path, "r") as f:
        lines = f.readlines()

    # Find header end
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == "endheader":
            header_end = i + 1
            break

    # Parse column names
    column_line = lines[header_end].strip()
    columns = column_line.split("\t")

    # Parse data
    n_rows = len(lines) - header_end - 1
    time_values = []
    for line in lines[header_end + 1:]:
        values = line.strip().split("\t")
        if values:
            time_values.append(float(values[0]))

    return {
        "n_frames": n_rows,
        "n_columns": len(columns),
        "columns": columns,
        "time_range": (min(time_values), max(time_values)) if time_values else (0, 0),
        "duration": max(time_values) - min(time_values) if time_values else 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OpenSim IK on keypoints")
    parser.add_argument("input", help="Path to processed keypoints JSON or TRC file")
    parser.add_argument(
        "--model", "-m",
        default="models/Rajagopal2023.osim",
        help="Path to OpenSim model file (default: models/Rajagopal2023.osim)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate output MOT file"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.suffix == ".trc":
        # Direct TRC input
        outputs = run_inverse_kinematics(
            args.model,
            str(input_path),
            args.output,
        )
    else:
        # JSON input - run full pipeline
        outputs = run_ik_from_processed(
            str(input_path),
            args.model,
            args.output,
        )

    print()
    print("=" * 60)
    print("Complete!")
    print("=" * 60)
    for key, path in outputs.items():
        print(f"  {key}: {path}")

    if args.validate and "mot" in outputs:
        print()
        print("Validating MOT file...")
        stats = validate_mot_file(outputs["mot"])
        print(f"  Frames: {stats['n_frames']}")
        print(f"  Columns: {stats['n_columns']}")
        print(f"  Duration: {stats['duration']:.2f}s")
