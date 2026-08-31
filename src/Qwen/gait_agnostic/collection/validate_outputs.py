#!/usr/bin/env python3
"""Validate raw collector outputs and CSV/NPZ equivalence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXPECTED_SCORE_KEYS = (
    "vel_track_x",
    "safe_height",
    "safe_orientation",
    "low_slip",
    "rho_task_safety",
    "rho_gait_quality",
    "total_stl_reward",
    "diagnostic_abs_velocity_error_y",
    "diagnostic_abs_velocity_error_yaw",
)


def validate_recorded_arrays(arrays: dict[str, np.ndarray]) -> None:
    expected_shape = (50, 500)
    required = {
        "trajectory",
        "timestep",
        "command_vx",
        "command_vy",
        "command_yaw",
        "mode",
        "done_this_step",
        "trajectory_terminated",
        "first_termination_timestep",
        *EXPECTED_SCORE_KEYS,
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"Missing arrays: {missing}")
    for key in required:
        if np.asarray(arrays[key]).shape != expected_shape:
            raise ValueError(
                f"Array {key} has shape {np.asarray(arrays[key]).shape}, "
                f"expected {expected_shape}"
            )
    for key in ("command_vx", "command_vy", "command_yaw", *EXPECTED_SCORE_KEYS):
        if not np.isfinite(np.asarray(arrays[key], dtype=np.float64)).all():
            raise ValueError(f"Non-finite values detected in {key}")


def validate_csv_and_npz_match(
    csv_path: Path, npz_path: Path, metadata_path: Path
) -> dict[str, object]:
    arrays = {key: value for key, value in np.load(npz_path, allow_pickle=False).items()}
    validate_recorded_arrays(arrays)
    csv_rows = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for csv_rows, row in enumerate(reader, start=1):
            trajectory = int(row["trajectory"])
            timestep = int(row["timestep"])
            for key in (
                "trajectory",
                "timestep",
                "mode",
                "first_termination_timestep",
            ):
                if int(arrays[key][trajectory, timestep]) != int(row[key]):
                    raise ValueError(f"CSV/NPZ equivalence mismatch for {key}")
            for key in ("command_vx", "command_vy", "command_yaw", *EXPECTED_SCORE_KEYS):
                if not np.isclose(
                    float(arrays[key][trajectory, timestep]),
                    float(row[key]),
                    atol=1e-7,
                    rtol=0.0,
                ):
                    raise ValueError(f"CSV/NPZ equivalence mismatch for {key}")
            for key in ("done_this_step", "trajectory_terminated"):
                if int(bool(arrays[key][trajectory, timestep])) != int(row[key]):
                    raise ValueError(f"CSV/NPZ equivalence mismatch for {key}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_rows = 25_000
    if expected_rows == 25000 and csv_rows != expected_rows:
        raise ValueError(f"CSV row count {csv_rows} does not equal {expected_rows}")
    if int(metadata.get("completed_rows", expected_rows)) != expected_rows:
        raise ValueError("Metadata completed_rows does not equal 25000")
    if metadata.get("status") not in (None, "complete"):
        raise ValueError(f"Metadata status must be complete, got {metadata['status']}")
    return {
        "expected_rows": expected_rows,
        "csv_rows": csv_rows,
        "shape": [50, 500],
        "summary": "csv/npz equivalence verified",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gait_dir", type=Path, help="Directory containing raw gait outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gait_dir = args.gait_dir.expanduser().resolve()
    summary = validate_csv_and_npz_match(
        gait_dir / "robustness_raw.csv",
        gait_dir / "robustness_raw.npz",
        gait_dir / "metadata.json",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
