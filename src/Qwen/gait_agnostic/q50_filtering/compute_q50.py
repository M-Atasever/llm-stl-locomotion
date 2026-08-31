#!/usr/bin/env python3
"""Gait-agnostic q50 analysis and filtering outputs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Iterable

import numpy as np


COMPONENTS = (
    "vel_track_x",
    "safe_height",
    "safe_orientation",
    "low_slip",
)
OUTPUT_FILENAMES = (
    "q50_results.json",
    "filtering_decisions.csv",
    "filtering_decisions.json",
    "analysis_report.md",
)
RULE_TEXT = "retain iff joint_q50 > 0.0; ties at 0.0 excluded"
RUN_ID_KEY = "run_id"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_validator_module(run_root: Path):
    validator_path = run_root / "collection" / "validate_outputs.py"
    spec = importlib.util.spec_from_file_location("gait_agnostic_validate_outputs", validator_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_metadata(metadata_path: Path) -> dict[str, object]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _validate_hash(actual_path: Path, expected_hash: str, label: str) -> str:
    actual_hash = sha256(actual_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{label} hash mismatch for {actual_path}: {actual_hash} != {expected_hash}"
        )
    return actual_hash


def _collect_component_values(csv_path: Path) -> dict[str, np.ndarray]:
    values = {component: [] for component in COMPONENTS}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for component in COMPONENTS:
                values[component].append(float(row[component]))
    arrays = {
        component: np.asarray(component_values, dtype=np.float64)
        for component, component_values in values.items()
    }
    row_counts = {component: int(array.size) for component, array in arrays.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"Inconsistent component row counts: {row_counts}")
    if next(iter(row_counts.values()), 0) != 25_000:
        raise ValueError(f"Expected 25000 rows per component, got {row_counts}")
    return arrays


def _quantiles(component_arrays: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        component: float(np.quantile(values, 0.5, method="linear"))
        for component, values in component_arrays.items()
    }


def validate_inputs(run_root: Path) -> dict[str, object]:
    run_root = run_root.expanduser().resolve()
    validator = _load_validator_module(run_root)
    validated = {
        RUN_ID_KEY: None,
        "validator_sha256": sha256(run_root / "collection" / "validate_outputs.py"),
        "datasets": {},
    }
    for gait in ("walk", "trot"):
        gait_dir = run_root / "collection" / "raw_data" / gait
        metadata_path = gait_dir / "metadata.json"
        csv_path = gait_dir / "robustness_raw.csv"
        npz_path = gait_dir / "robustness_raw.npz"
        metadata = _load_metadata(metadata_path)
        if validated[RUN_ID_KEY] is None:
            validated[RUN_ID_KEY] = metadata[RUN_ID_KEY]
        elif validated[RUN_ID_KEY] != metadata[RUN_ID_KEY]:
            raise ValueError("walk/trot metadata run_id mismatch")
        if metadata.get("gait") != gait:
            raise ValueError(f"Expected gait {gait}, got {metadata.get('gait')}")
        if metadata.get("individual_score_keys") != list(COMPONENTS):
            raise ValueError("Unexpected individual component order in metadata")
        if metadata.get("filtering_applied") is not False:
            raise ValueError(f"{gait} metadata must record filtering_applied=false")
        if metadata.get("q50_calculated") is not False:
            raise ValueError(f"{gait} metadata must record q50_calculated=false")

        validator_summary = validator.validate_csv_and_npz_match(
            csv_path,
            npz_path,
            metadata_path,
        )
        csv_hash = _validate_hash(csv_path, str(metadata["csv_sha256"]), f"{gait} csv")
        npz_hash = _validate_hash(npz_path, str(metadata["npz_sha256"]), f"{gait} npz")
        validated["datasets"][gait] = {
            "gait": gait,
            "metadata_sha256": sha256(metadata_path),
            "csv_sha256": csv_hash,
            "npz_sha256": npz_hash,
            "source_policy_tree_sha256": metadata["source_policy_tree_sha256"],
            "source_policy_path": metadata["source_policy_path"],
            "row_count": int(validator_summary["csv_rows"]),
            "trajectory_count": int(metadata["trajectory_count"]),
            "timesteps_per_trajectory": int(metadata["timesteps_per_trajectory"]),
            "validator_summary": validator_summary,
        }
    return validated


def decide_components(joint_q50: dict[str, float]) -> dict[str, list[str] | str]:
    retained = [component for component in COMPONENTS if joint_q50[component] > 0.0]
    excluded = [component for component in COMPONENTS if component not in retained]
    return {
        "retained_components": retained,
        "excluded_components": excluded,
        "rule": RULE_TEXT,
    }


def build_analysis(run_root: Path) -> dict[str, object]:
    validated = validate_inputs(run_root)
    per_gait = {}
    joint_arrays = {component: [] for component in COMPONENTS}
    for gait in ("walk", "trot"):
        csv_path = run_root / "collection" / "raw_data" / gait / "robustness_raw.csv"
        component_arrays = _collect_component_values(csv_path)
        for component, values in component_arrays.items():
            joint_arrays[component].append(values)
        per_gait[gait] = {
            "row_count": int(next(iter(component_arrays.values())).size),
            "component_order": list(COMPONENTS),
            "q50": _quantiles(component_arrays),
        }
    joint = {
        "row_count": sum(per_gait[gait]["row_count"] for gait in ("walk", "trot")),
        "q50": _quantiles(
            {
                component: np.concatenate(component_value_groups)
                for component, component_value_groups in joint_arrays.items()
            }
        ),
    }
    return {
        RUN_ID_KEY: validated[RUN_ID_KEY],
        "quantile": {
            "q": 0.5,
            "method": "linear",
            "numpy_version": np.__version__,
        },
        "source_validation": validated,
        "per_gait": per_gait,
        "joint": joint,
        "filtering_decisions": decide_components(joint["q50"]),
    }


def _stable_json(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _decision_rows(analysis: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    retained = set(analysis["filtering_decisions"]["retained_components"])
    for component in COMPONENTS:
        rows.append(
            {
                "component": component,
                "walk_q50": analysis["per_gait"]["walk"]["q50"][component],
                "trot_q50": analysis["per_gait"]["trot"]["q50"][component],
                "joint_q50": analysis["joint"]["q50"][component],
                "decision": "retain" if component in retained else "exclude",
                "rule": "joint_q50 > 0.0" if component in retained else "joint_q50 <= 0.0",
            }
        )
    return rows


def _render_report(analysis: dict[str, object], generated_at_utc: str) -> str:
    lines = [
        "# Gait-agnostic q50 Analysis",
        "",
        f"- Run ID: `{analysis[RUN_ID_KEY]}`",
        f"- Generated at: `{generated_at_utc}`",
        "- Quantile: NumPy `q=0.5`, `method=\"linear\"`",
        f"- Rule: `{analysis['filtering_decisions']['rule']}`",
        "",
        "## Decisions",
        "",
        f"- Retained: `{', '.join(analysis['filtering_decisions']['retained_components'])}`",
        f"- Excluded: `{', '.join(analysis['filtering_decisions']['excluded_components'])}`",
        "",
        "## q50 Summary",
        "",
        "| Component | Walk q50 | Trot q50 | Joint q50 | Decision |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    retained = set(analysis["filtering_decisions"]["retained_components"])
    for component in COMPONENTS:
        decision = "retain" if component in retained else "exclude"
        lines.append(
            "| {component} | {walk:.10f} | {trot:.10f} | {joint:.10f} | {decision} |".format(
                component=component,
                walk=analysis["per_gait"]["walk"]["q50"][component],
                trot=analysis["per_gait"]["trot"]["q50"][component],
                joint=analysis["joint"]["q50"][component],
                decision=decision,
            )
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Walk rows: `{analysis['per_gait']['walk']['row_count']}`",
            f"- Trot rows: `{analysis['per_gait']['trot']['row_count']}`",
            f"- Joint rows: `{analysis['joint']['row_count']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    analysis: dict[str, object],
    generated_at_utc: str | None = None,
) -> dict[str, str]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at_utc = generated_at_utc or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    q50_results = {
        RUN_ID_KEY: analysis[RUN_ID_KEY],
        "generated_at_utc": generated_at_utc,
        "quantile": analysis["quantile"],
        "per_gait": analysis["per_gait"],
        "joint": analysis["joint"],
    }
    decision_rows = _decision_rows(analysis)
    decisions_json = {
        RUN_ID_KEY: analysis[RUN_ID_KEY],
        "generated_at_utc": generated_at_utc,
        **analysis["filtering_decisions"],
        "rows": decision_rows,
    }
    report_text = _render_report(analysis, generated_at_utc)

    (output_dir / "q50_results.json").write_text(
        _stable_json(q50_results),
        encoding="utf-8",
    )
    with (output_dir / "filtering_decisions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("component", "walk_q50", "trot_q50", "joint_q50", "decision", "rule"),
        )
        writer.writeheader()
        writer.writerows(decision_rows)
    (output_dir / "filtering_decisions.json").write_text(
        _stable_json(decisions_json),
        encoding="utf-8",
    )
    (output_dir / "analysis_report.md").write_text(report_text, encoding="utf-8")

    output_hashes = {
        filename: sha256(output_dir / filename)
        for filename in OUTPUT_FILENAMES
    }
    manifest = {
        RUN_ID_KEY: analysis[RUN_ID_KEY],
        "generated_at_utc": generated_at_utc,
        "script_sha256": sha256(Path(__file__).resolve()),
        "quantile": {
            "numpy_q": 0.5,
            "method": "linear",
            "numpy_version": np.__version__,
        },
        "source_hashes": {
            gait: {
                "metadata_sha256": analysis["source_validation"]["datasets"][gait]["metadata_sha256"],
                "csv_sha256": analysis["source_validation"]["datasets"][gait]["csv_sha256"],
                "npz_sha256": analysis["source_validation"]["datasets"][gait]["npz_sha256"],
                "source_policy_tree_sha256": analysis["source_validation"]["datasets"][gait]["source_policy_tree_sha256"],
            }
            for gait in ("walk", "trot")
        },
        "row_counts": {
            "walk": analysis["per_gait"]["walk"]["row_count"],
            "trot": analysis["per_gait"]["trot"]["row_count"],
            "joint": analysis["joint"]["row_count"],
        },
        "validator_sha256": analysis["source_validation"]["validator_sha256"],
        "output_hashes": output_hashes,
    }
    (output_dir / "analysis_manifest.json").write_text(
        _stable_json(manifest),
        encoding="utf-8",
    )
    return output_hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="gait-agnostic root containing collection/raw_data and q50_filtering.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis output directory; defaults to run-root/runs/q50_filtering, leaving supplied selection JSON unchanged.",
    )
    parser.add_argument(
        "--generated-at-utc",
        default=None,
        help="Optional fixed UTC timestamp for deterministic output generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    analysis = build_analysis(run_root)
    output_dir = args.output_dir or run_root / "runs" / "q50_filtering"
    output_hashes = write_outputs(
        output_dir,
        analysis,
        generated_at_utc=args.generated_at_utc,
    )
    print(_stable_json({"analysis_manifest": "analysis_manifest.json", "output_hashes": output_hashes}))


if __name__ == "__main__":
    main()
