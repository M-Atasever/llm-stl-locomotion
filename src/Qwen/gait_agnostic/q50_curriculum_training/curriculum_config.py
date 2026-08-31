"""Immutable schedule for the slow-increment full-range q50 curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


Q50_REWARD_SOURCE = "../q50_filtered_training/stl_reward.py"
CONDITION = "q50_slow_full_range_command_curriculum"


def resolve_output_root(requested_path: Path, invocation_directory: Path) -> Path:
    """Resolve a relative run root against the caller's working directory."""
    requested_path = requested_path.expanduser()
    return requested_path if requested_path.is_absolute() else invocation_directory / requested_path


@dataclass(frozen=True)
class TrainingStage:
    """One static PPO phase in the globally predeclared curriculum."""

    name: str
    num_timesteps: int
    vx_min: float
    vx_max: float
    num_evals: int
    restore_from_stage: str | None


FULL_TRAINING_STAGES = (
    TrainingStage("stage_1_expand_050", 50_000_000, 0.0, 0.5, 5, None),
    TrainingStage("stage_2_expand_075", 50_000_000, 0.0, 0.75, 5, "stage_1_expand_050"),
    TrainingStage("stage_3_expand_100", 50_000_000, 0.0, 1.0, 5, "stage_2_expand_075"),
    TrainingStage("stage_4_expand_120", 50_000_000, 0.0, 1.2, 5, "stage_3_expand_100"),
    TrainingStage("stage_5_expand_140", 50_000_000, 0.0, 1.4, 5, "stage_4_expand_120"),
    TrainingStage("stage_6_expand_160", 50_000_000, 0.0, 1.6, 5, "stage_5_expand_140"),
    TrainingStage("stage_7_expand_180", 50_000_000, 0.0, 1.8, 5, "stage_6_expand_160"),
    TrainingStage("stage_8_full_190", 50_000_000, 0.0, 1.9, 5, "stage_7_expand_180"),
)


def select_stage(name: str) -> TrainingStage:
    """Return one declared stage or fail closed for a misspelled selection."""
    for stage in FULL_TRAINING_STAGES:
        if stage.name == name:
            return stage
    raise ValueError(f"Unknown curriculum stage: {name}")
