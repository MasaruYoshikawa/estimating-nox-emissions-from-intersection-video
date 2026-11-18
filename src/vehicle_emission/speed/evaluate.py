"""Reference-based measurement evaluation; historical speed summaries are not inputs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from vehicle_emission.measurement.evaluate import main as measurement_main
from vehicle_emission.measurement.speed import evaluate_speed_pairs


def evaluate_speed(
    *,
    calibration_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Reject the retired summary-copying API without touching historical outputs."""
    if not calibration_path.is_file():
        raise ValueError(f"calibration file not found: {calibration_path}")
    if not manifest_path.is_file():
        raise ValueError(f"manifest file not found: {manifest_path}")
    raise ValueError(
        "Pre-entered speed summary metrics cannot establish validation. Use "
        "evaluate_speed_pairs with independent reference observations, or the "
        "measurement.evaluate --config --output-directory CLI. No output was written."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Use the complete source-linked measurement configuration, not legacy summaries."""
    return measurement_main(argv)


__all__ = ["evaluate_speed", "evaluate_speed_pairs", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
