"""Render the five additional statistical figures from preserved results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from vehicle_emission.visualisation import comparisons, evidence


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_inputs(directory: Path, names: tuple[str, ...]) -> dict[Path, str]:
    receipt = directory / "receipt.json"
    recorded = json.loads(receipt.read_text(encoding="utf-8"))["outputs"]
    inputs = {receipt: _digest(receipt)}
    for name in names:
        path = directory / name
        actual = _digest(path)
        if recorded.get(name) != actual:
            raise ValueError(f"source receipt does not match {path}")
        inputs[path] = actual
    return inputs


def run(extension: Path, measurement: Path, output: Path) -> None:
    """Write a fresh figure bundle without changing source statistics."""
    if output.exists():
        raise FileExistsError("output directory must be fresh")
    inputs = _verified_inputs(
        extension,
        (
            "extension_hourly_panel.csv",
            "class_reference_weights.csv",
            "count_comparisons.csv",
            "volatility_comparisons.csv",
        ),
    )
    inputs.update(_verified_inputs(measurement, ("count_detail_pairs.csv",)))
    panel = pd.read_csv(extension / "extension_hourly_panel.csv", float_precision="round_trip")
    weights = pd.read_csv(extension / "class_reference_weights.csv", float_precision="round_trip")
    counts = comparisons.count_plot_data(
        pd.read_csv(extension / "count_comparisons.csv", float_precision="round_trip")
    )
    volatility = comparisons.volatility_plot_data(
        pd.read_csv(extension / "volatility_comparisons.csv", float_precision="round_trip")
    )
    ranks, association_summary = evidence.rank_associations(panel)
    fleet = evidence.fleet_contributions(panel, weights)
    detail = pd.read_csv(measurement / "count_detail_pairs.csv", float_precision="round_trip")
    output.mkdir(parents=True)
    tables = {
        "fleet_count_comparisons.csv": counts,
        "speed_variability_comparisons.csv": volatility,
        "hourly_rank_points.csv": ranks,
        "hourly_rank_summary.csv": association_summary,
        "fleet_weighting_contributions.csv": fleet,
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False, float_format="%.17g")
    comparisons.plot_count_comparisons(counts, output / "fig_fleet_count_advantages.png")
    comparisons.plot_volatility_comparisons(
        volatility, output / "fig_speed_variability_advantages.png"
    )
    evidence.plot_rank_associations(
        ranks, association_summary, output / "fig_hourly_rank_associations.png"
    )
    evidence.plot_count_discrepancies(detail, output / "fig_manual_count_discrepancies.png")
    evidence.plot_fleet_contributions(fleet, output / "fig_fleet_weighting_contributions.png")
    calculators = (Path(__file__), Path(comparisons.__file__), Path(evidence.__file__))
    receipt = {
        "kind": "derived_figure_bundle",
        "inputs": {str(path): digest for path, digest in inputs.items()},
        "calculator_sha256": {str(path): _digest(path) for path in calculators},
        "outputs": {path.name: _digest(path) for path in sorted(output.iterdir())},
        "float_parser": "round_trip",
        "uncertainty": "existing paired pointwise intervals; no new resampling",
        "rank_display": "average rank divided by matched site sample size",
        "fleet_shares": "ratios of summed recorded class counts and fixed-weighted counts",
        "count_discrepancy": "automated minus manual for every reference clip/category pair",
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Parse the three bundle directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension-directory",
        type=Path,
        default=Path("results/recorded_hourly_extension_20261109"),
    )
    parser.add_argument(
        "--measurement-directory",
        type=Path,
        default=Path("results/measurement_model_extension_20261109"),
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run(args.extension_directory, args.measurement_directory, args.output_directory)


if __name__ == "__main__":
    main()
