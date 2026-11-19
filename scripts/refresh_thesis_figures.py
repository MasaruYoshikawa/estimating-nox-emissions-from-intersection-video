"""Refresh thesis presentation from preserved, receipt-verified numerical results."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from scripts import thesis_ordering_figures as ordering_figures
from vehicle_emission.inference import recorded_hourly_extension, recorded_hourly_reanalysis
from vehicle_emission.measurement import thesis_diagnostics
from vehicle_emission.visualisation import comparisons, evidence, thesis_figures

ROOT = Path(__file__).resolve().parents[1]
EXTENSION_INPUTS = (
    "extension_hourly_panel.csv",
    "core_inference.csv",
    "residual_comparisons.csv",
    "class_reference_weights.csv",
    "count_comparisons.csv",
    "volatility_comparisons.csv",
)
MEASUREMENT_INPUTS = ("model_response_states.csv", "count_detail_pairs.csv")
ORDERING_INPUTS = ("model_ordering_panel.csv", "diurnal_profiles_scaled.csv")
PROFILE_OUTPUT = "fig_model_profile_response_checked.png"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def run(*, extension: Path, measurement: Path, ordering: Path, profile: Path, output: Path) -> None:
    """Render a fresh bundle without changing observations, estimates or source images."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("output directory must be fresh")
    for source in (extension, measurement, ordering):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError("output directory must be outside the source bundles")

    inputs = thesis_figures._verified_inputs(extension, EXTENSION_INPUTS)
    inputs.update(thesis_figures._verified_inputs(measurement, MEASUREMENT_INPUTS))
    inputs.update(thesis_figures._verified_inputs(ordering, ORDERING_INPUTS))
    inputs[profile] = _digest(profile)
    for path in (ROOT / "pyproject.toml", ROOT / "requirements-dev.lock"):
        inputs[path] = _digest(path)
    calculators = (
        Path(__file__),
        Path(recorded_hourly_extension.__file__),
        Path(recorded_hourly_reanalysis.__file__),
        Path(thesis_diagnostics.__file__),
        Path(ordering_figures.__file__),
        Path(comparisons.__file__),
        Path(evidence.__file__),
        Path(thesis_figures.__file__),
    )
    calculator_hashes = {str(path): _digest(path) for path in calculators}

    panel = _read(extension / "extension_hourly_panel.csv")
    core_tables = {
        name: _read(extension / name) for name in ("core_inference.csv", "residual_comparisons.csv")
    }
    counts = comparisons.count_plot_data(_read(extension / "count_comparisons.csv"))
    volatility = comparisons.volatility_plot_data(_read(extension / "volatility_comparisons.csv"))
    ranks, associations = evidence.rank_associations(panel)
    fleet = evidence.fleet_contributions(panel, _read(extension / "class_reference_weights.csv"))
    grid = _read(measurement / "model_response_states.csv")
    detail = _read(measurement / "count_detail_pairs.csv")
    ordering_panel = _read(ordering / "model_ordering_panel.csv")
    diurnal = _read(ordering / "diurnal_profiles_scaled.csv")

    output.mkdir(parents=True)
    for name, table in {
        "fleet_count_comparisons.csv": counts,
        "speed_variability_comparisons.csv": volatility,
        "hourly_rank_points.csv": ranks,
        "hourly_rank_summary.csv": associations,
        "fleet_weighting_contributions.csv": fleet,
    }.items():
        table.to_csv(output / name, index=False, float_format="%.17g")

    recorded_hourly_extension._figures(panel, core_tables, output)
    thesis_diagnostics._response_plot(grid, output / "fig_class_response_functions.png")
    ordering_figures.figures(ordering_panel, diurnal, output)
    comparisons.plot_count_comparisons(counts, output / "fig_fleet_count_advantages.png")
    comparisons.plot_volatility_comparisons(
        volatility, output / "fig_speed_variability_advantages.png"
    )
    evidence.plot_rank_associations(
        ranks, associations, output / "fig_hourly_rank_associations.png"
    )
    evidence.plot_count_discrepancies(detail, output / "fig_manual_count_discrepancies.png")
    evidence.plot_fleet_contributions(fleet, output / "fig_fleet_weighting_contributions.png")
    shutil.copyfile(profile, output / PROFILE_OUTPUT)

    if any(_digest(path) != expected for path, expected in inputs.items()):
        raise ValueError("source changed during rendering")
    if any(_digest(Path(path)) != expected for path, expected in calculator_hashes.items()):
        raise ValueError("plot code changed during rendering")
    receipt = {
        "kind": "thesis_presentation_refresh",
        "inputs": {str(path): digest for path, digest in inputs.items()},
        "calculator_sha256": calculator_hashes,
        "outputs": {path.name: _digest(path) for path in sorted(output.iterdir())},
        "float_parser": "round_trip",
        "uncertainty": "existing paired intervals; no inference or resampling executed",
        "preserved_images": {PROFILE_OUTPUT: str(profile)},
        "profile_scope": "existing constructed-profile image copied without model execution",
        "quartiles": "populated predictor quartiles; medians and IQRs are descriptive",
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Select the preserved inputs and a fresh output directory."""
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
    parser.add_argument(
        "--ordering-directory", type=Path, default=Path("results/model_ordering_20261109")
    )
    parser.add_argument(
        "--profile-image",
        type=Path,
        default=Path("results/thesis_figure_refresh_20261117/fig_model_profile_response_checked.png"),
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run(
        extension=args.extension_directory,
        measurement=args.measurement_directory,
        ordering=args.ordering_directory,
        profile=args.profile_image,
        output=args.output_directory,
    )


if __name__ == "__main__":
    main()
