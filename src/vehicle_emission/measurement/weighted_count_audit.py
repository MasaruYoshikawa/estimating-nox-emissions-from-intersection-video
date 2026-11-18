"""Fixed-weight discrepancies for manually counted clip--class cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "Bus", 1: "Car", 2: "Heavy truck", 3: "Light truck"}
PROJECT_CLASSES = {0: "bus", 1: "car", 2: "heavy_truck", 3: "light_truck"}
DETAIL_NAME = "count_detail_pairs.csv"
WEIGHTS_NAME = "class_reference_weights.csv"


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")


def _integers(series: pd.Series, name: str, *, minimum: int = 0) -> pd.Series:
    if series.map(lambda value: isinstance(value, bool | np.bool_)).any():
        raise ValueError(f"{name} must contain integers, not booleans")
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite integers") from exc
    if not (
        np.isfinite(values).all()
        and (values >= minimum).all()
        and (values <= 2**53 - 1).all()
        and (values == np.floor(values)).all()
    ):
        raise ValueError(f"{name} must contain finite integers >= {minimum}")
    return pd.Series(values.astype(np.int64), index=series.index)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.gt(0)))


def _weighted_cells(detail: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        detail,
        {"site", "kind", "clip", "category", "manual_count", "automated_count"},
        "count pairs",
    )
    if detail.kind.isna().any() or not detail.kind.isin({"classification", "entering"}).all():
        raise ValueError("count kind must be classification or entering")
    cells = detail.loc[
        detail.kind.eq("classification"),
        ["site", "clip", "category", "manual_count", "automated_count"],
    ].copy()
    if cells.empty:
        raise ValueError("no classification count pairs")
    if not cells.site.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("each classification pair needs a site")
    cells["clip"] = _integers(cells["clip"], "clip", minimum=1)
    if not cells.category.isin(CLASS_NAMES.values()).all():
        raise ValueError("classification categories must use the four project classes")
    if cells.duplicated(["site", "clip", "category"]).any():
        raise ValueError("duplicate clip--class count pair")
    if not cells.groupby(["site", "clip"]).size().eq(4).all():
        raise ValueError("each clip needs all four classification categories, including zeros")
    for column in ("manual_count", "automated_count"):
        cells[column] = _integers(cells[column], column)

    _require_columns(weights, {"class_id", "project_class", "normalised_weight"}, "weights")
    reference = weights[["class_id", "project_class", "normalised_weight"]].copy()
    reference["class_id"] = _integers(reference.class_id, "class_id")
    if reference.class_id.duplicated().any() or set(reference.class_id) != set(CLASS_NAMES):
        raise ValueError("weights need exactly one row for each of the four class IDs")
    if not reference.project_class.eq(reference.class_id.map(PROJECT_CLASSES)).all():
        raise ValueError("weight class IDs and project names disagree")
    if reference.normalised_weight.map(lambda value: isinstance(value, bool | np.bool_)).any():
        raise ValueError("weights must be finite positive numbers")
    try:
        reference["normalised_weight"] = pd.to_numeric(
            reference.normalised_weight, errors="raise"
        ).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be finite positive numbers") from exc
    if not (
        np.isfinite(reference.normalised_weight).all() and reference.normalised_weight.gt(0).all()
    ):
        raise ValueError("weights must be finite positive numbers")
    reference["category"] = reference.class_id.map(CLASS_NAMES)
    cells = cells.merge(reference, on="category", how="left", validate="many_to_one")
    cells["signed_difference"] = cells.automated_count - cells.manual_count
    cells["absolute_difference"] = cells.signed_difference.abs()
    for count in ("manual_count", "automated_count", "signed_difference", "absolute_difference"):
        cells[f"weighted_{count}"] = cells.normalised_weight * cells[count]
    if not np.isfinite(cells.filter(like="weighted_").to_numpy()).all():
        raise ValueError("weighted counts must remain finite")
    return cells.sort_values(["site", "clip", "class_id"]).reset_index(drop=True)


def weighted_count_audit(detail: pd.DataFrame, weights: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Keep absolute cell error separate from class/clip/site net cancellation.

    The denominator is the matching fixed-weight manual total. A zero total
    leaves the relative discrepancy unavailable, even when the numerator is zero.
    """
    cells = _weighted_cells(detail, weights)
    clips = cells.groupby(["site", "clip"], as_index=False, sort=True).agg(
        manual_count=("manual_count", "sum"),
        automated_count=("automated_count", "sum"),
        absolute_cell_difference=("absolute_difference", "sum"),
        weighted_manual_count=("weighted_manual_count", "sum"),
        weighted_automated_count=("weighted_automated_count", "sum"),
        weighted_signed_difference=("weighted_signed_difference", "sum"),
        weighted_absolute_cell_difference=("weighted_absolute_difference", "sum"),
        class_cells=("class_id", "size"),
    )
    clips["absolute_weighted_clip_net_difference"] = clips.weighted_signed_difference.abs()
    clips["relative_weighted_absolute_cell_difference"] = _ratio(
        clips.weighted_absolute_cell_difference, clips.weighted_manual_count
    )
    clips["relative_unweighted_absolute_cell_difference"] = _ratio(
        clips.absolute_cell_difference, clips.manual_count
    )
    sites = clips.groupby("site", as_index=False, sort=True).agg(
        clips=("clip", "size"),
        class_cells=("class_cells", "sum"),
        manual_count=("manual_count", "sum"),
        automated_count=("automated_count", "sum"),
        absolute_cell_difference=("absolute_cell_difference", "sum"),
        weighted_manual_count=("weighted_manual_count", "sum"),
        weighted_automated_count=("weighted_automated_count", "sum"),
        weighted_signed_difference=("weighted_signed_difference", "sum"),
        weighted_absolute_cell_difference=("weighted_absolute_cell_difference", "sum"),
        sum_absolute_weighted_clip_net_difference=("absolute_weighted_clip_net_difference", "sum"),
    )
    sites["absolute_weighted_site_net_difference"] = sites.weighted_signed_difference.abs()
    sites["relative_weighted_absolute_cell_difference"] = _ratio(
        sites.weighted_absolute_cell_difference, sites.weighted_manual_count
    )
    sites["relative_unweighted_absolute_cell_difference"] = _ratio(
        sites.absolute_cell_difference, sites.manual_count
    )
    classes = cells.groupby(
        ["site", "class_id", "category", "project_class", "normalised_weight"],
        as_index=False,
        sort=True,
    ).agg(
        clips=("clip", "size"),
        manual_count=("manual_count", "sum"),
        automated_count=("automated_count", "sum"),
        absolute_cell_difference=("absolute_difference", "sum"),
        weighted_manual_count=("weighted_manual_count", "sum"),
        weighted_automated_count=("weighted_automated_count", "sum"),
        weighted_signed_difference=("weighted_signed_difference", "sum"),
        weighted_absolute_cell_difference=("weighted_absolute_difference", "sum"),
    )
    classes["absolute_weighted_class_net_difference"] = classes.weighted_signed_difference.abs()
    classes["relative_weighted_absolute_cell_difference"] = _ratio(
        classes.weighted_absolute_cell_difference, classes.weighted_manual_count
    )
    denominators = sites.set_index("site")
    classes["share_of_site_weighted_manual_count"] = _ratio(
        classes.weighted_manual_count,
        classes.site.map(denominators.weighted_manual_count),
    )
    classes["share_of_site_weighted_absolute_cell_difference"] = _ratio(
        classes.weighted_absolute_cell_difference,
        classes.site.map(denominators.weighted_absolute_cell_difference),
    )
    for frame in (clips, sites, classes):
        numeric = frame.select_dtypes(include="number")
        if np.isinf(numeric.to_numpy()).any():
            raise ValueError("aggregate weighted counts must remain finite")
    return {
        "weighted_count_cells.csv": cells,
        "weighted_count_clips.csv": clips,
        "weighted_count_classes.csv": classes,
        "weighted_count_sites.csv": sites,
    }


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_input(directory: Path, name: str) -> dict[Path, str]:
    receipt = directory / "receipt.json"
    receipt_digest = _digest(receipt)
    recorded = json.loads(receipt.read_text())
    if not isinstance(recorded, dict) or not isinstance(recorded.get("outputs"), dict):
        raise ValueError(f"invalid source receipt in {directory}")
    source = directory / name
    digest = _digest(source)
    if recorded["outputs"].get(name) != digest:
        raise ValueError(f"source receipt hash mismatch for {source}")
    return {receipt: receipt_digest, source: digest}


def run(extension_directory: Path, measurement_directory: Path, output_directory: Path) -> None:
    """Read receipt-bound pairs and weights and write one fresh audit bundle."""
    if output_directory.exists() or output_directory.is_symlink():
        raise ValueError("output directory must be fresh")
    output_resolved = output_directory.resolve()
    if any(
        output_resolved.is_relative_to(source.resolve())
        for source in (extension_directory, measurement_directory)
    ):
        raise ValueError("output directory must be outside source bundles")
    inputs = {
        **_verified_input(measurement_directory, DETAIL_NAME),
        **_verified_input(extension_directory, WEIGHTS_NAME),
    }
    calculator = Path(__file__).resolve()
    calculator_digest = _digest(calculator)
    tables = weighted_count_audit(
        pd.read_csv(measurement_directory / DETAIL_NAME, float_precision="round_trip"),
        pd.read_csv(extension_directory / WEIGHTS_NAME, float_precision="round_trip"),
    )
    output_directory.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(output_directory / name, index=False, float_format="%.17g")
    if any(_digest(path) != expected for path, expected in inputs.items()):
        raise ValueError("source changed during audit; no receipt was written")
    if _digest(calculator) != calculator_digest:
        raise ValueError("calculator changed during audit; no receipt was written")
    receipt = {
        "status": "fixed_weight_manual_class_count_audit",
        "inputs": {str(path): digest for path, digest in sorted(inputs.items())},
        "calculator_sha256": {calculator.name: calculator_digest},
        "outputs": {name: _digest(output_directory / name) for name in tables},
        "observation_unit": "clip--class count pair, not joint class-specific entering events",
        "weighted_absolute_cell_difference": "sum(weight * abs(automated - manual))",
        "absolute_weighted_clip_net_difference": "abs(sum_class(weight * (automated - manual)))",
        "relative_denominator": "sum(weight * manual) for the same group",
        "zero_denominator": "relative discrepancy unavailable (empty CSV cell)",
        "sampling": "descriptive audit of the available clips; no sampling interval",
    }
    (output_directory / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
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
