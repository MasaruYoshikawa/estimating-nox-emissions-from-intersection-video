"""Recalculate recorded-hour statistics from a prepared public numeric panel."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

from vehicle_emission.inference.recorded_hourly_extension import (
    CLASS_COLUMNS,
    COUNT_BASELINES,
    MODELS,
    REGIONAL,
    SHARES,
    SITES,
    STAGES,
    ExtensionConfig,
    _figures,
    descriptive_tables,
    infer_extensions,
    reference_class_weights,
)
from vehicle_emission.inference.recorded_hourly_reanalysis import (
    FAMILY,
    PREDICTORS,
    _controls,
    _csv,
    _hash,
    _json,
    _partial_rank_points,
)

STATUS = "prepared_panel_statistical_replay"
KEYS = ["site", "monitor_start_utc"]
FLAGS = ("entering_known", "heavy_known", "class_counts_known")
REQUIRED = {
    *KEYS,
    "monitor_end_utc",
    "local_calendar_date",
    "hour_jst",
    "weekday_jst",
    "ambient_nox_ppm",
    "captured_clip_count",
    *PREDICTORS,
    *COUNT_BASELINES,
    *CLASS_COLUMNS,
    *SHARES,
    *REGIONAL,
    *FLAGS,
    "between_clip_speed_cv",
    *(f"{model}_raw" for model in MODELS),
}


def _numeric(
    frame: pd.DataFrame, columns: list[str], *, nullable: bool = False, nonnegative: bool = True
) -> None:
    for name in columns:
        try:
            frame[name] = pd.to_numeric(frame[name], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} requires numeric values") from error
        values = frame[name].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if nonnegative:
            valid &= values >= 0
        if nullable:
            valid |= np.isnan(values)
        if not valid.all():
            raise ValueError(f"{name} contains invalid numeric values")


def validate_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Check the prepared input and derive masks from its present values."""
    if frame.columns.duplicated().any() or not REQUIRED.issubset(frame):
        raise ValueError(f"panel requires distinct columns: {sorted(REQUIRED)}")
    rows = frame.copy()
    if rows.empty or set(rows.site) != set(SITES):
        raise ValueError("panel must contain both supported sites")
    for name in ("monitor_start_utc", "monitor_end_utc"):
        if (
            not rows[name]
            .astype("string")
            .str.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
            .fillna(False)
            .all()
        ):
            raise ValueError("monitor intervals require canonical UTC timestamps")
    starts = pd.to_datetime(rows.monitor_start_utc, utc=True, errors="raise")
    ends = pd.to_datetime(rows.monitor_end_utc, utc=True, errors="raise")
    if (
        not starts.eq(starts.dt.floor("h")).all()
        or not (ends - starts).eq(pd.Timedelta(hours=1)).all()
    ):
        raise ValueError("monitor intervals require whole calendar hours")
    local = starts.dt.tz_convert("Asia/Tokyo")
    if (
        not rows.local_calendar_date.eq(local.dt.strftime("%Y-%m-%d")).all()
        or not rows.hour_jst.eq(local.dt.hour).all()
        or not rows.weekday_jst.eq(local.dt.dayofweek).all()
    ):
        raise ValueError("panel calendar fields disagree with UTC-to-JST conversion")
    if rows.duplicated(KEYS).any():
        raise ValueError("duplicate panel site-hour keys")
    for name in FLAGS:
        if not rows[name].map(lambda value: isinstance(value, bool | np.bool_)).all():
            raise ValueError(f"{name} requires explicit boolean values")
    _numeric(
        rows,
        ["ambient_nox_ppm", "captured_clip_count", *PREDICTORS, *(f"{m}_raw" for m in MODELS)],
    )
    _numeric(
        rows,
        [
            *COUNT_BASELINES,
            *CLASS_COLUMNS,
            *SHARES,
            "wind_speed",
            "setagaya_nox_ppm",
            "between_clip_speed_cv",
        ],
        nullable=True,
    )
    _numeric(rows, ["temperature"], nullable=True, nonnegative=False)
    for name in ("captured_clip_count", "vehicle_count", *CLASS_COLUMNS, *COUNT_BASELINES[:2]):
        values = rows[name].dropna()
        if not values.eq(np.floor(values)).all():
            raise ValueError(f"{name} requires integer counts")
    if not rows.captured_clip_count.gt(0).all():
        raise ValueError("every included hour requires captured clips")
    for flag, name in (
        ("entering_known", "captured_clip_entering_count"),
        ("heavy_known", "captured_clip_bus_heavy_count"),
    ):
        if not rows[flag].eq(rows[name].notna()).all():
            raise ValueError(f"{flag} disagrees with {name} availability")
    known = rows.class_counts_known
    if (
        not rows.loc[known, list(CLASS_COLUMNS)].notna().all().all()
        or not rows.loc[~known, list(CLASS_COLUMNS)].isna().all().all()
    ):
        raise ValueError("class-count availability disagrees with its source flag")
    total = rows[list(CLASS_COLUMNS)].sum(axis=1, min_count=len(CLASS_COLUMNS))
    if not total.loc[known].eq(rows.loc[known, "captured_clip_entering_count"]).all():
        raise ValueError("class totals disagree with entering counts")
    heavy = rows.entering_class_0 + rows.entering_class_2
    if not heavy.loc[known].eq(rows.loc[known, "captured_clip_bus_heavy_count"]).all():
        raise ValueError("heavy-class totals disagree with bus/heavy entering counts")
    weights = reference_class_weights().set_index("class_id").normalised_weight
    weighted = sum(rows[CLASS_COLUMNS[i]] * weights.loc[i] for i in range(4))
    if not np.allclose(
        weighted, rows.fixed_rate_weighted_entries, rtol=1e-12, atol=1e-12, equal_nan=True
    ):
        raise ValueError("fixed-rate weighted entries disagree with published class weights")
    for class_id, share in ((0, "bus_share"), (2, "heavy_truck_share"), (3, "light_truck_share")):
        expected = rows[CLASS_COLUMNS[class_id]] / total.where(total.gt(0))
        if not np.allclose(expected, rows[share], rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError(f"{share} disagrees with recorded class counts")
    rows["residual_complete"] = np.isfinite(rows[["vehicle_count", *SHARES, *REGIONAL]]).all(axis=1)
    return rows.sort_values(KEYS, kind="stable").reset_index(drop=True)


def load_panel(path: Path) -> pd.DataFrame:
    """Read stored floating values without rounding or duplicate-header repair."""
    with path.open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError("panel requires a nonempty, distinct CSV header")
    return validate_panel(pd.read_csv(path, float_precision="round_trip"))


def _calculator_paths() -> list[Path]:
    base = Path(__file__).resolve().parents[1]
    return [
        Path(__file__).resolve(),
        base / "inference" / "recorded_hourly_extension.py",
        base / "inference" / "recorded_hourly_reanalysis.py",
        base / "inference" / "paired_inference.py",
        base / "robustness" / "temporal.py",
        base / "vehicle_classes.py",
        base / "emissions" / "movestar.py",
        *[
            base / "emissions" / name
            for name in (
                "VehicleSrcCoeff.csv",
                "EmsRate_1.csv",
                "EmsRate_2.csv",
                "EmsRate_6.csv",
                "EmsRate_13.csv",
            )
        ],
    ]


def _compare_tables(directory: Path, expected: Path, names: list[str]) -> dict[str, str]:
    comparison = {}
    for name in names:
        reference = expected / name
        if not reference.is_file():
            continue
        actual = pd.read_csv(directory / name, float_precision="round_trip")
        original = pd.read_csv(reference, float_precision="round_trip")
        try:
            pd.testing.assert_frame_equal(
                actual, original, check_exact=False, rtol=1e-12, atol=1e-12
            )
        except AssertionError as error:
            raise ValueError(f"replayed table differs from expected bundle: {name}") from error
        comparison[name] = _hash(reference)
    if not comparison:
        raise ValueError("expected bundle contains no replayed statistical tables")
    return comparison


def replay_panel(
    *,
    panel: Path,
    output_directory: Path,
    config: ExtensionConfig | None = None,
    expected_bundle: Path | None = None,
    day_offsets: tuple[int, ...] = (-7, -1, 1, 7),
) -> dict[str, Any]:
    """Calculate a fresh, portable bundle without videos or the source database."""
    config = config or ExtensionConfig()
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (config.core_resamples, config.extension_resamples)
    ):
        raise ValueError("resample counts must be integers")
    config.validate()
    if (
        isinstance(config.seed, bool)
        or not isinstance(config.seed, int)
        or config.seed < 0
        or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in config.block_lengths)
    ):
        raise ValueError("require a non-negative integer seed and positive integer block lengths")
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output_directory}")
    if output_directory.resolve().is_relative_to(panel.resolve().parent):
        raise ValueError("output must be separate from the published input bundle")
    panel_hash = _hash(panel)
    rows = load_panel(panel)
    calendar_days = (
        pd.Timestamp(rows.local_calendar_date.max()) - pd.Timestamp(rows.local_calendar_date.min())
    ).days + 1
    if max(config.block_lengths) > calendar_days:
        raise ValueError("block length exceeds the complete calendar")
    root = Path(__file__).resolve().parents[3]
    paths = _calculator_paths()
    calculators = {str(path.relative_to(root)): _hash(path) for path in paths}
    parameters = {
        "status": STATUS,
        "input_sha256": {"extension_hourly_panel.csv": panel_hash},
        "input_layer": "prepared_recorded_hour_aggregates",
        "source_measurement_validation": "not_performed_by_statistical_replay",
        "core_interval_family": FAMILY,
        "core_family_size": 10,
        "extension_interval_family": "pointwise_descriptive",
        "residual_stages": list(STAGES),
        "residual_support": "derived_common_complete_case_all_stages",
        "calendar_start_date": str(rows.local_calendar_date.min()),
        "calendar_end_date": str(rows.local_calendar_date.max()),
        "full_union_calendar_days": calendar_days,
        "temporal_day_offsets": list(day_offsets),
        "floating_point_parser": "round_trip",
        "runtime": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        **asdict(config),
    }
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".panel-replay-", dir=output_directory.parent
    ) as temporary:
        stage = Path(temporary) / "calculation"
        stage.mkdir()
        _json(stage / "parameters.json", parameters)
        _csv(stage / "extension_hourly_panel.csv", rows)
        _csv(stage / "class_reference_weights.csv", reference_class_weights())
        tables = {**infer_extensions(rows, config), **descriptive_tables(rows)}
        controls = _controls(rows, day_offsets)
        controls["weekday_preserved_for_nonwrapped_pairs"] = controls.day_offset.mod(7).eq(0)
        nonwrapped = controls.retained_hours - controls.retained_wrapped_hours
        controls["retained_weekday_preserving_hours"] = nonwrapped * controls.day_offset.mod(7).eq(
            0
        ) + controls.retained_wrapped_hours * (controls.day_offset.abs() - calendar_days).mod(7).eq(
            0
        )
        tables["offset_control_contrasts.csv"] = controls
        tables["partial_rank_point_estimates.csv"] = _partial_rank_points(rows)
        for name, frame in tables.items():
            _csv(stage / name, frame)
        comparison = (
            _compare_tables(stage, expected_bundle, list(tables)) if expected_bundle else {}
        )
        _figures(rows, tables, stage)
        summary = {
            "status": STATUS,
            "sites": {
                site: {
                    "core_hours": len(group),
                    "core_days": int(group.local_calendar_date.nunique()),
                    "weighted_count_hours": int(group.fixed_rate_weighted_entries.notna().sum()),
                    "residual_common_hours": int(group.residual_complete.sum()),
                    "volatility_hours": int(group.between_clip_speed_cv.notna().sum()),
                    "volatility_median": float(group.between_clip_speed_cv.median()),
                }
                for site, group in rows.groupby("site", sort=True)
            },
        }
        _json(stage / "summary.json", summary)
        receipt = {
            **parameters,
            "calculator_sha256": calculators,
            "expected_table_sha256": comparison,
            "outputs": {path.name: _hash(path) for path in sorted(stage.iterdir())},
        }
        _json(stage / "receipt.json", receipt)
        if _hash(panel) != panel_hash or any(
            _hash(path) != calculators[str(path.relative_to(root))] for path in paths
        ):
            raise ValueError("an input or calculator changed during replay")
        lock = output_directory.parent / f".{output_directory.name}.publish.lock"
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            if output_directory.exists() or output_directory.is_symlink():
                raise FileExistsError(f"refusing to overwrite existing output: {output_directory}")
            stage.rename(output_directory)
        finally:
            os.close(descriptor)
            lock.unlink()
    return summary


def main() -> None:
    """Replay the fixed paired statistics from one prepared hourly panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--core-resamples", type=int, default=20000)
    parser.add_argument("--extension-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-bundle", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            replay_panel(
                panel=args.panel,
                output_directory=args.output_directory,
                config=ExtensionConfig(args.core_resamples, args.extension_resamples, args.seed),
                expected_bundle=args.expected_bundle,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
