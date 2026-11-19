"""Apply station-specific provider quality without treating numeric zero as missing."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from vehicle_emission.inference.recorded_hourly_extension import REGIONAL, SHARES
from vehicle_emission.inference.recorded_hourly_reanalysis import _csv, _hash, _json

QUALITY_COLUMNS = (
    "monitoring_place",
    "monitor_start_utc",
    "provider_station_id",
    "provider_value",
    "nox_ppm",
    "quality_status",
)


def parse_tokyo_hourly(text: str, station: str, place: str) -> pd.DataFrame:
    """Convert Tokyo's end-labelled JST NOx records to UTC interval starts."""
    raw = pd.read_csv(io.StringIO(text.lstrip("\ufeff").replace("\\r\\n", "\n")), dtype=str)
    required = {"STATIONID", "ITEMID", "YEAR", "MONTH", "DAY", "HOUR", "VALUE"}
    if not required.issubset(raw) or raw.empty:
        raise ValueError("provider hourly response lacks its documented fields")
    if not raw.STATIONID.eq(station).all() or not raw.ITEMID.eq("005").all():
        raise ValueError("provider station or pollutant differs from the requested NOx series")
    hour = pd.to_numeric(raw.HOUR, errors="raise")
    if not hour.between(1, 24).all() or not hour.eq(np.floor(hour)).all():
        raise ValueError("provider hour labels must be integers from 1 through 24")
    dates = pd.to_datetime(raw[["YEAR", "MONTH", "DAY"]].rename(columns=str.lower))
    starts = (dates + pd.to_timedelta(hour - 1, unit="h")).dt.tz_localize("Asia/Tokyo")
    values = raw.VALUE.fillna("").str.strip()
    missing = values.eq("欠測")
    uncollected = values.eq("")
    numeric = pd.to_numeric(values.mask(missing | uncollected), errors="raise") / 1000
    if not (numeric.isna() | (np.isfinite(numeric) & numeric.ge(0))).all():
        raise ValueError("provider NOx must be non-negative or explicitly missing")
    result = pd.DataFrame(
        {
            "monitoring_place": place,
            "monitor_start_utc": starts.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "provider_station_id": station,
            "provider_value": values,
            "nox_ppm": numeric,
            "quality_status": np.where(
                missing, "missing", np.where(uncollected, "uncollected", "valid")
            ),
        }
    )
    if result.duplicated(["monitoring_place", "monitor_start_utc"]).any():
        raise ValueError("duplicate provider station-hour")
    return result


def apply_monitor_quality(
    panel: pd.DataFrame, quality: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude flagged receptors and blank flagged background only where used."""
    keys = ["site", "monitor_start_utc"]
    if panel.empty or panel.duplicated(keys).any():
        raise ValueError("analysis panel requires distinct site-hour keys")
    if not set(QUALITY_COLUMNS).issubset(quality) or quality.empty:
        raise ValueError("quality evidence requires station, interval, value and status")
    qkeys = ["monitoring_place", "monitor_start_utc"]
    if quality[qkeys].isna().any().any() or quality.duplicated(qkeys).any():
        raise ValueError("quality evidence requires distinct nonmissing station-hour keys")
    if not quality.quality_status.isin(["valid", "missing", "uncollected"]).all():
        raise ValueError("unrecognised provider quality status")
    valid = quality.quality_status.eq("valid")
    values = pd.to_numeric(quality.nox_ppm, errors="raise")
    if not (np.isfinite(values[valid]) & values[valid].ge(0)).all() or values[~valid].notna().any():
        raise ValueError("provider numeric values disagree with their quality status")
    result = panel.copy()
    result["receptor_quality_status"] = "unreviewed"
    result["background_quality_status"] = "unreviewed"
    excluded = pd.Series(False, index=result.index)
    changes: list[dict[str, object]] = []
    lookup = quality.set_index(qkeys)
    for index, row in result.iterrows():
        for place, column, status_column in (
            (row.site, "ambient_nox_ppm", "receptor_quality_status"),
            ("Setagaya", "setagaya_nox_ppm", "background_quality_status"),
        ):
            key = (place, row.monitor_start_utc)
            if key not in lookup.index:
                continue
            evidence = lookup.loc[key]
            status = evidence.quality_status
            result.at[index, status_column] = status
            if status == "valid":
                if not np.isclose(row[column], evidence.nox_ppm, rtol=0, atol=1e-12):
                    raise ValueError(
                        f"provider numeric value differs: {place}/{row.monitor_start_utc}"
                    )
                continue
            action = "exclude_receptor_hour" if place == row.site else "set_background_missing"
            changes.append(
                {
                    "site": row.site,
                    "monitoring_place": place,
                    "monitor_start_utc": row.monitor_start_utc,
                    "column": column,
                    "original_nox_ppm": row[column],
                    "quality_status": status,
                    "action": action,
                    "provider_station_id": evidence.provider_station_id,
                }
            )
            if place == row.site:
                excluded.at[index] = True
            else:
                result.at[index, column] = np.nan
    result = result.loc[~excluded].copy()
    result["residual_complete"] = result[["vehicle_count", *SHARES, *REGIONAL]].notna().all(axis=1)
    columns = [
        "site",
        "monitoring_place",
        "monitor_start_utc",
        "column",
        "original_nox_ppm",
        "quality_status",
        "action",
        "provider_station_id",
    ]
    return result.reset_index(drop=True), pd.DataFrame(changes, columns=columns)


def prepare_quality_bundle(
    panel: Path, quality_paths: list[Path], output: Path
) -> dict[str, object]:
    """Write corrected inputs separately from source data and statistical outputs."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("quality output directory must be fresh")
    inputs = {"panel": panel, **{f"quality_{i}": path for i, path in enumerate(quality_paths)}}
    hashes = {name: _hash(path) for name, path in inputs.items()}
    quality = pd.concat(
        [
            pd.read_csv(path, dtype={"provider_station_id": str}, float_precision="round_trip")
            for path in quality_paths
        ],
        ignore_index=True,
    )
    original = pd.read_csv(panel, float_precision="round_trip")
    prepared, changes = apply_monitor_quality(original, quality)
    summary = {
        "status": "provider_quality_corrected_recorded_aggregates",
        "original_hours": original.groupby("site").size().to_dict(),
        "retained_hours": prepared.groupby("site").size().to_dict(),
        "residual_complete_hours": prepared.groupby("site").residual_complete.sum().to_dict(),
        "unreviewed_zero_receptor_hours": prepared.loc[
            prepared.ambient_nox_ppm.eq(0) & prepared.receptor_quality_status.eq("unreviewed")
        ]
        .groupby("site")
        .size()
        .to_dict(),
        "quality_actions": changes.action.value_counts().to_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".monitor-quality-", dir=output.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        _csv(stage / "extension_hourly_panel.csv", prepared)
        _csv(stage / "quality_changes.csv", changes)
        _csv(stage / "provider_quality.csv", quality)
        _json(stage / "summary.json", summary)
        _json(
            stage / "receipt.json",
            {
                "kind": "monitor_quality_preparation",
                "inputs": {
                    name: {"path": str(path), "sha256": hashes[name]}
                    for name, path in inputs.items()
                },
                "calculator_sha256": _hash(Path(__file__)),
                "outputs": {path.name: _hash(path) for path in sorted(stage.iterdir())},
                "numeric_zero_policy": "retain_unless_provider_marks_missing_or_uncollected",
            },
        )
        if any(_hash(path) != hashes[name] for name, path in inputs.items()):
            raise ValueError("quality input changed during preparation")
        stage.rename(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--quality", type=Path, action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_quality_bundle(args.panel, args.quality, args.output_directory), indent=2
        )
    )


if __name__ == "__main__":
    main()
