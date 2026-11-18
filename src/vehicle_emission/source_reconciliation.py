"""Verify a raw source bundle and describe recorded metadata, not valid coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from vehicle_emission.asset_lineage import SITES
from vehicle_emission.inference.clip_coverage import _union
from vehicle_emission.source_export import AUDIT_COLUMNS, TABLES, _count_audit
from vehicle_emission.time_utils import (
    SORAMAME_HOUR_CONVENTIONS,
    SoramameHourConvention,
    local_timestamp_to_utc,
    parse_aware_timestamp,
    soramame_hour_interval_utc,
)

VIDEO_COLUMNS = [
    "source_site",
    "video_id",
    "video_path",
    "recording_datetime_utc",
    "duration_seconds",
    "source_record_role",
    "stored_media_end_utc",
    "stored_interval_status",
    "stored_interval_crosses_hour",
    "emission_methods",
    "movestar_record_status",
    "movestar_nox_grams",
    "movestar_nox_hybrid_adjusted_grams",
    "movestar_track_count",
    "count_audit_status",
    "recorded_entering_count",
    "recorded_directions",
]
HOUR_COLUMNS = [
    "source_site",
    "hour_start_utc",
    "hour_end_utc",
    "stored_media_clip_count",
    "stored_media_summed_seconds",
    "stored_media_union_seconds",
    "stored_media_overlap_seconds",
    "stored_media_has_internal_gap",
    "video_ids",
]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path, headers: list[str]) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != headers or len(set(headers)) != len(headers):
            raise ValueError(f"{path.name}: CSV header differs from manifest")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name}: malformed CSV record")
            yield row


def _number(value: str) -> int | float | str | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if math.isfinite(number) and number.is_integer() else number


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _utc(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _interval(row: dict[str, str]) -> tuple[str, datetime | None, datetime | None]:
    try:
        start = parse_aware_timestamp(row.get("recording_datetime_utc", "")).astimezone(UTC)
    except ValueError:
        return "missing_or_invalid_aware_start", None, None
    try:
        duration = float(row.get("duration_seconds", ""))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        end = start + timedelta(seconds=duration)
        if end <= start:
            raise ValueError("duration is below timestamp resolution")
    except (ValueError, OverflowError):
        return "missing_or_invalid_positive_duration", None, None
    return "stored_metadata_only", start, end


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _reference_value(value: str | None, *, text: bool = False) -> float | str | None:
    value = value.strip() if value is not None else ""
    if value in {"", "-", "—", "NaN", "null"}:
        return None
    return value if text else float(value)


def _compare_reference(
    expected: list[dict[str, Any]], stored: list[dict[str, str]], fields: list[str]
) -> dict[str, Any]:
    wanted = {row["timestamp_utc"]: row for row in expected}
    found = {row["timestamp_utc"]: row for row in stored}
    common = wanted.keys() & found.keys()
    mismatch: Counter[str] = Counter()
    missing_columns = sorted(set(fields) - set(stored[0])) if stored else fields
    for timestamp in sorted(common):
        for field in fields:
            observed = _reference_value(found[timestamp].get(field), text=field == "wind_direction")
            if wanted[timestamp][field] != observed:
                mismatch[field] += 1
    differences = {
        "missing_stored_rows": len(wanted.keys() - found.keys()),
        "extra_stored_rows": len(found.keys() - wanted.keys()),
        "duplicate_original_keys": len(expected) - len(wanted),
        "duplicate_stored_keys": len(stored) - len(found),
    }
    return {
        "status": (
            "matches_original_rows"
            if not (any(differences.values()) or mismatch or missing_columns)
            else "mismatch"
        ),
        "original_rows": len(expected),
        "stored_rows": len(stored),
        "compared_rows": len(common),
        "compared_cells": len(common) * len(fields),
        "mismatched_cells_by_field": dict(mismatch),
        "missing_export_columns": missing_columns,
        **differences,
    }


def _environment_check(
    *,
    monitor_sources: Mapping[str, Path],
    weather_source: Path | None,
    hour_convention: SoramameHourConvention | None,
    monitor_rows: list[dict[str, str]],
    weather_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[Path, str]]:
    if monitor_sources and hour_convention not in SORAMAME_HOUR_CONVENTIONS:
        raise ValueError("original monitor comparison requires an explicit hour convention")
    references: dict[Path, str] = {}
    comparisons: dict[str, Any] = {}
    monitor_fields = {
        "NO(ppm)": "no_ppm",
        "NO2(ppm)": "no2_ppm",
        "NOx(ppm)": "nox_ppm",
        "CO(ppm)": "co_ppm",
        "Ox(ppm)": "ox_ppm",
    }
    for label, path in sorted(monitor_sources.items()):
        assert hour_convention is not None
        references[path] = _hash(path)
        expected = []
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"Date", "Time", *monitor_fields} <= set(reader.fieldnames or []):
                raise ValueError(f"{path.name}: missing original monitor columns")
            for row in reader:
                start, _ = soramame_hour_interval_utc(
                    row["Date"], int(row["Time"]), convention=hour_convention
                )
                expected.append(
                    {
                        "timestamp_utc": start,
                        **{
                            field: _reference_value(row[source_field])
                            for source_field, field in monitor_fields.items()
                        },
                    }
                )
        comparisons[label] = _compare_reference(
            expected,
            [row for row in monitor_rows if row["monitoring_place"] == label],
            list(monitor_fields.values()),
        )
    weather_comparison: dict[str, Any] = {"status": "not_requested"}
    if weather_source is not None:
        references[weather_source] = _hash(weather_source)
        weather_fields = {
            "Tempareture": "temperature",
            "precipitation": "precipitation",
            "wind speed": "wind_speed",
            "wind direction": "wind_direction",
            "cloud thinckness": "cloud_thickness",
        }
        expected = []
        with weather_source.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"Date", "Time", *weather_fields} <= set(reader.fieldnames or []):
                raise ValueError(f"{weather_source.name}: missing original weather columns")
            for row in reader:
                timestamp = local_timestamp_to_utc(
                    f"{row['Date']} {row['Time']}",
                    input_format="%d/%m/%Y %H:%M:%S",
                    timezone_name="Asia/Tokyo",
                )
                expected.append(
                    {
                        "timestamp_utc": timestamp,
                        **{
                            field: _reference_value(
                                row[source_field], text=field == "wind_direction"
                            )
                            for source_field, field in weather_fields.items()
                        },
                    }
                )
        weather_comparison = _compare_reference(
            expected, weather_rows, list(weather_fields.values())
        )
    return {
        "monitor_hour_convention": hour_convention,
        "monitor_comparisons": comparisons,
        "weather_comparison": weather_comparison,
        "timestamp_mapping": "Monitor convention is caller-supplied and compared with stored UTC interval starts. Weather source JST labels map to stored UTC observation instants; no wind/temperature hour support is inferred.",
        "precipitation_support": "For previous-hour JMA precipitation, video/monitor interval [t,t+1h) aligns with weather label t+1h, not weather label t. This comparison does not select final eligibility.",
    }, references


def reconcile_source(
    *,
    source_directory: Path,
    output_directory: Path,
    monitor_sources: Mapping[str, Path] | None = None,
    weather_source: Path | None = None,
    hour_convention: SoramameHourConvention | None = None,
) -> dict[str, Any]:
    """Verify copied rows and calculate descriptive stored-media interval unions.

    Unknown timing, source identities and measurement validity stay unknown.
    Clip emission totals are never spread into seconds or split across hours.
    """
    source = source_directory.resolve(strict=True)
    destination = output_directory.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("output must be outside the source bundle")
    if output_directory.is_symlink() or destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_directory}")
    manifest_path = source / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("source manifest must not be a symbolic link")
    manifest_hash = _hash(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = {f"{table}.csv" for table in TABLES} | {"entering_count_audit.csv"}
    if (
        manifest.get("schema_version") != "raw_source_export_v1"
        or manifest.get("status") != "raw_source_export_not_analysis_ready"
        or set(manifest.get("outputs", {})) != names
        or set(manifest.get("tables", {})) != set(TABLES)
    ):
        raise ValueError("unsupported or incomplete raw source manifest")
    sites = manifest.get("sites", [])
    if not sites or len(sites) != len(set(sites)) or not set(sites) <= SITES:
        raise ValueError("source manifest has invalid site identities")
    source_hashes = {"manifest.json": manifest_hash}
    for name, expected in manifest["outputs"].items():
        path = source / name
        if path.is_symlink() or not path.is_file() or _hash(path) != expected:
            raise ValueError(f"{name}: missing file or SHA-256 mismatch")
        source_hashes[name] = expected

    videos: dict[str, dict[str, str]] = {}
    emissions: dict[str, list[dict[str, str]]] = defaultdict(list)
    counts: list[dict[str, Any]] = []
    monitor_summary: dict[str, Counter[str]] = defaultdict(Counter)
    monitor_rows: list[dict[str, str]] = []
    weather_rows: list[dict[str, str]] = []
    weather_summary: Counter[str] = Counter()
    row_counts: dict[str, int] = {}
    site_table_rows: dict[str, Counter[str]] = defaultdict(Counter)
    for table in TABLES:
        metadata = manifest["tables"][table]
        columns = metadata["columns"] or list(TABLES[table])
        per_video = table in {
            "video_master",
            "video_emissions",
            "entering_counts",
            "crossing_events",
        }
        headers = (["source_site"] if per_video else []) + columns
        if metadata["available"] and not set(TABLES[table]) <= set(columns):
            raise ValueError(f"{table}: missing required columns")
        count = 0
        for row in _rows(source / f"{table}.csv", headers):
            count += 1
            if per_video:
                site, video_id = row["source_site"], row["video_id"]
                if site not in sites or not video_id:
                    raise ValueError(f"{table}: invalid source site/video reference")
                try:
                    if str(int(video_id)) != video_id:
                        raise ValueError("noncanonical ID")
                except ValueError as exc:
                    raise ValueError(f"{table}: invalid integer video identity") from exc
                if table == "video_master":
                    if video_id in videos or row["intersection_id"] != site:
                        raise ValueError("video_master: duplicate video ID or site mismatch")
                    videos[video_id] = row
                elif video_id not in videos or videos[video_id]["source_site"] != site:
                    raise ValueError(f"{table}: orphan or site-mismatched video reference")
                site_table_rows[site][table] += 1
            if table == "video_emissions":
                emissions[row["video_id"]].append(row)
            elif table == "entering_counts":
                counts.append(
                    {
                        **row,
                        "video_id": int(row["video_id"]),
                        "class_id": _number(row["class_id"]),
                        "vehicle_count": _number(row["vehicle_count"]),
                    }
                )
            elif table == "air_pollution_history":
                monitor_rows.append(row)
                monitor_summary[row["monitoring_place"]]["rows"] += 1
                if not row.get("nox_ppm", ""):
                    monitor_summary[row["monitoring_place"]]["missing_nox_rows"] += 1
            elif table == "weather_history":
                weather_rows.append(row)
                weather_summary["rows"] += 1
                for field in ("precipitation", "wind_speed", "wind_direction"):
                    if not row.get(field, ""):
                        weather_summary[f"missing_{field}_rows"] += 1
        if count != metadata["rows"] or (not metadata["available"] and count):
            raise ValueError(f"{table}: row count differs from manifest")
        row_counts[table] = count

    replay = _count_audit(
        [(row["source_site"], int(row["video_id"])) for row in videos.values()], counts
    )
    expected_audit = [{key: _text(row[key]) for key in AUDIT_COLUMNS} for row in replay]
    supplied_audit = list(_rows(source / "entering_count_audit.csv", AUDIT_COLUMNS))
    if expected_audit != supplied_audit:
        raise ValueError("entering_count_audit.csv differs from raw-count replay")
    environment, references = _environment_check(
        monitor_sources=monitor_sources or {},
        weather_source=weather_source,
        hour_convention=hour_convention,
        monitor_rows=monitor_rows,
        weather_rows=weather_rows,
    )
    audited: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in replay:
        audited[str(row["video_id"])].append(row)

    video_rows: list[dict[str, Any]] = []
    intervals: dict[tuple[str, datetime], list[tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(
        list
    )
    hour_videos: dict[tuple[str, datetime], list[str]] = defaultdict(list)
    interval_statuses: dict[str, Counter[str]] = defaultdict(Counter)
    count_statuses: dict[str, Counter[str]] = defaultdict(Counter)
    for video_id, row in videos.items():
        site = row["source_site"]
        status, start, end = _interval(row)
        interval_statuses[site][status] += 1
        record: dict[str, Any] = {key: row.get(key, "") for key in VIDEO_COLUMNS[:5]}
        name = row.get("video_name", "")
        count_test = bool(
            re.search(r"/counting_test\d+/", row.get("video_path", "").replace("\\", "/"))
        )
        record["source_record_role"] = (
            "count_test"
            if count_test
            else (
                "dated_filename"
                if re.fullmatch(r"data_\d{8}_\d{4}_\d+", name)
                else "clock_only_filename" if re.fullmatch(r"data_\d{4}_\d+", name) else "other"
            )
        )
        record.update(
            stored_media_end_utc=_utc(end) if end else "",
            stored_interval_status=status,
            stored_interval_crosses_hour="",
        )
        if start is not None and end is not None:
            hour = start.replace(minute=0, second=0, microsecond=0)
            record["stored_interval_crosses_hour"] = end > hour + timedelta(hours=1)
            while hour < end:
                stop = hour + timedelta(hours=1)
                intervals[(site, hour)].append(
                    (pd.Timestamp(max(hour, start)), pd.Timestamp(min(stop, end)))
                )
                hour_videos[(site, hour)].append(video_id)
                hour = stop
        methods = emissions.get(video_id, [])
        primary = [item for item in methods if item["estimation_method"] == "MOVEStar"]
        record.update(
            emission_methods=json.dumps(sorted({item["estimation_method"] for item in methods})),
            movestar_record_status=(
                "one_record"
                if len(primary) == 1
                else "missing" if not primary else "duplicate_records"
            ),
        )
        for field, source_field in (
            ("movestar_nox_grams", "nox_grams"),
            ("movestar_nox_hybrid_adjusted_grams", "nox_hybrid_adjusted_grams"),
            ("movestar_track_count", "vehicle_count"),
        ):
            record[field] = primary[0].get(source_field, "") if len(primary) == 1 else ""
        audit = audited[video_id]
        complete = all(item["recorded_entering_count"] is not None for item in audit)
        count_status = "arithmetic_consistent" if complete else "unavailable_or_inconsistent"
        count_statuses[site][count_status] += 1
        record.update(
            count_audit_status=count_status,
            recorded_entering_count=(
                sum(item["recorded_entering_count"] for item in audit) if complete else ""
            ),
            recorded_directions=json.dumps(sorted(item["direction"] for item in audit)),
        )
        video_rows.append(record)

    hour_rows: list[dict[str, Any]] = []
    for (site, hour), pieces in sorted(intervals.items()):
        merged = _union(pieces)
        summed = sum((end - start).total_seconds() for start, end in pieces)
        union = sum((end - start).total_seconds() for start, end in merged)
        hour_rows.append(
            dict(
                zip(
                    HOUR_COLUMNS,
                    (
                        site,
                        _utc(hour),
                        _utc(hour + timedelta(hours=1)),
                        len(pieces),
                        summed,
                        union,
                        max(0.0, summed - union),
                        len(merged) > 1,
                        json.dumps(sorted(hour_videos[(site, hour)])),
                    ),
                    strict=True,
                )
            )
        )
    summary = {
        "status": "raw_source_reconciliation_not_analysis_ready",
        "sites": {
            site: {
                "table_rows": dict(site_table_rows[site]),
                "stored_interval_statuses": dict(interval_statuses[site]),
                "source_record_roles": dict(
                    Counter(
                        row["source_record_role"]
                        for row in video_rows
                        if row["source_site"] == site
                    )
                ),
                "count_audit_video_statuses": dict(count_statuses[site]),
                "count_audit_direction_statuses": dict(
                    Counter(row["count_status"] for row in replay if row["source_site"] == site)
                ),
                "stored_media_touched_hours": sum(row["source_site"] == site for row in hour_rows),
                "stored_media_union_seconds": sum(
                    row["stored_media_union_seconds"]
                    for row in hour_rows
                    if row["source_site"] == site
                ),
            }
            for site in sorted(sites)
        },
        "monitor_stored_labels": {
            name: dict(values) for name, values in sorted(monitor_summary.items())
        },
        "weather_stored_rows": dict(weather_summary),
        "original_environment_comparison": environment,
        "interpretation": {
            "coverage": "Stored timestamp plus media duration only; not valid wall-clock recording coverage or hourly eligibility.",
            "counts": "Replayed arithmetic only; direction sums do not establish unique independently validated entering vehicles.",
            "emissions": "MOVEStar nox_grams is unadjusted; nox_hybrid_adjusted_grams is separate. Track count is not entering count. Clip totals are not split across hours.",
            "timing": "Recorded aware timestamp is used as stored, not independently verified. No target/weather shift or interval convention is assigned.",
            "provenance": "No station-role, checkpoint, calibration-production-use or measurement-accuracy acceptance is assigned.",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".source-reconciliation-", dir=destination.parent
    ) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        _write_csv(stage / "video_reconciliation.csv", VIDEO_COLUMNS, video_rows)
        _write_csv(stage / "stored_media_hourly.csv", HOUR_COLUMNS, hour_rows)
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verification = {
            "schema_version": "raw_source_reconciliation_v1",
            "status": summary["status"],
            "source_hashes": source_hashes,
            "original_environment_sources": [
                {"path": str(path), "sha256": digest} for path, digest in sorted(references.items())
            ],
            "reconciler_sha256": _hash(Path(__file__)),
            "exporter_sha256": manifest.get("exporter_sha256"),
            "exporter_matches_current_code": manifest.get("exporter_sha256")
            == _hash(Path(__file__).with_name("source_export.py")),
            "database": manifest.get("database"),
            "source_tables": manifest["tables"],
            "verified_table_rows": row_counts,
            "count_audit_replayed": True,
            "verified_count_audit_rows": len(replay),
            "outputs": {path.name: _hash(path) for path in sorted(stage.iterdir())},
        }
        (stage / "verification.json").write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if any(_hash(source / name) != expected for name, expected in source_hashes.items()):
            raise ValueError("source bundle changed during reconciliation; no output published")
        if any(_hash(path) != expected for path, expected in references.items()):
            raise ValueError(
                "original environmental source changed during reconciliation; no output published"
            )
        lock = destination.parent / f".{destination.name}.publish.lock"
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"refusing to overwrite existing output: {output_directory}")
            stage.rename(destination)
        finally:
            os.close(descriptor)
            lock.unlink()
    return verification


def main(argv: Sequence[str] | None = None) -> int:
    """Verify a received raw export and publish a fresh descriptive audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--monitor-source", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--weather-source", type=Path)
    parser.add_argument("--soramame-hour-convention", choices=SORAMAME_HOUR_CONVENTIONS)
    args = parser.parse_args(argv)
    monitor_sources = {}
    for item in args.monitor_source:
        label, separator, path = item.partition("=")
        if not separator or not label or not path or label in monitor_sources:
            parser.error("--monitor-source must specify a distinct nonempty LABEL=PATH")
        monitor_sources[label] = Path(path)
    verification = reconcile_source(
        source_directory=args.source_directory,
        output_directory=args.output_directory,
        monitor_sources=monitor_sources,
        weather_source=args.weather_source,
        hour_convention=args.soramame_hour_convention,
    )
    print(
        json.dumps(
            {"status": verification["status"], "tables": verification["verified_table_rows"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
