"""Read-only, name-level reconciliation of video assets and processing records.

This reports candidate links. A matching filename is not proof that the same
video bytes, detector checkpoint, or recording time produced a database row.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

SITES = frozenset({"Sukiyabashi", "Shinjuku"})
DATED_NAME = re.compile(r"^data_(\d{8})_(\d{4})_[^/\\]+\.mp4$", re.IGNORECASE)
CLOCK_NAME = re.compile(r"^data_(\d{4})_[^/\\]+\.mp4$", re.IGNORECASE)
SEQUENCE_NAME = re.compile(r"^data_\d+\.mp4$", re.IGNORECASE)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _records(document: object, *, site: str, field: str) -> list[dict[str, Any]]:
    root = _object(document, "inventory")
    if root.get("schema_version") != 1 or root.get("site") != site:
        raise ValueError(f"inventory must have schema_version=1 and site={site!r}")
    values = root.get(field)
    if not isinstance(values, list):
        raise ValueError(f"inventory {field!r} must be a list")
    return [_object(value, f"{field}[{index}]") for index, value in enumerate(values)]


def _video_name(value: object, *, suffix: str = "") -> str:
    if not isinstance(value, str) or value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise ValueError("asset name must be a nonempty basename")
    if suffix and not value.lower().endswith(suffix):
        raise ValueError(f"asset name must end with {suffix}")
    return value


def _folder_date(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
        raise ValueError("folder_date must be YYYYMMDD")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"invalid folder_date: {value}") from exc
    return value


def _name_pattern(name: str) -> tuple[str, str | None]:
    dated = DATED_NAME.fullmatch(name)
    if dated:
        return "date_and_clock", dated.group(1)
    if CLOCK_NAME.fullmatch(name):
        return "clock_only", None
    if SEQUENCE_NAME.fullmatch(name):
        return "sequence_only", None
    return "other", None


def load_video_master(database_path: Path, site: str) -> dict[str, Any]:
    """Read the site's video names and recorded paths with a read-only connection."""

    if site not in SITES or not database_path.is_file():
        raise ValueError("site or database path is invalid")
    uri = f"{database_path.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        rows = connection.execute(
            """SELECT video_name, video_path, recording_datetime_utc,
                      processed_datetime_utc
               FROM video_master WHERE intersection_id = ? ORDER BY video_name""",
            (site,),
        ).fetchall()
    return {
        "schema_version": 1,
        "site": site,
        "videos": [
            {
                "video_name": row[0],
                "video_path": row[1],
                "recording_datetime_utc": row[2],
                "processed_datetime_utc": row[3],
            }
            for row in rows
        ],
    }


def reconcile_assets(
    *,
    site: str,
    drive_inventory: Mapping[str, object],
    database_index: Mapping[str, object] | None = None,
    feature_index: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Compare scoped asset names without inferring hashes or timestamps."""

    if site not in SITES:
        raise ValueError(f"unsupported site: {site!r}")
    drive_rows = _records(drive_inventory, site=site, field="files")
    database_rows = (
        _records(database_index, site=site, field="videos") if database_index is not None else None
    )
    feature_rows = (
        _records(feature_index, site=site, field="features") if feature_index is not None else None
    )

    drive_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    name_patterns: Counter[str] = Counter()
    date_label_mismatches = 0
    for row in drive_rows:
        name = _video_name(row.get("name"), suffix=".mp4")
        folder_date = _folder_date(row.get("folder_date"))
        size = row.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("size_bytes must be a nonnegative integer")
        pattern, filename_date = _name_pattern(name)
        name_patterns[pattern] += 1
        if filename_date is not None and filename_date != folder_date:
            date_label_mismatches += 1
        stem = Path(name).stem
        if pattern == "clock_only" and stem.startswith("data_"):
            lookup_key = f"data_{folder_date}_{stem[5:]}"
        else:
            lookup_key = stem
        drive_by_name[lookup_key].append(
            {"folder_date": folder_date, "name": name, "size_bytes": size}
        )

    database_counts: Counter[str] | None = None
    database_by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    if database_rows is not None:
        database_counts = Counter()
        for row in database_rows:
            name = _video_name(row.get("video_name"))
            database_counts[name] += 1
            details = {"video_name": name}
            for field in ("video_path", "recording_datetime_utc", "processed_datetime_utc"):
                if field in row:
                    value = row[field]
                    if not isinstance(value, str):
                        raise ValueError(f"database {field} must be a string")
                    details[field] = value
            database_by_name[name].append(details)
    feature_counts: Counter[str] | None = None
    if feature_rows is not None:
        feature_counts = Counter(_video_name(row.get("video_name")) for row in feature_rows)

    keys = set(drive_by_name)
    if database_counts is not None:
        keys.update(database_counts)
    if feature_counts is not None:
        keys.update(feature_counts)
    records: list[dict[str, Any]] = []
    raw_database_status_counts: Counter[str] = Counter()
    feature_status_counts: Counter[str] = Counter()
    for name in sorted(keys):
        raw = sorted(
            drive_by_name.get(name, []),
            key=lambda item: (item["folder_date"], item["name"], item["size_bytes"]),
        )
        raw_count = len(raw)
        db_count = database_counts[name] if database_counts is not None else None
        feature_count = feature_counts[name] if feature_counts is not None else None
        if raw_count > 1 or (db_count is not None and db_count > 1):
            raw_database_status = "ambiguous_name"
        elif database_counts is None:
            raw_database_status = "database_not_checked"
        elif raw_count == 1 and db_count == 1:
            raw_database_status = "name_match_only"
        elif raw_count == 1:
            raw_database_status = "drive_only"
        elif db_count == 1:
            raw_database_status = "database_only"
        else:
            raw_database_status = "feature_only"
        if feature_count is None:
            feature_status = "not_checked"
        elif feature_count == 0:
            feature_status = "missing"
        elif feature_count == 1:
            feature_status = "one_row"
        else:
            feature_status = "multiple_rows"
        raw_database_status_counts[raw_database_status] += 1
        feature_status_counts[feature_status] += 1
        records.append(
            {
                "video_name": name,
                "drive_files": raw,
                "database_rows": db_count,
                "database_records": sorted(
                    database_by_name[name],
                    key=lambda item: (
                        item.get("video_path", ""),
                        item.get("recording_datetime_utc", ""),
                        item.get("processed_datetime_utc", ""),
                    ),
                ),
                "feature_rows": feature_count,
                "raw_database_status": raw_database_status,
                "feature_status": feature_status,
            }
        )

    return {
        "schema_version": 1,
        "site": site,
        "comparison_level": "name_only",
        "database_checked": database_index is not None,
        "features_checked": feature_index is not None,
        "counts": {
            "drive_files": len(drive_rows),
            "drive_names": len(drive_by_name),
            "database_rows": len(database_rows) if database_rows is not None else None,
            "database_names": len(database_counts) if database_counts is not None else None,
            "feature_rows": len(feature_rows) if feature_rows is not None else None,
            "feature_names": len(feature_counts) if feature_counts is not None else None,
            "drive_names_without_database_row": (
                sum(name not in database_counts for name in drive_by_name)
                if database_counts is not None
                else None
            ),
            "database_names_without_drive_file": (
                sum(name not in drive_by_name for name in database_counts)
                if database_counts is not None
                else None
            ),
            "database_names_without_feature_row": (
                sum(name not in feature_counts for name in database_counts)
                if database_counts is not None and feature_counts is not None
                else None
            ),
            "feature_names_without_database_row": (
                sum(name not in database_counts for name in feature_counts)
                if database_counts is not None and feature_counts is not None
                else None
            ),
            "repeated_drive_names": sum(len(rows) > 1 for rows in drive_by_name.values()),
            "extra_drive_name_occurrences": sum(len(rows) - 1 for rows in drive_by_name.values()),
            "name_patterns": dict(sorted(name_patterns.items())),
            "date_label_mismatches": date_label_mismatches,
            "raw_database_statuses": dict(sorted(raw_database_status_counts.items())),
            "feature_statuses": dict(sorted(feature_status_counts.items())),
        },
        "records": records,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return _object(json.loads(path.read_text(encoding="utf-8")), str(path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=sorted(SITES), required=True)
    parser.add_argument("--drive-inventory", type=Path, required=True)
    database_source = parser.add_mutually_exclusive_group()
    database_source.add_argument("--database", type=Path)
    database_source.add_argument("--database-index", type=Path)
    parser.add_argument("--feature-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    database_index = None
    if args.database is not None:
        database_index = load_video_master(args.database, args.site)
    elif args.database_index is not None:
        database_index = _load_json(args.database_index)
    report = reconcile_assets(
        site=args.site,
        drive_inventory=_load_json(args.drive_inventory),
        database_index=database_index,
        feature_index=_load_json(args.feature_index) if args.feature_index else None,
    )
    source_paths = (
        args.drive_inventory,
        args.database,
        args.database_index,
        args.feature_index,
    )
    if args.output.resolve() in {path.resolve() for path in source_paths if path is not None}:
        raise ValueError("output must not replace an input")
    output_bytes = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if args.output.exists():
        if args.output.read_bytes() != output_bytes:
            raise FileExistsError(f"refusing to overwrite a different report: {args.output}")
        print(json.dumps(report["counts"], sort_keys=True))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(output_bytes)
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
