"""Export recorded SQLite evidence without manufacturing analysis-ready seconds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from vehicle_emission.asset_lineage import SITES

TABLES = {
    "video_master": ("video_id", "intersection_id"),
    "video_emissions": ("video_id", "estimation_method"),
    "entering_counts": ("video_id", "direction", "class_id", "vehicle_count"),
    "crossing_events": ("video_id", "event_id"),
    "vehicle_classification_master": ("class_id", "class_name"),
    "air_pollution_history": ("timestamp_utc", "monitoring_place"),
    "weather_history": ("timestamp_utc",),
}
AUDIT_COLUMNS = [
    "source_site",
    "video_id",
    "direction",
    "direction_total",
    "class_row_sum",
    "recorded_entering_count",
    "count_status",
]


def _signature(database: Path) -> tuple[int, int, int, int]:
    sidecars = _sidecars(database)
    if sidecars.get("-wal", 0) > 0 or "-journal" in sidecars:
        raise ValueError("use a closed source snapshot; nonempty SQLite WAL or journal exists")
    status = database.stat()
    return status.st_size, status.st_mtime_ns, status.st_ctime_ns, status.st_ino


def _sidecars(database: Path) -> dict[str, int]:
    return {
        suffix: path.stat().st_size
        for suffix in ("-wal", "-shm", "-journal")
        if (path := Path(str(database) + suffix)).exists()
    }


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _count_audit(
    videos: list[tuple[str, int]], rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source_site"], row["video_id"], row["direction"])].append(row)
    represented = {(site, video) for site, video, _ in grouped}
    for site, video in videos:
        if (site, video) not in represented:
            grouped[(site, video, "")] = []
    audit = []
    for (site, video, direction), records in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        total = class_sum = selected = None
        status = "not_recorded"
        keys = [row["class_id"] for row in records]
        values = [row["vehicle_count"] for row in records]
        if records:
            if not isinstance(direction, str) or not direction.strip():
                status = "invalid_direction"
            elif len(keys) != len(set(keys)):
                status = "duplicate_logical_key"
            elif any(value is not None and value not in (0, 1, 2, 3) for value in keys):
                status = "noncanonical_class"
            elif any(
                not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
                or value != int(value)
                for value in values
            ):
                status = "invalid_count"
            else:
                totals = [row["vehicle_count"] for row in records if row["class_id"] is None]
                classes = [row["vehicle_count"] for row in records if row["class_id"] is not None]
                total = int(totals[0]) if totals else None
                class_sum = int(sum(classes)) if classes else None
                if total is not None and class_sum is not None and total != class_sum:
                    status = "direction_total_class_mismatch"
                elif total is not None:
                    selected = total
                    status = "total_and_classes_agree" if classes else "direction_total_only"
                else:
                    selected = class_sum
                    status = "class_sum_only"
        audit.append(
            dict(
                zip(
                    AUDIT_COLUMNS,
                    (site, video, direction, total, class_sum, selected, status),
                    strict=True,
                )
            )
        )
    return audit


def export_source(
    *, database: Path, output_directory: Path, sites: Sequence[str] = tuple(sorted(SITES))
) -> dict[str, Any]:
    """Publish a fresh raw bundle from one closed, read-only SQLite snapshot.

    Counts select direction totals or class sums, never their sum. Audit status
    is arithmetic consistency, not independent entering-event validation.
    """
    if not sites or len(set(sites)) != len(sites) or not set(sites) <= SITES:
        raise ValueError("sites must be distinct supported site names")
    database = database.resolve(strict=True)
    if not database.is_file():
        raise ValueError("database must be an existing file")
    destination = output_directory.resolve()
    if destination.exists() or output_directory.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output_directory}")
    signature = _signature(database)
    sidecars_before = _sidecars(database)
    destination.parent.mkdir(parents=True, exist_ok=True)
    placeholders = ",".join("?" for _ in sites)
    metadata: dict[str, Any] = {}
    count_rows: list[dict[str, Any]] = []
    videos: list[tuple[str, int]] = []
    video_ids: set[int] = set()
    with tempfile.TemporaryDirectory(prefix=".source-export-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            for table, required in TABLES.items():
                definition = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
                ).fetchone()
                columns = [
                    row[1] for row in connection.execute(f"PRAGMA table_info({_quote(table)})")
                ]
                if not definition and table == "video_master":
                    raise ValueError("video_master is required")
                if definition and not set(required) <= set(columns):
                    raise ValueError(
                        f"{table} lacks required fields: {sorted(set(required) - set(columns))}"
                    )
                per_video = table in {
                    "video_master",
                    "video_emissions",
                    "entering_counts",
                    "crossing_events",
                }
                headers = (["source_site"] if per_video else []) + (columns or list(required))
                row_count = 0
                query = None
                with (stage / f"{table}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle, lineterminator="\n")
                    writer.writerow(headers)
                    if definition:
                        if table == "video_master":
                            query = f"SELECT intersection_id, * FROM video_master WHERE intersection_id IN ({placeholders}) ORDER BY video_id"
                        elif per_video:
                            query = (
                                f"SELECT v.intersection_id, t.* FROM {_quote(table)} t JOIN video_master v USING(video_id) WHERE v.intersection_id IN ({placeholders}) ORDER BY "
                                + ",".join(f"t.{_quote(field)}" for field in required)
                            )
                        else:
                            query = f"SELECT * FROM {_quote(table)} ORDER BY " + ",".join(
                                map(_quote, required)
                            )
                        for row in connection.execute(query, tuple(sites) if per_video else ()):
                            writer.writerow(row)
                            row_count += 1
                            record = dict(zip(headers, row, strict=True))
                            if table == "video_master":
                                if record["video_id"] in video_ids:
                                    raise ValueError(
                                        "video_master contains duplicate selected video_id"
                                    )
                                video_ids.add(record["video_id"])
                                videos.append((record["source_site"], record["video_id"]))
                            elif table == "entering_counts":
                                count_rows.append(record)
                metadata[table] = {
                    "available": bool(definition),
                    "columns": columns,
                    "create_sql": definition[0] if definition else None,
                    "rows": row_count,
                    "query": query,
                    "parameters": list(sites) if per_video and definition else [],
                }
            connection.rollback()
        audit = _count_audit(videos, count_rows)
        with (stage / "entering_count_audit.csv").open("w", newline="", encoding="utf-8") as handle:
            audit_writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS, lineterminator="\n")
            audit_writer.writeheader()
            audit_writer.writerows(audit)
        manifest = {
            "schema_version": "raw_source_export_v1",
            "status": "raw_source_export_not_analysis_ready",
            "sites": sorted(sites),
            "exporter_sha256": _sha256(Path(__file__)),
            "database": {
                "name": database.name,
                "size_bytes": signature[0],
                "mtime_ns": signature[1],
                "ctime_ns": signature[2],
                "inode": signature[3],
                "content_hash": None,
                "reader_sidecars_before": sidecars_before,
                "reader_sidecars_after": _sidecars(database),
            },
            "tables": metadata,
            "interpretation": {
                "timestamps": "Recorded text is preserved; clock and interval semantics are not inferred.",
                "coverage": "Stored start/duration does not establish valid recording intervals or valid seconds.",
                "emissions": "Unadjusted and hybrid-adjusted columns are distinct; video_emissions.vehicle_count counts tracks, not entering events.",
                "counts": "Arithmetic audit only; crossing events require explicit line/direction definitions and unique-event validation.",
                "monitors": "Stored rows retain missing values; original-observation status and interval semantics require feed evidence.",
                "provenance": "Copied source fields are not verification of checkpoint identity, calibration production use or measurement accuracy.",
            },
            "outputs": {path.name: _sha256(path) for path in sorted(stage.iterdir())},
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if _signature(database) != signature:
            raise ValueError("source database changed during export; no bundle published")
        lock = destination.parent / f".{destination.name}.publish.lock"
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"refusing to overwrite existing output: {destination}")
            stage.rename(destination)
        finally:
            os.close(descriptor)
            lock.unlink()
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Export source facts without calculating correlations or eligibility."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sites", nargs="+", choices=sorted(SITES), default=sorted(SITES))
    args = parser.parse_args(argv)
    manifest = export_source(
        database=args.database, output_directory=args.output_directory, sites=args.sites
    )
    print(
        json.dumps(
            {
                "output_directory": str(args.output_directory),
                "tables": {name: value["rows"] for name, value in manifest["tables"].items()},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
