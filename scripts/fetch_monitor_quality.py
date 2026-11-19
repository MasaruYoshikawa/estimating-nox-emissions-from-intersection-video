"""Retrieve and reconcile the receptor stations' official hourly NOx records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

BASE = "https://soramame.env.go.jp"
STATIONS = {"Sukiyabashi": "13101520", "Shinjuku": "13104540"}
QUALITY_COLUMNS = (
    "monitoring_place",
    "monitor_start_utc",
    "provider_station_id",
    "provider_value",
    "nox_ppm",
    "quality_status",
)


def parse_provider_rows(records: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep literal provider missing markers distinct from numeric zero."""
    station_names = {value: key for key, value in STATIONS.items()}
    output: list[dict[str, str]] = []
    keys: set[tuple[str, str]] = set()
    for record in records:
        station = record["SKT_CD"]
        if station not in station_names:
            continue
        hour = int(record["SKT_TIME"])
        if hour not in range(1, 25):
            raise ValueError("provider hour must be an end label from 1 to 24")
        local_midnight = datetime.strptime(record["SKT_DATE"], "%Y/%m/%d").replace(
            tzinfo=timezone(timedelta(hours=9))
        )
        start = (local_midnight + timedelta(hours=hour - 1)).astimezone(UTC)
        timestamp = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        key = (station, timestamp)
        if key in keys:
            raise ValueError("duplicate provider station-hour")
        keys.add(key)
        value = record["NOX"].strip()
        if value in {"", "-"}:
            status, number = "missing", ""
        else:
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError("provider NOx must be missing or finite nonnegative")
            status, number = "valid", value
        output.append(
            dict(
                zip(
                    QUALITY_COLUMNS,
                    (station_names[station], timestamp, station, value, number, status),
                    strict=True,
                )
            )
        )
    return sorted(output, key=lambda row: (row["monitoring_place"], row["monitor_start_utc"]))


def reconcile_source(source: Path, quality: list[dict[str, str]], site: str) -> dict[str, int]:
    """Require exact dates/hours and numerical agreement for every valid NOx row."""
    official = {row["monitor_start_utc"]: row for row in quality if row["monitoring_place"] == site}
    counts = {
        "source_rows": 0,
        "valid_numeric_matches": 0,
        "zero_coded_missing": 0,
        "nonzero_source_now_missing": 0,
    }
    seen: set[str] = set()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            day, month, year = (int(part) for part in row["Date"].split("/"))
            local = datetime(year, month, day, tzinfo=timezone(timedelta(hours=9)))
            hour = int(row["Time"])
            if hour not in range(1, 25):
                raise ValueError("source hour must be an end label from 1 to 24")
            timestamp = (
                (local + timedelta(hours=hour - 1)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            if timestamp in seen or timestamp not in official:
                raise ValueError("source/provider hourly identities differ")
            seen.add(timestamp)
            provider = official[timestamp]
            original = float(row["NOx(ppm)"])
            counts["source_rows"] += 1
            if provider["quality_status"] == "valid":
                if not math.isclose(original, float(provider["nox_ppm"]), abs_tol=1e-12):
                    raise ValueError(f"valid source/provider NOx mismatch: {site} {timestamp}")
                counts["valid_numeric_matches"] += 1
            elif original == 0:
                counts["zero_coded_missing"] += 1
            else:
                counts["nonzero_source_now_missing"] += 1
    if set(official) != seen:
        raise ValueError("source/provider hourly coverage differs")
    return counts


def _fetch(url: str) -> bytes:
    with urlopen(url, timeout=45) as response:
        if response.status != 200:
            raise ValueError(f"provider HTTP status: {response.status}")
        return response.read()


def fetch_quality(root: Path, output: Path) -> dict[str, Any]:
    """Save original official bytes, reconciled quality and a portable receipt."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing evidence: {output}")
    params = {
        "Start_YM": "202605",
        "End_YM": "202606",
        "TDFKN_CD": "13",
        "SKT_CD": "13101520,13104010,13104540",
        "REQUEST_DATA": "NOX,NO,NO2,CO,OX",
    }
    url = BASE + "/soramame/api/data_search?" + urlencode(params)
    raw = _fetch(url)
    records = json.loads(raw)
    quality = parse_provider_rows(records)
    if len(quality) != 2 * 61 * 24:
        raise ValueError("expected complete two-station May/June hourly source coverage")
    sources = {
        site: root / f"data/airpollution_history/AirPollution_{site}.csv" for site in STATIONS
    }
    reconciliation = {
        site: reconcile_source(source, quality, site) for site, source in sources.items()
    }
    master_url = BASE + "/data/map/kyokuNoudo/2026/05/01/01.csv"
    master = _fetch(master_url)
    master_rows = list(csv.DictReader(io.StringIO(master.decode("utf-8-sig"))))
    identity = [
        row for row in master_rows if row["測定局コード"] in {"13101520", "13104010", "13104540"}
    ]
    if len(identity) != 3:
        raise ValueError("provider master does not uniquely identify all three stations")
    api_doc_url = BASE + "/data/apiManual/3.html"
    format_url = BASE + "/data/legend/download/format.html"
    api_doc, format_doc = _fetch(api_doc_url), _fetch(format_url)
    output.mkdir(parents=True)
    files = {
        "soramame_receptor_hourly_raw.json": raw,
        "soramame_station_master_20260501.csv": master,
        "soramame_api_parameters.html": api_doc,
        "soramame_download_format.html": format_doc,
    }
    for name, content in files.items():
        (output / name).write_bytes(content)
    with (output / "receptor_quality.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=QUALITY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(quality)
    summary = {
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "sources": {
            "soramame_receptor_hourly_raw.json": url,
            "soramame_station_master_20260501.csv": master_url,
            "soramame_api_parameters.html": api_doc_url,
            "soramame_download_format.html": format_url,
        },
        "interval_mapping": "provider date JST plus end-labelled hour minus one hour; UTC start",
        "provider_missing_markers": ["", "-"],
        "numeric_zero": "valid",
        "station_identity": identity,
        "source_reconciliation": reconciliation,
        "input_sha256": {
            str(source.relative_to(root)): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in sources.values()
        },
        "calculator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir())
        },
    }
    (output / "receipt.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = fetch_quality(args.root.resolve(), args.output_directory)
    print(json.dumps(result["source_reconciliation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
