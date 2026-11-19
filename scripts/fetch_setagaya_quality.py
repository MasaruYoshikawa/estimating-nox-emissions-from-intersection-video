"""Archive and reconcile Tokyo's original hourly NOx quality for Setagaya."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from vehicle_emission.inference.monitor_quality import parse_tokyo_hourly
from vehicle_emission.inference.recorded_hourly_reanalysis import _csv, _hash, _json

URL = (
    "https://www.taiki.kankyo.metro.tokyo.lg.jp/taikikankyoapi/api/HourSokuho"
    "?startdatetime=2026050101&enddatetime=2026063024&stationid=0110&itemid=005"
)
DOCUMENTATION = "https://www.taiki.kankyo.metro.tokyo.lg.jp/taikikankyo5g/apiguid/index.html"


def fetch(output: Path, source: Path) -> dict[str, object]:
    """Require complete source agreement before publishing the quality evidence."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("provider evidence output must be fresh")
    source_hash = _hash(source)
    with urlopen(URL, timeout=60) as response:
        content = response.read()
    quality = parse_tokyo_hourly(content.decode("utf-8-sig"), "0110", "Setagaya")
    original = pd.read_csv(source, encoding="utf-8-sig")
    starts = (
        (
            pd.to_datetime(original.Date, dayfirst=True)
            + pd.to_timedelta(pd.to_numeric(original.Time) - 1, unit="h")
        )
        .dt.tz_localize("Asia/Tokyo")
        .dt.tz_convert("UTC")
    )
    original["monitor_start_utc"] = starts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if len(quality) != 1464 or len(original) != 1464:
        raise ValueError("expected a complete May-June hourly series")
    matched = quality.merge(original, on="monitor_start_utc", validate="one_to_one")
    if len(matched) != 1464:
        raise ValueError("provider and original intervals differ")
    valid = matched.quality_status.eq("valid")
    if not np.isclose(
        matched.loc[valid, "nox_ppm"], matched.loc[valid, "NOx(ppm)"], atol=1e-12, rtol=0
    ).all():
        raise ValueError("provider valid values do not match the original source")
    if not matched.loc[~valid, "NOx(ppm)"].eq(0).all():
        raise ValueError("provider missing flags do not reconcile to the recorded zero values")
    summary = {
        "provider": "Tokyo Metropolitan Government",
        "url": URL,
        "documentation": DOCUMENTATION,
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "provider_station_id": "0110",
        "monitoring_place": "Setagaya",
        "units": "provider ppb divided by 1000 for original ppm",
        "time": "end-labelled JST hour 1..24 mapped to UTC interval start",
        "provider_rows": len(quality),
        "matched_source_rows": len(matched),
        "valid_numeric_rows": int(valid.sum()),
        "numeric_mismatches": 0,
        "missing_rows": int((~valid).sum()),
        "missing_cause": "not specified by the hourly response",
    }
    if _hash(source) != source_hash:
        raise ValueError("original source changed during retrieval")
    output.mkdir(parents=True)
    (output / "provider_hourly.csv").write_bytes(content)
    _csv(output / "provider_quality.csv", quality)
    _json(
        output / "source.json",
        {
            **summary,
            "original_source_sha256": source_hash,
            "calculator_sha256": _hash(Path(__file__)),
            "parser_sha256": _hash(
                Path(__file__).parents[1] / "src/vehicle_emission/inference/monitor_quality.py"
            ),
            "outputs": {path.name: _hash(path) for path in sorted(output.iterdir())},
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=Path("data/airpollution_history/AirPollution_Setagaya.csv")
    )
    args = parser.parse_args()
    print(json.dumps(fetch(args.output_directory, args.source), indent=2))


if __name__ == "__main__":
    main()
