"""Official receptor data keep missing markers and physical zeros distinct."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/fetch_monitor_quality.py"
SPEC = importlib.util.spec_from_file_location("fetch_monitor_quality", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(value: str, hour: str = "10") -> dict[str, str]:
    return {
        "SKT_CD": "13104540",
        "SKT_DATE": "2026/05/07",
        "SKT_TIME": hour,
        "NOX": value,
    }


def test_provider_missing_is_not_numeric_zero() -> None:
    missing = MODULE.parse_provider_rows([record("-")])[0]
    zero = MODULE.parse_provider_rows([record("0")])[0]
    assert missing["quality_status"] == "missing" and missing["nox_ppm"] == ""
    assert zero["quality_status"] == "valid" and zero["nox_ppm"] == "0"
    assert zero["monitor_start_utc"] == "2026-05-07T00:00:00Z"
    assert zero["provider_station_id"] == "13104540"


def test_end_label_24_is_same_local_date_23_start() -> None:
    parsed = MODULE.parse_provider_rows([record("0.012", "24")])[0]
    assert parsed["monitor_start_utc"] == "2026-05-07T14:00:00Z"


@pytest.mark.parametrize("value", ["-0.001", "NaN", "Infinity", "unknown"])
def test_unexpected_values_are_not_silently_missing(value: str) -> None:
    with pytest.raises(ValueError):
        MODULE.parse_provider_rows([record(value)])


def test_duplicate_or_invalid_hour_fails() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.parse_provider_rows([record("0.012"), record("0.013")])
    with pytest.raises(ValueError, match="end label"):
        MODULE.parse_provider_rows([record("0.012", "0")])


def test_source_agreement_requires_all_valid_values(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("Date,Time,NOx(ppm)\n7/05/2026,10,0.012\n")
    quality = MODULE.parse_provider_rows([record("0.012")])
    assert MODULE.reconcile_source(source, quality, "Shinjuku") == {
        "source_rows": 1,
        "valid_numeric_matches": 1,
        "zero_coded_missing": 0,
        "nonzero_source_now_missing": 0,
    }
    quality[0]["nox_ppm"] = "0.013"
    with pytest.raises(ValueError, match="mismatch"):
        MODULE.reconcile_source(source, quality, "Shinjuku")


def test_new_provider_missing_status_retains_nonzero_source_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("Date,Time,NOx(ppm)\n7/05/2026,10,0.012\n")
    quality = MODULE.parse_provider_rows([record("-")])
    counts = MODULE.reconcile_source(source, quality, "Shinjuku")
    assert counts["nonzero_source_now_missing"] == 1
    assert counts["zero_coded_missing"] == 0
    assert source.read_text().endswith("7/05/2026,10,0.012\n")
