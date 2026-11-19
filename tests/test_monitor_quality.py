"""Provider quality is interval-specific and distinct from a numerical zero."""

from __future__ import annotations

import pandas as pd
import pytest

from vehicle_emission.inference.monitor_quality import apply_monitor_quality, parse_tokyo_hourly


def test_tokyo_end_hour_and_missing_are_not_numeric_zero() -> None:
    text = (
        "STATIONID,ITEMID,YEAR,MONTH,DAY,HOUR,VALUE\n"
        "0110,005,2026,5,1,1,0\n"
        "0110,005,2026,5,1,24,欠測\n"
        "0110,005,2026,5,2,1,\n"
    )
    parsed = parse_tokyo_hourly(text, "0110", "Setagaya")
    assert parsed.monitor_start_utc.tolist() == [
        "2026-04-30T15:00:00Z",
        "2026-05-01T14:00:00Z",
        "2026-05-01T15:00:00Z",
    ]
    assert parsed.quality_status.tolist() == ["valid", "missing", "uncollected"]
    assert parsed.nox_ppm.iloc[0] == 0
    assert parsed.nox_ppm.iloc[1:].isna().all()


@pytest.mark.parametrize("old,new", [("0110", "0142"), ("005", "003"), (",24,", ",25,")])
def test_provider_station_pollutant_and_hour_must_match(old: str, new: str) -> None:
    text = "STATIONID,ITEMID,YEAR,MONTH,DAY,HOUR,VALUE\n0110,005,2026,5,1,24,5\n"
    with pytest.raises(ValueError):
        parse_tokyo_hourly(text.replace(old, new), "0110", "Setagaya")


def panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site": ["Sukiyabashi", "Shinjuku"],
            "monitor_start_utc": ["2026-05-01T00:00:00Z"] * 2,
            "ambient_nox_ppm": [0.0, 0.01],
            "setagaya_nox_ppm": [0.0, 0.0],
            "vehicle_count": [5, 6],
            "bus_share": [0.1, 0.1],
            "heavy_truck_share": [0.1, 0.1],
            "light_truck_share": [0.1, 0.1],
            "temperature": [20, 20],
            "wind_speed": [2, 2],
            "residual_complete": [True, True],
        }
    )


def quality() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "monitoring_place": ["Sukiyabashi", "Setagaya"],
            "monitor_start_utc": ["2026-05-01T00:00:00Z"] * 2,
            "provider_station_id": ["13101520", "0110"],
            "provider_value": ["0", "欠測"],
            "nox_ppm": [0, None],
            "quality_status": ["valid", "missing"],
        }
    )


def test_valid_zero_retained_background_missing_only_narrows_regional_support() -> None:
    original = panel()
    fixed, changes = apply_monitor_quality(original, quality())
    assert len(fixed) == 2
    assert fixed.ambient_nox_ppm.tolist() == [0.0, 0.01]
    assert fixed.setagaya_nox_ppm.isna().all()
    assert not fixed.residual_complete.any()
    assert changes.action.tolist() == ["set_background_missing"] * 2
    assert original.setagaya_nox_ppm.eq(0).all()


def test_only_flagged_receptor_is_excluded() -> None:
    evidence = quality()
    evidence.loc[0, ["quality_status", "nox_ppm"]] = ["missing", None]
    fixed, changes = apply_monitor_quality(panel(), evidence)
    assert fixed.site.tolist() == ["Shinjuku"]
    assert changes.action.eq("exclude_receptor_hour").sum() == 1


def test_unreviewed_zero_is_not_invented_missing() -> None:
    fixed, _ = apply_monitor_quality(panel(), quality().iloc[1:])
    assert len(fixed) == 2
    assert fixed.receptor_quality_status.eq("unreviewed").all()


def test_numeric_mismatch_and_duplicate_evidence_rejected() -> None:
    evidence = quality()
    evidence.loc[0, "nox_ppm"] = 0.005
    with pytest.raises(ValueError, match="numeric value differs"):
        apply_monitor_quality(panel(), evidence)
    with pytest.raises(ValueError, match="distinct"):
        apply_monitor_quality(panel(), pd.concat([quality(), quality()]))


def test_quality_status_must_agree_with_value() -> None:
    evidence = quality()
    evidence.loc[1, "nox_ppm"] = 0
    with pytest.raises(ValueError, match="quality status"):
        apply_monitor_quality(panel(), evidence)
