from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vehicle_emission import database
from vehicle_emission.inference.correlation import (
    compute_spearman_bootstrap,
    compute_spearman_bootstrap_bca,
    compute_time_lagged_cross_correlation,
    load_aligned_time_series,
)


def test_spearman_bootstrap_basics() -> None:
    # Strong positive linear correlation
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    r, ci_low, ci_high = compute_spearman_bootstrap(x, y, num_resamples=50, seed=42)
    assert pytest.approx(r, abs=1e-5) == 1.0
    assert pytest.approx(ci_low, abs=1e-5) == 1.0
    assert pytest.approx(ci_high, abs=1e-5) == 1.0


def test_spearman_bootstrap_determinism() -> None:
    # Create some noisy correlated data
    rng = np.random.default_rng(100)
    x = rng.normal(size=50)
    y = x + rng.normal(scale=0.5, size=50)

    r1, low1, high1 = compute_spearman_bootstrap(x, y, num_resamples=100, seed=123)
    r2, low2, high2 = compute_spearman_bootstrap(x, y, num_resamples=100, seed=123)

    # Assert determinism across calls (INV-001)
    assert r1 == r2
    assert low1 == low2
    assert high1 == high2


def test_spearman_bootstrap_insufficient_data() -> None:
    # Less than 3 data points should return NaN
    r, low, high = compute_spearman_bootstrap([1, 2], [3, 4])
    assert np.isnan(r)
    assert np.isnan(low)
    assert np.isnan(high)


def test_time_lagged_cross_correlation_structure() -> None:
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
        }
    )

    results = compute_time_lagged_cross_correlation(
        df, "x", "y", max_lag_minutes=20, step_minutes=10, num_resamples=10
    )

    # max_lag = 20, step = 10 -> lags should be -20, -10, 0, 10, 20 (5 steps)
    assert len(results) == 5
    lags = [item["lag_minutes"] for item in results]
    assert lags == [-20.0, -10.0, 0.0, 10.0, 20.0]

    # Zero lag should have perfect correlation
    zero_lag_item = next(item for item in results if item["lag_minutes"] == 0.0)
    assert pytest.approx(zero_lag_item["r_s"], abs=1e-5) == 1.0


def test_load_aligned_time_series_runs() -> None:
    db_path = "data/results.db"
    if not Path(db_path).is_file() or Path(db_path).stat().st_size == 0:
        pytest.skip("Empirical database is not available in this clone")
    df_hourly = load_aligned_time_series(db_path, "June26th_6AM_8PM", resample_10min=False)
    assert isinstance(df_hourly, pd.DataFrame)
    if not df_hourly.empty:
        assert "nox_suki" in df_hourly.columns
        assert "nox_deducted" in df_hourly.columns
        assert "vehicle_count" in df_hourly.columns


def test_spearman_bootstrap_bca() -> None:
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    y = [1.2, 1.8, 3.5, 3.8, 5.5, 5.2, 7.1, 7.9, 9.2, 9.8, 11.5, 11.8, 13.5, 13.8, 15.5]

    r, low, high = compute_spearman_bootstrap_bca(x, y, num_resamples=100, seed=42)
    assert r > 0.9
    assert low > 0.8
    assert high <= 1.0


def test_entering_count_aggregation_uses_each_vehicle_once(tmp_path) -> None:
    """Direction totals define the total; class rows define composition only."""

    db_path = str(tmp_path / "counts.db")
    database.init_db(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        INSERT INTO video_master (
            video_id, video_name, video_path, intersection_id,
            recording_datetime_utc, recording_date, recording_time,
            processed_datetime_utc, group_id
        ) VALUES (1, 'counts', 'counts.mp4', 'Sukiyabashi',
                  '2026-04-30T15:00:00Z', '2026-04-30', '15:00:00',
                  '2026-04-30T15:01:00Z', 'count-fixture')
        """
    )
    for place, nox in (("Sukiyabashi", 0.01), ("Setagaya", 0.005), ("Shinjuku", 0.008)):
        connection.execute(
            "INSERT INTO air_pollution_history VALUES (?, ?, NULL, NULL, ?, NULL, NULL)",
            ("2026-04-30T15:00:00Z", place, nox),
        )
    connection.execute(
        "INSERT INTO weather_history VALUES (?, NULL, 0, 1, 'east', NULL, 1)",
        ("2026-04-30T15:00:00Z",),
    )
    connection.commit()
    connection.close()
    database.insert_entering_counts(
        db_path,
        1,
        [
            {"direction": "entering_from_left", "class_id": None, "vehicle_count": 10},
            {"direction": "entering_from_left", "class_id": 1, "vehicle_count": 8},
            {"direction": "entering_from_left", "class_id": 2, "vehicle_count": 2},
            # This direction represents a legacy/class-only producer. Its five
            # vehicles must still be included even though the other direction
            # has an explicit total row.
            {"direction": "entering_from_right", "class_id": 1, "vehicle_count": 5},
        ],
    )

    aligned = load_aligned_time_series(db_path, "count-fixture", resample_10min=False)

    assert len(aligned) == 1
    assert aligned.loc[0, "vehicle_count"] == 15
    assert aligned.loc[0, "heavy_vehicle_count"] == 2
    assert aligned.loc[0, "timestamp"] == pd.Timestamp("2026-04-30T15:00:00Z")


def test_inference_rejects_legacy_duplicate_direction_totals(tmp_path) -> None:
    """Read-only inference cannot bypass the legacy count-integrity audit."""

    db_path = str(tmp_path / "duplicate-counts.db")
    database.init_db(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute("DROP INDEX entering_counts_direction_total_identity")
    connection.executemany(
        """
        INSERT INTO entering_counts (video_id, direction, class_id, vehicle_count)
        VALUES (1, 'entering_from_left', NULL, ?)
        """,
        [(4,), (5,)],
    )
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="duplicate logical keys"):
        load_aligned_time_series(db_path, "unused", resample_10min=False)
