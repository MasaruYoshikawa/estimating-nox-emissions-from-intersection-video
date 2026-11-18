"""Fixtures for plotting recorded paired estimates without changing intervals."""

from pathlib import Path

import pandas as pd
import pytest

from vehicle_emission.visualisation.comparisons import (
    ADVANTAGES,
    BLOCKS,
    MODELS,
    REFERENCES,
    SITES,
    count_plot_data,
    plot_count_comparisons,
    plot_volatility_comparisons,
    volatility_plot_data,
)


def comparison_rows(estimands: list[str]) -> pd.DataFrame:
    rows = []
    for site in SITES:
        for estimand in estimands:
            for block in BLOCKS:
                direct = estimand.startswith("lower_minus_higher:")
                lower = estimand.startswith("lower:")
                rows.append(
                    {
                        "site": site,
                        "estimand": estimand,
                        "estimate": -0.06 if direct else 0.02,
                        "ci_low": -0.12 if block == 1 else -0.14,
                        "ci_high": 0.11 if block == 1 else 0.13,
                        "block_length_days": block,
                        "interval_family": "pointwise_descriptive",
                        "status": "estimated",
                        "n_hours": 11,
                        "point_stratum_hours": (
                            11
                            if ":" in estimand and not estimand.startswith(("lower:", "higher:"))
                            else (6 if lower else 5)
                        ),
                        "point_stratum_days": 3,
                        "resample_sha256": "fixture-shared-draws",
                    }
                )
    return pd.DataFrame(rows)


def count_fixture() -> pd.DataFrame:
    return comparison_rows([f"{m}:{r}" for m in MODELS for r in REFERENCES])


def volatility_fixture() -> pd.DataFrame:
    return comparison_rows(
        [f"{s}:{a}" for s in ("lower", "higher", "lower_minus_higher") for a in ADVANTAGES]
    )


def test_count_estimates_endpoints_and_support_are_unchanged() -> None:
    source = count_fixture()
    original = source.copy(deep=True)
    result = count_plot_data(source)
    keys = ["site", "estimand", "block_length_days"]
    pd.testing.assert_frame_equal(
        result[source.columns].sort_values(keys).reset_index(drop=True),
        source.sort_values(keys).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(source, original)
    assert set(result.model) == {"MOVEStar", "VT-CPFM"}
    assert len(result) == 24


@pytest.mark.parametrize(
    "fault", ["duplicate", "family", "missing", "block", "support", "reversed", "nonfinite"]
)
def test_count_rejects_mismatched_comparison_semantics(fault: str) -> None:
    source = count_fixture()
    if fault == "duplicate":
        source = pd.concat([source, source.iloc[:1]])
    elif fault == "family":
        source.loc[0, "interval_family"] = "ten_estimate_family"
    elif fault == "missing":
        source = source.iloc[1:]
    elif fault == "block":
        source.loc[0, "block_length_days"] = 2
    elif fault == "support":
        source.loc[0, "point_stratum_hours"] = 10
    elif fault == "reversed":
        source.loc[0, "ci_low"] = 0.9
    else:
        source.loc[0, "ci_low"] = float("nan")
    with pytest.raises(ValueError):
        count_plot_data(source)


def test_volatility_preserves_direct_sign_intervals_and_group_support() -> None:
    source = volatility_fixture()
    irrelevant = source.iloc[:1].assign(estimand="lower:vehicle_count")
    result = volatility_plot_data(pd.concat([source, irrelevant]))
    assert len(result) == 24
    assert result.loc[result.stratum.eq("lower_minus_higher"), "estimate"].eq(-0.06).all()
    assert result.loc[result.stratum.eq("lower"), "point_stratum_hours"].eq(6).all()
    assert result.loc[result.stratum.eq("higher"), "point_stratum_hours"].eq(5).all()
    assert result.loc[result.stratum.eq("lower_minus_higher"), "point_stratum_hours"].eq(11).all()
    keys = ["site", "estimand", "block_length_days"]
    pd.testing.assert_frame_equal(
        result[source.columns].sort_values(keys).reset_index(drop=True),
        source.sort_values(keys).reset_index(drop=True),
    )


def test_volatility_requires_stable_point_estimates_across_blocks() -> None:
    source = volatility_fixture()
    source.loc[0, "estimate"] = 0.8
    with pytest.raises(ValueError, match="match across block"):
        volatility_plot_data(source)


@pytest.mark.parametrize("kind", ["count", "volatility"])
def test_comparison_figures_render(tmp_path: Path, kind: str) -> None:
    output = tmp_path / f"{kind}.png"
    if kind == "count":
        plot_count_comparisons(count_fixture(), output)
    else:
        plot_volatility_comparisons(volatility_fixture(), output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 10_000
