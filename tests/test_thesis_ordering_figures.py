"""Focused fixtures for descriptive hourly model ordering and diurnal scaling."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from scripts.thesis_ordering_figures import MODELS, SERIES, figures, ordering, run, scale_diurnal


def fixture_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site": ["Sukiyabashi"] * 4,
            "monitor_start_utc": ["a", "b", "c", "d"],
            MODELS[0]: [0.0, 1.0, 1.0, 3.0],
            MODELS[1]: [0.0, 3.0, 2.0, 1.0],
            "clip_speed_mean": [1.0, 2.0, 3.0, 4.0],
            "bus_share": [0.1, 0.2, 0.3, np.nan],
            "heavy_truck_share": [0.0, 0.0, 0.0, 0.0],
        }
    )


def test_rank_sign_ratio_and_pair_partition() -> None:
    panel, table = ordering(fixture_panel())
    summary = table.iloc[0]
    assert panel.continuous_minus_movestar_rank.tolist() == [0, 1.5, 0.5, -2]
    assert panel.continuous_to_movestar_ratio.iloc[1] == 3
    assert np.isnan(panel.continuous_to_movestar_ratio.iloc[0])
    assert summary.n_ratio_hours == 3
    assert summary.n_ratio_zero_denominator == 1
    assert summary.concordant_pairs == 3
    assert summary.discordant_pairs == 2
    assert summary.movestar_only_tied_pairs == 1
    assert summary.continuous_only_tied_pairs == summary.both_tied_pairs == 0
    assert summary.total_hour_pairs == 6
    assert summary.rank_difference_bus_heavy_entering_share_n == 3


def test_exact_tie_categories_missing_models_and_unique_keys() -> None:
    panel = fixture_panel()
    panel[MODELS[1]] = [0, 1, 1, np.nan]
    _, table = ordering(panel)
    assert table.iloc[0].both_tied_pairs == 1
    assert table.iloc[0].concordant_pairs == 2
    assert table.iloc[0].n_unpaired_model_hours == 1
    with pytest.raises(ValueError, match="unique"):
        ordering(pd.concat([panel, panel]))


def test_diurnal_support_weighting_and_units() -> None:
    source = pd.DataFrame(
        {
            "site": ["s"] * 4,
            "hour_jst": [9, 10, 9, 10],
            "variable": [MODELS[0], MODELS[0], "ambient_nox_ppm", "ambient_nox_ppm"],
            "n_hours": [1, 3, 2, 2],
            "mean": [1, 3, 0.01, 0.02],
            "median": [1, 3, 0.01, 0.02],
            "q25": [0, 2, 0.005, 0.01],
            "q75": [2, 4, 0.02, 0.03],
        }
    )
    result = scale_diurnal(source)
    model = result.loc[result.variable.eq(MODELS[0])]
    assert model.support_weighted_overall_mean.tolist() == [2.5, 2.5]
    assert model.display_mean.tolist() == pytest.approx([0.4, 1.2])
    assert result.loc[result.variable.eq("ambient_nox_ppm"), "display_mean"].tolist() == [10, 20]


def test_zero_normalisation_and_existing_output(tmp_path: Path) -> None:
    zero = pd.DataFrame(
        {
            "site": ["s"],
            "hour_jst": [9],
            "variable": [MODELS[0]],
            "n_hours": [1],
            "mean": [0],
            "median": [0],
            "q25": [0],
            "q75": [0],
        }
    )
    assert scale_diurnal(zero).display_mean.isna().all()
    with pytest.raises(FileExistsError):
        run(tmp_path / "absent", tmp_path)


def test_end_to_end_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    first = fixture_panel()
    pd.concat([first, first.assign(site="Shinjuku")]).to_csv(
        source / "extension_hourly_panel.csv", index=False
    )
    rows = [
        {
            "site": site,
            "hour_jst": hour,
            "variable": variable,
            "n_hours": 2,
            "mean": 2.0,
            "median": 2.0,
            "q25": 1.0,
            "q75": 3.0,
        }
        for site in ("Sukiyabashi", "Shinjuku")
        for hour in (9, 10)
        for variable in (
            "ambient_nox_ppm",
            *MODELS,
            "vehicle_count",
            "captured_clip_bus_heavy_count",
        )
    ]
    pd.DataFrame(rows).to_csv(source / "diurnal_profiles.csv", index=False)
    output = tmp_path / "output"
    run(source, output)
    receipt = json.loads((output / "receipt.json").read_text())
    for name, expected in receipt["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert len(receipt["outputs"]) == 5


@pytest.mark.parametrize(
    "shape", ["empty", "one", "ties", "constant_x", "constant_y", "nonfinite_x"]
)
def test_ordering_figures_handle_sparse_and_degenerate_panels(
    shape: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = fixture_panel()
    source = pd.concat([source, source.assign(site="Shinjuku")], ignore_index=True)
    if shape == "one":
        source = source.groupby("site", sort=False).head(1)
    panel, _ = ordering(source)
    if shape == "empty":
        panel = panel.iloc[:0]
    elif shape == "ties":
        panel["clip_speed_mean"] = [1, 1, 2, 2] * 2
        panel["bus_heavy_entering_share"] = [0.1, 0.1, 0.2, 0.2] * 2
    elif shape == "constant_x":
        panel["clip_speed_mean"] = 5
        panel["bus_heavy_entering_share"] = 0.1
    elif shape == "constant_y":
        panel["continuous_minus_movestar_rank"] = 0
    elif shape == "nonfinite_x":
        panel["clip_speed_mean"] = np.nan
        panel["bus_heavy_entering_share"] = np.inf
    diurnal = scale_diurnal(
        pd.DataFrame(
            [
                {
                    "site": site,
                    "hour_jst": 9,
                    "variable": variable,
                    "n_hours": 1,
                    "mean": 2.0,
                    "median": 2.0,
                    "q25": 1.0,
                    "q75": 3.0,
                }
                for site in ("Sukiyabashi", "Shinjuku")
                for variable in ("ambient_nox_ppm", *SERIES)
            ]
        )
    )
    original_panel, original_diurnal = panel.copy(deep=True), diurnal.copy(deep=True)
    arrows, bar_labels = [], []
    annotate, text = plt.Axes.annotate, plt.Axes.text

    def capture_arrow(self: plt.Axes, *args: object, **kwargs: object) -> object:
        start, end = kwargs["xytext"], kwargs["xy"]
        assert np.isfinite(np.asarray([start, end], dtype=float)).all()
        assert start != end
        arrows.append((start, end))
        return annotate(self, *args, **kwargs)

    def capture_label(
        self: plt.Axes, x: float, y: float, label: str, *args: object, **kwargs: object
    ) -> object:
        if label.startswith(("+", "-")):
            value = float(label)
            assert kwargs["ha"] == "left"
            if value < 0:
                assert x == 0.02
            else:
                assert x == pytest.approx(value + 0.02, abs=0.0005)
            bar_labels.append(value)
        return text(self, x, y, label, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "annotate", capture_arrow)
    monkeypatch.setattr(plt.Axes, "text", capture_label)
    figures(panel, diurnal, tmp_path)
    if shape in ("empty", "one", "constant_x", "nonfinite_x"):
        assert not arrows
    if shape == "ties":
        assert any(value < 0 for value in bar_labels)
    for name in ("fig_model_rank_disagreement.png", "fig_recorded_diurnal_profiles.png"):
        assert (tmp_path / name).stat().st_size > 0
    pd.testing.assert_frame_equal(panel, original_panel)
    pd.testing.assert_frame_equal(diurnal, original_diurnal)
