"""Hand calculations, count-cell integrity and reproducible weighted audit tests."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from vehicle_emission.measurement.weighted_count_audit import (
    CLASS_NAMES,
    DETAIL_NAME,
    PROJECT_CLASSES,
    WEIGHTS_NAME,
    run,
    weighted_count_audit,
)

ROOT = Path(__file__).resolve().parents[1]


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class_id": list(CLASS_NAMES),
            "project_class": list(PROJECT_CLASSES.values()),
            "normalised_weight": [2.0, 1.0, 3.0, 0.5],
        }
    )


def _clip(
    manual: tuple[int, ...] = (2, 10, 1, 2),
    automated: tuple[int, ...] = (3, 8, 1, 2),
    *,
    site: str = "Site A",
    clip: int = 1,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site": site,
            "kind": "classification",
            "clip": clip,
            "category": list(CLASS_NAMES.values()),
            "manual_count": manual,
            "automated_count": automated,
        }
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bundles(tmp_path: Path) -> tuple[Path, Path]:
    measurement, extension = tmp_path / "measurement", tmp_path / "extension"
    for directory, name, frame in (
        (measurement, DETAIL_NAME, _clip()),
        (extension, WEIGHTS_NAME, _weights()),
    ):
        directory.mkdir()
        frame.to_csv(directory / name, index=False)
        (directory / "receipt.json").write_text(
            json.dumps({"outputs": {name: _hash(directory / name)}})
        )
    return extension, measurement


def test_hand_calculation_keeps_class_cancellation_separate() -> None:
    tables = weighted_count_audit(_clip(), _weights())
    row = tables["weighted_count_clips.csv"].iloc[0]
    assert row.manual_count == 15
    assert row.automated_count == 14
    assert row.absolute_cell_difference == 3
    assert row.weighted_manual_count == 18
    assert row.weighted_automated_count == 18
    assert row.weighted_signed_difference == 0
    assert row.absolute_weighted_clip_net_difference == 0
    assert row.weighted_absolute_cell_difference == 4
    assert row.relative_weighted_absolute_cell_difference == pytest.approx(4 / 18)
    assert row.relative_unweighted_absolute_cell_difference == pytest.approx(3 / 15)
    cells = tables["weighted_count_cells.csv"]
    assert cells.weighted_absolute_difference.tolist() == [2, 2, 0, 0]
    classes = tables["weighted_count_classes.csv"].set_index("category")
    assert classes.loc["Bus", "share_of_site_weighted_absolute_cell_difference"] == 0.5
    assert classes.loc["Car", "share_of_site_weighted_manual_count"] == pytest.approx(10 / 18)
    assert classes.share_of_site_weighted_manual_count.sum() == pytest.approx(1)
    assert classes.share_of_site_weighted_absolute_cell_difference.sum() == pytest.approx(1)


def test_site_aggregation_preserves_between_clip_cancellation() -> None:
    pairs = pd.concat(
        [
            _clip((2, 10, 1, 2), (3, 10, 1, 2)),
            _clip((2, 10, 1, 2), (1, 10, 1, 2), clip=2),
            _clip(site="Site B"),
        ],
        ignore_index=True,
    )
    tables = weighted_count_audit(pairs, _weights())
    sites = tables["weighted_count_sites.csv"].set_index("site")
    a = sites.loc["Site A"]
    assert a.clips == 2
    assert a.class_cells == 8
    assert a.weighted_manual_count == 36
    assert a.weighted_absolute_cell_difference == 4
    assert a.sum_absolute_weighted_clip_net_difference == 4
    assert a.absolute_weighted_site_net_difference == 0
    assert a.relative_weighted_absolute_cell_difference == pytest.approx(4 / 36)
    classes = tables["weighted_count_classes.csv"]
    bus = classes.loc[classes.site.eq("Site A") & classes.category.eq("Bus")].iloc[0]
    assert bus.weighted_absolute_cell_difference == 4
    assert bus.absolute_weighted_class_net_difference == 0
    assert tables["weighted_count_cells.csv"].shape[0] == 12


@pytest.mark.parametrize("automated", [(0, 0, 0, 0), (1, 0, 0, 0)])
def test_zero_manual_denominator_is_not_zero_error_rate(automated: tuple[int, ...]) -> None:
    tables = weighted_count_audit(_clip((0, 0, 0, 0), automated), _weights())
    for name in (
        "weighted_count_clips.csv",
        "weighted_count_sites.csv",
        "weighted_count_classes.csv",
    ):
        assert tables[name].relative_weighted_absolute_cell_difference.isna().all()
    row = tables["weighted_count_sites.csv"].iloc[0]
    assert row.weighted_manual_count == 0
    assert row.weighted_absolute_cell_difference == 2 * automated[0]
    assert tables["weighted_count_cells.csv"].shape[0] == 4
    if automated[0] == 0:
        assert (
            tables["weighted_count_classes.csv"]
            .share_of_site_weighted_absolute_cell_difference.isna()
            .all()
        )


def test_entering_counts_are_not_treated_as_class_counts() -> None:
    entering = _clip(site="Other unit")
    entering["kind"] = "entering"
    entering["category"] = ["Left", "Right", "Top", "Bottom"]
    expected = weighted_count_audit(_clip(), _weights())
    actual = weighted_count_audit(pd.concat([_clip(), entering]), _weights())
    for name in expected:
        assert_frame_equal(actual[name], expected[name])


@pytest.mark.parametrize("column", ["manual_count", "automated_count", "clip"])
@pytest.mark.parametrize("value", [-1, 0.5, np.nan, np.inf, "missing", True])
def test_invalid_counts_and_clip_identities_rejected(column: str, value: object) -> None:
    pairs = _clip().astype({column: object})
    pairs.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        weighted_count_audit(pairs, _weights())


def test_zero_clip_identity_rejected() -> None:
    with pytest.raises(ValueError, match="clip"):
        weighted_count_audit(_clip(clip=0), _weights())


@pytest.mark.parametrize(
    "column,value",
    [
        ("site", None),
        ("site", ""),
        ("category", None),
        ("category", ""),
        ("category", "unknown"),
        ("kind", None),
        ("kind", ""),
        ("kind", "unknown"),
    ],
)
def test_missing_or_unknown_identity_rejected(column: str, value: object) -> None:
    pairs = _clip()
    pairs.loc[0, column] = value
    with pytest.raises(ValueError):
        weighted_count_audit(pairs, _weights())


def test_duplicate_cells_rejected_even_if_one_other_class_is_missing() -> None:
    pairs = _clip()
    pairs.loc[0, "category"] = "Car"
    with pytest.raises(ValueError, match="duplicate"):
        weighted_count_audit(pairs, _weights())


def test_all_four_categories_required_even_zero_counts() -> None:
    with pytest.raises(ValueError, match="all four"):
        weighted_count_audit(_clip().iloc[1:], _weights())
    pairs = _clip()
    pairs.loc[0, "category"] = "Other"
    with pytest.raises(ValueError, match="four project classes"):
        weighted_count_audit(pairs, _weights())


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, "unknown", True])
def test_invalid_weights_rejected(value: object) -> None:
    weights = _weights().astype({"normalised_weight": object})
    weights.loc[0, "normalised_weight"] = value
    with pytest.raises(ValueError, match="finite positive"):
        weighted_count_audit(_clip(), weights)


def test_error_larger_than_manual_total_is_not_clamped() -> None:
    tables = weighted_count_audit(_clip((0, 1, 0, 0), (2, 1, 0, 0)), _weights())
    assert (
        tables["weighted_count_sites.csv"].iloc[0].relative_weighted_absolute_cell_difference == 4
    )


@pytest.mark.parametrize("case", ["duplicate", "missing", "wrong_class", "wrong_name"])
def test_weight_class_universe_and_mapping_are_required(case: str) -> None:
    weights = _weights()
    if case == "duplicate":
        weights.loc[0, "class_id"] = 1
    elif case == "missing":
        weights = weights.iloc[1:]
    elif case == "wrong_class":
        weights.loc[0, "class_id"] = 4
    else:
        weights.loc[0, "project_class"] = "car"
    with pytest.raises(ValueError):
        weighted_count_audit(_clip(), weights)


@pytest.mark.parametrize("name", ["pairs", "weights"])
def test_missing_required_column_rejected(name: str) -> None:
    pairs, weights = _clip(), _weights()
    if name == "pairs":
        pairs = pairs.drop(columns="manual_count")
    else:
        weights = weights.drop(columns="normalised_weight")
    with pytest.raises(ValueError, match="missing columns"):
        weighted_count_audit(pairs, weights)


def test_no_classification_pairs_rejected() -> None:
    pairs = _clip()
    pairs["kind"] = "entering"
    with pytest.raises(ValueError, match="no classification"):
        weighted_count_audit(pairs, _weights())


def test_runner_checks_receipts_preserves_inputs_and_replays(tmp_path: Path) -> None:
    extension, measurement = _source_bundles(tmp_path)
    before = {
        path: _hash(path) for directory in (extension, measurement) for path in directory.iterdir()
    }
    first, second = tmp_path / "first", tmp_path / "second"
    run(extension, measurement, first)
    run(extension, measurement, second)
    assert {path: _hash(path) for path in before} == before
    assert {path.name for path in first.iterdir()} == {path.name for path in second.iterdir()}
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    receipt = json.loads((first / "receipt.json").read_text())
    assert len(receipt["inputs"]) == 4
    for name, expected in receipt["outputs"].items():
        assert _hash(first / name) == expected
    calculator = ROOT / "src/vehicle_emission/measurement/weighted_count_audit.py"
    assert receipt["calculator_sha256"] == {calculator.name: _hash(calculator)}


@pytest.mark.parametrize("which", ["pairs", "weights"])
def test_receipt_tamper_rejected_before_any_output(tmp_path: Path, which: str) -> None:
    extension, measurement = _source_bundles(tmp_path)
    source = measurement / DETAIL_NAME if which == "pairs" else extension / WEIGHTS_NAME
    source.write_text(source.read_text() + "\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="hash mismatch"):
        run(extension, measurement, output)
    assert not output.exists()


@pytest.mark.parametrize("case", ["directory", "file", "dangling_symlink"])
def test_existing_output_rejected_before_source_access(tmp_path: Path, case: str) -> None:
    output = tmp_path / "existing"
    if case == "directory":
        output.mkdir()
    elif case == "file":
        output.write_text("unchanged")
    else:
        output.symlink_to(tmp_path / "missing-target")
    with pytest.raises(ValueError, match="fresh"):
        run(tmp_path / "absent", tmp_path / "also_absent", output)
    if case == "file":
        assert output.read_text() == "unchanged"


@pytest.mark.parametrize("which", ["extension", "measurement"])
def test_output_cannot_be_nested_in_source_bundle(tmp_path: Path, which: str) -> None:
    extension, measurement = _source_bundles(tmp_path)
    source = extension if which == "extension" else measurement
    output = source / "new_output"
    with pytest.raises(ValueError, match="outside source"):
        run(extension, measurement, output)
    assert not output.exists()


def test_invalid_counts_do_not_create_output(tmp_path: Path) -> None:
    extension, measurement = _source_bundles(tmp_path)
    pairs = _clip()
    pairs.loc[0, "manual_count"] = -1
    pairs.to_csv(measurement / DETAIL_NAME, index=False)
    (measurement / "receipt.json").write_text(
        json.dumps({"outputs": {DETAIL_NAME: _hash(measurement / DETAIL_NAME)}})
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="manual_count"):
        run(extension, measurement, output)
    assert not output.exists()


def test_input_change_during_calculation_leaves_no_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extension, measurement = _source_bundles(tmp_path)
    original = weighted_count_audit

    def changing_audit(detail: pd.DataFrame, weights: pd.DataFrame) -> dict[str, pd.DataFrame]:
        tables = original(detail, weights)
        source = measurement / DETAIL_NAME
        source.write_text(source.read_text() + "\n")
        return tables

    monkeypatch.setattr(
        "vehicle_emission.measurement.weighted_count_audit.weighted_count_audit", changing_audit
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="source changed"):
        run(extension, measurement, output)
    assert not (output / "receipt.json").exists()


def test_calculator_change_during_run_leaves_no_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extension, measurement = _source_bundles(tmp_path)
    reads = 0

    def changing_digest(path: Path) -> str:
        nonlocal reads
        if path.name == "weighted_count_audit.py":
            reads += 1
            if reads > 1:
                return "0" * 64
        return _hash(path)

    monkeypatch.setattr(
        "vehicle_emission.measurement.weighted_count_audit._digest", changing_digest
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="calculator changed"):
        run(extension, measurement, output)
    assert not (output / "receipt.json").exists()


def test_recorded_pairs_reconcile_to_original_count_totals() -> None:
    detail = pd.read_csv(ROOT / "results/measurement_model_extension_20261109" / DETAIL_NAME)
    weights = pd.read_csv(
        ROOT / "results/recorded_hourly_extension_20261109" / WEIGHTS_NAME,
        float_precision="round_trip",
    )
    tables = weighted_count_audit(detail, weights)
    sites = tables["weighted_count_sites.csv"].set_index("site")
    for site, clips, manual, automated, difference in (
        ("Sukiyabashi", 7, 1500, 1511, 69),
        ("Shinjuku", 5, 1041, 1064, 45),
    ):
        row = sites.loc[site]
        assert row.clips == clips
        assert row.class_cells == clips * 4
        assert row.manual_count == manual
        assert row.automated_count == automated
        assert row.absolute_cell_difference == difference
        assert row.weighted_absolute_cell_difference >= row.absolute_weighted_site_net_difference
        source = detail.loc[detail.site.eq(site) & detail.kind.eq("classification")]
        reference = dict(
            zip(weights.class_id.map(CLASS_NAMES), weights.normalised_weight, strict=True)
        )
        expected_manual = sum(
            reference[row.category] * row.manual_count for row in source.itertuples()
        )
        expected_absolute = sum(
            reference[row.category] * abs(row.automated_count - row.manual_count)
            for row in source.itertuples()
        )
        assert row.weighted_manual_count == pytest.approx(expected_manual)
        assert row.weighted_absolute_cell_difference == pytest.approx(expected_absolute)
