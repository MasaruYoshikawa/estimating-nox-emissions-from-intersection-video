"""Source binding and fresh-output tests for the figure-bundle runner."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from vehicle_emission.visualisation import comparisons, evidence
from vehicle_emission.visualisation.thesis_figures import run


@pytest.fixture
def sources(tmp_path: Path) -> tuple[Path, Path]:
    extension, measurement = tmp_path / "extension", tmp_path / "measurement"
    for directory, names in (
        (
            extension,
            (
                "extension_hourly_panel.csv",
                "class_reference_weights.csv",
                "count_comparisons.csv",
                "volatility_comparisons.csv",
            ),
        ),
        (measurement, ("count_detail_pairs.csv",)),
    ):
        directory.mkdir()
        recorded = {}
        for name in names:
            path = directory / name
            path.write_text("value\n1\n2\n", encoding="utf-8")
            recorded[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (directory / "receipt.json").write_text(json.dumps({"outputs": recorded}))
    return extension, measurement


@pytest.fixture
def render_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    def render(*arguments: object) -> None:
        output = arguments[-1]
        assert isinstance(output, Path)
        output.write_bytes(b"deterministic test image")

    monkeypatch.setattr(comparisons, "count_plot_data", lambda frame: frame.copy())
    monkeypatch.setattr(comparisons, "volatility_plot_data", lambda frame: frame.copy())
    monkeypatch.setattr(evidence, "rank_associations", lambda frame: (frame.copy(), frame.copy()))
    monkeypatch.setattr(evidence, "fleet_contributions", lambda frame, weights: frame.copy())
    monkeypatch.setattr(comparisons, "plot_count_comparisons", render)
    monkeypatch.setattr(comparisons, "plot_volatility_comparisons", render)
    monkeypatch.setattr(evidence, "plot_rank_associations", render)
    monkeypatch.setattr(evidence, "plot_count_discrepancies", render)
    monkeypatch.setattr(evidence, "plot_fleet_contributions", render)


def test_output_must_be_fresh_before_reading_sources(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError, match="fresh"):
        run(tmp_path / "missing", tmp_path / "also_missing", tmp_path)


@pytest.mark.parametrize(
    ("parent", "name"),
    [
        (0, "extension_hourly_panel.csv"),
        (0, "class_reference_weights.csv"),
        (0, "count_comparisons.csv"),
        (0, "volatility_comparisons.csv"),
        (1, "count_detail_pairs.csv"),
    ],
)
def test_changed_source_is_rejected_without_output(
    sources: tuple[Path, Path], tmp_path: Path, parent: int, name: str
) -> None:
    (sources[parent] / name).write_text("value\n3\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="source receipt"):
        run(*sources, output)
    assert not output.exists()


def test_receipt_covers_inputs_calculators_and_all_outputs(
    sources: tuple[Path, Path], tmp_path: Path, render_stubs: None
) -> None:
    before = {p: p.read_bytes() for folder in sources for p in folder.iterdir()}
    output = tmp_path / "output"
    run(*sources, output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert len(receipt["inputs"]) == 7
    assert len(receipt["calculator_sha256"]) == 3
    assert len(receipt["outputs"]) == 10
    assert len(list(output.glob("*.png"))) == 5
    assert len(list(output.glob("*.csv"))) == 5
    for group in ("inputs", "calculator_sha256"):
        for name, digest in receipt[group].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    for name, digest in receipt["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert all(path.read_bytes() == content for path, content in before.items())
    assert pd.read_csv(output / "fleet_count_comparisons.csv").value.tolist() == [1, 2]


def test_output_directory_does_not_change_rendering_receipt(
    sources: tuple[Path, Path], tmp_path: Path, render_stubs: None
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    run(*sources, first)
    run(*sources, second)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }
