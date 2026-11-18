"""Preserved-input and fresh-output checks for the complete presentation refresh."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from scripts import refresh_thesis_figures as refresh


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def sources(tmp_path: Path) -> dict[str, Path]:
    directories = {}
    for label, names in (
        ("extension", refresh.EXTENSION_INPUTS),
        ("measurement", refresh.MEASUREMENT_INPUTS),
        ("ordering", refresh.ORDERING_INPUTS),
    ):
        directory = tmp_path / label
        directory.mkdir()
        recorded = {}
        for name in names:
            path = directory / name
            path.write_text("value\n1.0000000000000002\n2\n", encoding="utf-8")
            recorded[name] = digest(path)
        (directory / "receipt.json").write_text(json.dumps({"outputs": recorded}))
        directories[label] = directory
    profile = tmp_path / "profile.png"
    profile.write_bytes(b"preserved constructed-profile image")
    return {**directories, "profile": profile}


@pytest.fixture
def render_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    def image(*arguments: object) -> None:
        destination = arguments[-1]
        assert isinstance(destination, Path)
        destination.write_bytes(b"deterministic rendered image")

    def core_figures(panel: pd.DataFrame, tables: dict, output: Path) -> None:
        assert set(tables) == {"core_inference.csv", "residual_comparisons.csv"}
        for name in (
            "fig_recorded_calendar_support.png",
            "fig_core_associations.png",
            "fig_rank_residual_associations.png",
            "fig_block_comparison.png",
        ):
            image(output / name)

    def ordering_figures(panel: pd.DataFrame, diurnal: pd.DataFrame, output: Path) -> None:
        image(output / "fig_model_rank_disagreement.png")
        image(output / "fig_recorded_diurnal_profiles.png")

    monkeypatch.setattr(refresh.comparisons, "count_plot_data", lambda frame: frame.copy())
    monkeypatch.setattr(refresh.comparisons, "volatility_plot_data", lambda frame: frame.copy())
    monkeypatch.setattr(
        refresh.evidence, "rank_associations", lambda frame: (frame.copy(), frame.copy())
    )
    monkeypatch.setattr(
        refresh.evidence, "fleet_contributions", lambda frame, weights: frame.copy()
    )
    monkeypatch.setattr(refresh.recorded_hourly_extension, "_figures", core_figures)
    monkeypatch.setattr(refresh.ordering_figures, "figures", ordering_figures)
    monkeypatch.setattr(refresh.thesis_diagnostics, "_response_plot", image)
    monkeypatch.setattr(refresh.comparisons, "plot_count_comparisons", image)
    monkeypatch.setattr(refresh.comparisons, "plot_volatility_comparisons", image)
    monkeypatch.setattr(refresh.evidence, "plot_rank_associations", image)
    monkeypatch.setattr(refresh.evidence, "plot_count_discrepancies", image)
    monkeypatch.setattr(refresh.evidence, "plot_fleet_contributions", image)


def test_fresh_output_is_checked_before_accessing_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError, match="fresh"):
        refresh.run(
            extension=tmp_path / "missing",
            measurement=tmp_path / "missing",
            ordering=tmp_path / "missing",
            profile=tmp_path / "missing.png",
            output=tmp_path,
        )


def test_output_cannot_be_nested_in_a_source_bundle(sources: dict[str, Path]) -> None:
    output = sources["extension"] / "new"
    with pytest.raises(ValueError, match="outside"):
        refresh.run(**sources, output=output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("parent", "name"),
    [
        *(("extension", name) for name in refresh.EXTENSION_INPUTS),
        *(("measurement", name) for name in refresh.MEASUREMENT_INPUTS),
        *(("ordering", name) for name in refresh.ORDERING_INPUTS),
    ],
)
def test_tampered_numeric_input_is_rejected_before_output(
    sources: dict[str, Path], tmp_path: Path, parent: str, name: str
) -> None:
    (sources[parent] / name).write_text("value\n3\n", encoding="utf-8")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="source receipt"):
        refresh.run(**sources, output=output)
    assert not output.exists()


def test_missing_profile_is_rejected_before_output(
    sources: dict[str, Path], tmp_path: Path
) -> None:
    sources["profile"] = tmp_path / "missing.png"
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        refresh.run(**sources, output=output)
    assert not output.exists()


def test_receipt_and_tables_bind_preserved_inputs_without_inference(
    sources: dict[str, Path], tmp_path: Path, render_stubs: None
) -> None:
    before = {
        path: path.read_bytes()
        for source in sources.values()
        for path in (source.iterdir() if source.is_dir() else [source])
    }
    output = tmp_path / "output"
    refresh.run(**sources, output=output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["kind"] == "thesis_presentation_refresh"
    assert len(receipt["inputs"]) == 16
    assert len(receipt["calculator_sha256"]) == 8
    assert len(receipt["outputs"]) == 18
    assert len(list(output.glob("*.png"))) == 13
    assert len(list(output.glob("*.csv"))) == 5
    assert receipt["preserved_images"] == {refresh.PROFILE_OUTPUT: str(sources["profile"])}
    assert (output / refresh.PROFILE_OUTPUT).read_bytes() == sources["profile"].read_bytes()
    for group in ("inputs", "calculator_sha256"):
        for name, expected in receipt[group].items():
            assert digest(Path(name)) == expected
    for name, expected in receipt["outputs"].items():
        assert digest(output / name) == expected
    assert all(path.read_bytes() == original for path, original in before.items())
    values = pd.read_csv(output / "hourly_rank_points.csv", float_precision="round_trip")
    assert values.value.tolist() == [1.0000000000000002, 2.0]


def test_source_mutation_during_rendering_does_not_produce_a_receipt(
    sources: dict[str, Path], tmp_path: Path, render_stubs: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def alter_source(frame: pd.DataFrame, output: Path) -> None:
        output.write_bytes(b"test image")
        (sources["measurement"] / "count_detail_pairs.csv").write_text("value\n3\n")

    monkeypatch.setattr(refresh.evidence, "plot_count_discrepancies", alter_source)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="source changed"):
        refresh.run(**sources, output=output)
    assert not (output / "receipt.json").exists()


def test_replay_is_independent_of_the_fresh_output_name(
    sources: dict[str, Path], tmp_path: Path, render_stubs: None
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    refresh.run(**sources, output=first)
    refresh.run(**sources, output=second)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }


def test_code_mutation_during_rendering_does_not_produce_a_receipt(
    sources: dict[str, Path], tmp_path: Path, render_stubs: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calculator = tmp_path / "calculator.py"
    calculator.write_text("original code\n")
    monkeypatch.setattr(refresh, "__file__", str(calculator))

    def alter_code(frame: pd.DataFrame, output: Path) -> None:
        output.write_bytes(b"test image")
        calculator.write_text("modified code\n")

    monkeypatch.setattr(refresh.evidence, "plot_count_discrepancies", alter_code)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="plot code changed"):
        refresh.run(**sources, output=output)
    assert not (output / "receipt.json").exists()
