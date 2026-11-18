from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import tomllib
from pathlib import Path

import pytest

from vehicle_emission.data_preflight import (
    REQUIRED_INPUTS,
    InputRequirement,
    build_preflight_report,
    write_preflight_report,
)

# These schema-v1 helpers remain historical fixture regressions only. The
# canonical ``make all`` route is required to exclude them (enforced below).
from vehicle_emission.reproduction import prepare_working_database, run_fixture_smoke


def _initialise_clean_freeze_repository(tmp_path: Path) -> tuple[str, Path]:
    """Create a clean repository whose empirical inputs remain intentionally untracked."""

    (tmp_path / ".gitignore").write_text(
        "source.db\nfreeze.json\nbuild/\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", ".gitignore"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, tmp_path / "build" / "test-working-provenance.db"


def test_data_preflight_is_machine_readable_and_deterministic(tmp_path) -> None:
    (tmp_path / "present.csv").write_text("header\nvalue\n", encoding="utf-8")
    requirements = (
        InputRequirement("present.csv", "file", "restore present"),
        InputRequirement("missing.db", "file", "restore missing"),
    )

    first = build_preflight_report(tmp_path, requirements)
    second = build_preflight_report(tmp_path, requirements)
    assert first == second
    assert first["status"] == "missing"
    assert first["required_inputs"][0]["status"] == "ready"
    assert first["required_inputs"][1] == {
        "path": "missing.db",
        "status": "missing",
        "recovery": "restore missing",
    }

    report_path = tmp_path / "build" / "preflight.json"
    write_preflight_report(first, report_path)
    assert json.loads(report_path.read_text(encoding="utf-8")) == first


def test_data_preflight_requires_the_recorded_primary_checkpoint() -> None:
    required_paths = {item.path for item in REQUIRED_INPUTS}
    assert "runs/rtdetr_finetune-4/weights/best.pt" in required_paths
    assert "runs/rtdetr_finetune-3/weights/best.pt" not in required_paths


def test_data_preflight_requires_registered_and_checked_out_submodule(tmp_path) -> None:
    checkout = tmp_path / "vendor" / "example"
    checkout.mkdir(parents=True)
    (checkout / "README.md").write_text("fixture\n", encoding="utf-8")

    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    requirement = InputRequirement(
        "vendor/example",
        "registered_git_submodule",
        "restore registration and initialise",
        commit,
    )

    assert build_preflight_report(tmp_path, (requirement,))["status"] == "missing"

    (tmp_path / ".gitmodules").write_text(
        '[submodule "vendor/example"]\n'
        "\tpath = vendor/example\n"
        "\turl = https://example.invalid/example.git\n",
        encoding="utf-8",
    )
    assert build_preflight_report(tmp_path, (requirement,))["status"] == "ready"

    wrong_commit = InputRequirement(
        "vendor/example",
        "registered_git_submodule",
        "restore registration and initialise",
        "0" * 40,
    )
    assert build_preflight_report(tmp_path, (wrong_commit,))["status"] == "missing"


def test_fixture_pipeline_smoke_is_deterministic(tmp_path) -> None:
    summary_path = run_fixture_smoke(tmp_path / "smoke")
    first = summary_path.read_bytes()
    second = run_fixture_smoke(tmp_path / "smoke").read_bytes()
    assert first == second
    assert json.loads(first)["status"] == "ready"


def test_prepare_working_database_requires_freeze_and_protects_inputs(tmp_path) -> None:
    source_db = tmp_path / "source.db"
    sqlite3.connect(source_db).close()
    missing_freeze = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="analysis-freeze"):
        prepare_working_database(source_db, Path("build/test-working.db"), missing_freeze)


def test_prepare_working_database_verifies_freeze_provenance(tmp_path, monkeypatch) -> None:
    git_commit, working = _initialise_clean_freeze_repository(tmp_path)
    monkeypatch.chdir(tmp_path)
    source_db = tmp_path / "source.db"
    sqlite3.connect(source_db).close()
    source_hash = hashlib.sha256(source_db.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "status": "accepted",
        "approved_by": "test-reviewer",
        "approved_at_utc": "2026-08-13T00:00:00Z",
        "git_commit": git_commit,
        "source_db_sha256": source_hash,
        "soramame_hour_convention": "end-labelled-hour",
        "eligibility_spec_sha256": "a" * 64,
        "inference_spec_sha256": "b" * 64,
    }
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(manifest), encoding="utf-8")
    prepare_working_database(source_db, working, freeze)
    assert working.read_bytes() == source_db.read_bytes()
    working.unlink()

    manifest["source_db_sha256"] = "0" * 64
    freeze.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="database hash"):
        prepare_working_database(source_db, working, freeze)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("approved_by", "", "approved_by"),
        ("approved_at_utc", "not-a-time", "approved_at_utc"),
        ("soramame_hour_convention", "not-a-convention", "soramame_hour_convention"),
        ("eligibility_spec_sha256", "g" * 64, "eligibility_spec_sha256"),
    ],
)
def test_prepare_working_database_rejects_malformed_freeze_fields(
    tmp_path, monkeypatch, field, value, message
) -> None:
    git_commit, working = _initialise_clean_freeze_repository(tmp_path)
    monkeypatch.chdir(tmp_path)
    source_db = tmp_path / "source.db"
    sqlite3.connect(source_db).close()
    manifest = {
        "schema_version": 1,
        "status": "accepted",
        "approved_by": "test-reviewer",
        "approved_at_utc": "2026-08-13T00:00:00Z",
        "git_commit": git_commit,
        "source_db_sha256": hashlib.sha256(source_db.read_bytes()).hexdigest(),
        "soramame_hour_convention": "end-labelled-hour",
        "eligibility_spec_sha256": "a" * 64,
        "inference_spec_sha256": "b" * 64,
    }
    manifest[field] = value
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        prepare_working_database(source_db, working, freeze)


def test_prepare_working_database_rejects_a_dirty_source_tree(tmp_path, monkeypatch) -> None:
    git_commit, working = _initialise_clean_freeze_repository(tmp_path)
    monkeypatch.chdir(tmp_path)
    source_db = tmp_path / "source.db"
    sqlite3.connect(source_db).close()
    manifest = {
        "schema_version": 1,
        "status": "accepted",
        "approved_by": "test-reviewer",
        "approved_at_utc": "2026-08-13T00:00:00Z",
        "git_commit": git_commit,
        "source_db_sha256": hashlib.sha256(source_db.read_bytes()).hexdigest(),
        "soramame_hour_convention": "end-labelled-hour",
        "eligibility_spec_sha256": "a" * 64,
        "inference_spec_sha256": "b" * 64,
    }
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "uncommitted.py").write_text("# uncommitted\n", encoding="utf-8")

    with pytest.raises(ValueError, match="uncommitted or untracked"):
        prepare_working_database(source_db, working, freeze)


def test_make_all_routes_to_v4_without_unrelated_global_data_preflight() -> None:
    import shutil

    if shutil.which("make") is None:
        pytest.skip("make utility is not available on this system")
    repository_root = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["make", "-n", "all", "PYTHON=python3.11"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository_root,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        pytest.skip(f"make execution not available on this system: {error}")
    commands = completed.stdout

    assert "vehicle_emission.data_preflight" not in commands
    assert "vehicle_emission.inference.frozen_analysis execute-v4" in commands
    assert "--config data/manifests/analysis-config-v4.json" in commands
    assert "--output-directory build/analysis_v4" in commands
    assert f"--repository-root {repository_root}" in commands

    forbidden_legacy_modules = (
        "vehicle_emission.reproduction prepare",
        "vehicle_emission.load_history_csvs",
        "vehicle_emission.calculate_emissions",
        "vehicle_emission.inference.run_analysis",
        "vehicle_emission.inference.plot_lag",
    )
    assert all(module not in commands for module in forbidden_legacy_modules)


def test_history_loader_cli_requires_an_explicit_or_accepted_hour_convention() -> None:
    source = Path("src/vehicle_emission/load_history_csvs.py").read_text(encoding="utf-8")
    assert "--soramame-hour-convention" in source
    assert "--runtime-settings" in source
    assert "Supply --soramame-hour-convention" in source


def test_cpu_test_lock_contains_every_direct_requirement() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    direct = project["dependencies"] + project["optional-dependencies"]["dev"]
    locked = {
        line.strip().lower()
        for line in Path("requirements-dev.lock").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert {requirement.lower() for requirement in direct} <= locked
