"""Check root pytest configuration in isolated interpreter sessions."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")


@pytest.mark.parametrize("configured_temporary_variables", [False, True])
def test_conftest_import_preserves_temporary_settings(
    tmp_path: Path, configured_temporary_variables: bool
) -> None:
    environment = os.environ.copy()
    for name in ("TEMP", "TMP", "TMPDIR"):
        if configured_temporary_variables:
            directory = tmp_path / name.lower()
            directory.mkdir()
            environment[name] = str(directory)
        else:
            environment.pop(name, None)
    environment["PROJECT_CONFTEST"] = str(CONFTEST)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    probe = """
import os
import runpy
import tempfile
from pathlib import Path

names = ("TEMP", "TMP", "TMPDIR")
before = {name: os.environ.get(name) for name in names}
tempfile.tempdir = None
files_before = set(Path.cwd().rglob("*"))
runpy.run_path(os.environ["PROJECT_CONFTEST"])
assert {name: os.environ.get(name) for name in names} == before
assert tempfile.tempdir is None
assert set(Path.cwd().rglob("*")) == files_before
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("explicit_basetemp", [False, True])
def test_pytest_preserves_other_sessions_and_basetemp(
    tmp_path: Path, explicit_basetemp: bool
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    shutil.copyfile(CONFTEST, project / "conftest.py")
    temporary_root = tmp_path / "temporary"
    other_session = temporary_root / "py_tmp" / "s-other-session"
    other_session.mkdir(parents=True)
    sentinel = other_session / "keep.txt"
    sentinel.write_text("another session's file", encoding="utf-8")
    expected_basetemp = tmp_path / "requested-basetemp" if explicit_basetemp else None
    environment = os.environ.copy()
    for name in ("TEMP", "TMP", "TMPDIR"):
        environment[name] = str(temporary_root)
    environment["EXPECTED_TEMP"] = str(temporary_root)
    environment["EXPECTED_BASETEMP"] = str(expected_basetemp) if expected_basetemp else ""
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_PLUGINS", None)

    (project / "test_probe.py").write_text(
        """
import os
import tempfile
from pathlib import Path


def test_temp_settings(tmp_path, pytestconfig):
    expected_temp = Path(os.environ["EXPECTED_TEMP"]).resolve()
    assert Path(tempfile.gettempdir()).resolve() == expected_temp
    for name in ("TEMP", "TMP", "TMPDIR"):
        assert Path(os.environ[name]).resolve() == expected_temp
    expected_basetemp = os.environ["EXPECTED_BASETEMP"]
    if expected_basetemp:
        assert Path(pytestconfig.option.basetemp).resolve() == Path(expected_basetemp).resolve()
        assert tmp_path.is_relative_to(Path(expected_basetemp).resolve())
    else:
        assert pytestconfig.option.basetemp is None
        assert tmp_path.is_relative_to(expected_temp)
""",
        encoding="utf-8",
    )
    command = [sys.executable, "-m", "pytest", "-q", "--confcutdir", str(project)]
    if expected_basetemp:
        command.extend(["--basetemp", str(expected_basetemp)])
    command.append(str(project / "test_probe.py"))
    completed = subprocess.run(
        command,
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert sentinel.read_text(encoding="utf-8") == "another session's file"
