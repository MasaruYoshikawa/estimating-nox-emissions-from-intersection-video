"""Local source-video metadata inventory tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vehicle_emission.asset_lineage import reconcile_assets
from vehicle_emission.source_inventory import build_inventory, main


def _source_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "videos"
    day = root / "20260501"
    day.mkdir(parents=True)
    return root, day


def test_inventory_is_sorted_metadata_and_lineage_compatible(tmp_path: Path) -> None:
    root, day = _source_tree(tmp_path)
    later = root / "20260502"
    later.mkdir()
    (later / "data_002.mp4").write_bytes(b"123")
    (day / "data_003.mp4").write_bytes(b"12")
    (day / "data_001.MP4").write_bytes(b"1")
    (root / ".DS_Store").write_bytes(b"ignored")
    (day / ".DS_Store").write_bytes(b"ignored")

    inventory = build_inventory(site="Sukiyabashi", source_root=root)

    assert inventory == {
        "schema_version": 1,
        "site": "Sukiyabashi",
        "files": [
            {"folder_date": "20260501", "name": "data_001.MP4", "size_bytes": 1},
            {"folder_date": "20260501", "name": "data_003.mp4", "size_bytes": 2},
            {"folder_date": "20260502", "name": "data_002.mp4", "size_bytes": 3},
        ],
    }
    report = reconcile_assets(site="Sukiyabashi", drive_inventory=inventory)
    assert report["counts"]["drive_files"] == 3
    assert report["counts"]["database_rows"] is None


def test_cli_replay_is_byte_identical_without_media_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, day = _source_tree(tmp_path)
    (day / "data_001.mp4").write_bytes(b"1234")
    output = tmp_path / "inventory.json"
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if path.suffix.lower() == ".mp4":
            raise AssertionError("video content was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    args = ["--site", "Shinjuku", "--source-root", str(root), "--output", str(output)]
    assert main(args) == 0
    first = output.read_bytes()
    first_mtime = output.stat().st_mtime_ns
    assert main(args) == 0
    assert output.read_bytes() == first
    assert output.stat().st_mtime_ns == first_mtime
    assert json.loads(first)["site"] == "Shinjuku"


@pytest.mark.parametrize("folder_name", ["202605", "20260230", "20261301", "notes"])
def test_invalid_date_folder_is_rejected(tmp_path: Path, folder_name: str) -> None:
    root = tmp_path / "videos"
    (root / folder_name).mkdir(parents=True)
    with pytest.raises(ValueError, match="folder"):
        build_inventory(site="Sukiyabashi", source_root=root)


def test_missing_or_empty_root_and_bad_site_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    with pytest.raises(ValueError, match="source root"):
        build_inventory(site="Sukiyabashi", source_root=root)
    root.mkdir()
    with pytest.raises(ValueError, match="no YYYYMMDD folders"):
        build_inventory(site="Sukiyabashi", source_root=root)
    with pytest.raises(ValueError, match="unsupported site"):
        build_inventory(site="Other", source_root=root)


@pytest.mark.parametrize("at_root", [True, False])
def test_unexpected_file_is_rejected(tmp_path: Path, at_root: bool) -> None:
    root, day = _source_tree(tmp_path)
    parent = root if at_root else day
    (parent / "notes.txt").write_text("not video", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        build_inventory(site="Sukiyabashi", source_root=root)


def test_nested_folder_and_media_symlink_are_rejected(tmp_path: Path) -> None:
    root, day = _source_tree(tmp_path)
    (day / "nested").mkdir()
    with pytest.raises(ValueError, match="unexpected date-folder entry"):
        build_inventory(site="Sukiyabashi", source_root=root)
    (day / "nested").rmdir()
    target = tmp_path / "outside.mp4"
    target.write_bytes(b"media")
    try:
        (day / "data_001.mp4").symlink_to(target)
    except OSError as error:
        pytest.skip(f"Symlinks not supported: {error}")
    with pytest.raises(ValueError, match="unexpected date-folder entry"):
        build_inventory(site="Sukiyabashi", source_root=root)


@pytest.mark.parametrize("duplicate_date", [True, False])
def test_duplicate_folder_or_location_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duplicate_date: bool
) -> None:
    root, day = _source_tree(tmp_path)
    (day / "data_001.mp4").write_bytes(b"x")
    duplicated_parent = root if duplicate_date else day
    original_iterdir = Path.iterdir

    def duplicated_iterdir(path: Path):  # type: ignore[no-untyped-def]
        entries = list(original_iterdir(path))
        return iter(entries + entries if path == duplicated_parent else entries)

    monkeypatch.setattr(Path, "iterdir", duplicated_iterdir)
    with pytest.raises(ValueError, match="duplicate"):
        build_inventory(site="Sukiyabashi", source_root=root)


def test_output_inside_source_is_rejected_without_changing_video(tmp_path: Path) -> None:
    root, day = _source_tree(tmp_path)
    video = day / "data_001.mp4"
    video.write_bytes(b"original media")
    with pytest.raises(ValueError, match="outside the source video tree"):
        main(
            [
                "--site",
                "Sukiyabashi",
                "--source-root",
                str(root),
                "--output",
                str(video),
            ]
        )
    assert video.read_bytes() == b"original media"


def test_divergent_existing_output_is_preserved(tmp_path: Path) -> None:
    root, day = _source_tree(tmp_path)
    (day / "data_001.mp4").write_bytes(b"media")
    output = tmp_path / "inventory.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(
            [
                "--site",
                "Sukiyabashi",
                "--source-root",
                str(root),
                "--output",
                str(output),
            ]
        )
    assert output.read_text(encoding="utf-8") == "existing"


def test_output_symlink_is_rejected_without_changing_target(tmp_path: Path) -> None:
    root, day = _source_tree(tmp_path)
    (day / "data_001.mp4").write_bytes(b"media")
    target = tmp_path / "target.json"
    target.write_text("existing", encoding="utf-8")
    output = tmp_path / "inventory.json"
    try:
        output.symlink_to(target)
    except OSError as error:
        pytest.skip(f"Symlinks not supported: {error}")
    with pytest.raises(ValueError, match="symbolic link"):
        main(
            [
                "--site",
                "Sukiyabashi",
                "--source-root",
                str(root),
                "--output",
                str(output),
            ]
        )
    assert target.read_text(encoding="utf-8") == "existing"
