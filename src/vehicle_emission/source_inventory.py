"""Export a local video tree as metadata for the asset-lineage checker."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from vehicle_emission.asset_lineage import SITES


class VideoFile(TypedDict):
    folder_date: str
    name: str
    size_bytes: int


class VideoInventory(TypedDict):
    schema_version: int
    site: str
    files: list[VideoFile]


def _date_folder(name: str) -> str:
    if not re.fullmatch(r"[0-9]{8}", name):
        raise ValueError(f"expected a YYYYMMDD folder, found {name!r}")
    try:
        datetime.strptime(name, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"invalid calendar date folder: {name!r}") from exc
    return name


def build_inventory(*, site: str, source_root: Path | Sequence[Path]) -> VideoInventory:
    """List dated source MP4 names and sizes without opening their contents."""

    if site not in SITES:
        raise ValueError(f"unsupported site: {site!r}")
    roots = [source_root] if isinstance(source_root, Path) else list(source_root)
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"source root is not a directory: {root}")

    files: list[VideoFile] = []
    seen_folders: set[str] = set()
    seen_locations: set[tuple[str, str]] = set()
    for root in roots:
        for folder in sorted(root.iterdir(), key=lambda path: path.name):
            if (
                folder.name in {".DS_Store", "desktop.ini"}
                and folder.is_file()
                and not folder.is_symlink()
            ):
                continue
            if folder.is_symlink() or not folder.is_dir():
                raise ValueError(f"unexpected source-root entry: {folder}")
            folder_date = _date_folder(folder.name)
            if folder_date in seen_folders:
                raise ValueError(f"duplicate date folder: {folder_date}")
            seen_folders.add(folder_date)

            for clip in sorted(folder.iterdir(), key=lambda path: path.name):
                if (
                    clip.name in {".DS_Store", "desktop.ini"}
                    and clip.is_file()
                    and not clip.is_symlink()
                ):
                    continue
                if clip.is_symlink() or not clip.is_file() or clip.suffix.lower() != ".mp4":
                    raise ValueError(f"unexpected date-folder entry: {clip}")
                location = (folder_date, clip.name)
                if location in seen_locations:
                    raise ValueError(f"duplicate source location: {folder_date}/{clip.name}")
                seen_locations.add(location)
                files.append(
                    {
                        "folder_date": folder_date,
                        "name": clip.name,
                        "size_bytes": clip.stat().st_size,
                    }
                )

    if not seen_folders:
        raise ValueError("source roots have no YYYYMMDD folders")
    return {"schema_version": 1, "site": site, "files": files}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=sorted(SITES), required=True)
    parser.add_argument("--source-root", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    roots = [path.resolve() for path in args.source_root]
    output = args.output.resolve()
    for root in roots:
        if output == root or root in output.parents:
            raise ValueError("output must be outside the source video tree")
    if args.output.is_symlink():
        raise ValueError("output must not be a symbolic link")

    inventory = build_inventory(site=args.site, source_root=args.source_root)
    output_bytes = (json.dumps(inventory, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if args.output.exists():
        if args.output.read_bytes() != output_bytes:
            raise FileExistsError(f"refusing to overwrite a different inventory: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            handle.write(output_bytes)
    print(f"{args.site}: {len(inventory['files'])} MP4 files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
