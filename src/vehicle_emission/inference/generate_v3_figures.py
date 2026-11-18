"""Render provisional analysis components from computed rows and a hashed receipt.

The historical module name remains a CLI entry point. Estimands follow the
project glossary; this module performs no statistical analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

SITES = ("Sukiyabashi", "Shinjuku")
SITE_ESTIMANDS = ("rq1_index", "rq1_count", "rq3_paired_advantage", "rq2_wind_contrast")
SCOPE = "locked_retrospective_two_site"
EVIDENCE_STATUS = "provisional_components_only"
REQUIRED_COLUMNS = {
    "site",
    "estimand",
    "estimate",
    "ci_low",
    "ci_high",
    "n_hours",
    "n_days",
    "resamples_requested",
    "resamples_valid",
    "interval_method",
    "interval_adjustment",
    "interval_family",
    "confidence_level",
    "family_size",
    "classification",
    "analysis_scope",
    "evidence_status",
    "observation_sha256",
    "resample_sha256",
    "failure_reason",
}
FIGURE_NAMES = (
    "fig_rq1_correlations_components_v1.png",
    "fig_rq3_paired_advantage_components_v1.png",
    "fig_rq2_wind_contrast_components_v1.png",
    "fig_cross_site_heterogeneity_components_v1.png",
)
RECEIPT_NAME = "figure_receipt_components_v1.json"


class FigureInputError(ValueError):
    """The result table or its receipt cannot support a figure."""


@dataclass(frozen=True)
class FigureRow:
    """One computed estimand and its recorded support."""

    site: str
    estimand: str
    estimate: float
    ci_low: float
    ci_high: float
    n_hours: int
    n_days: int
    confidence_level: float
    family_size: int
    classification: str
    failure_reason: str
    observation_sha256: str
    resample_sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise FigureInputError(f"{field} must be a lowercase SHA-256")
    return value


def _integer(value: str, field: str, minimum: int = 0) -> int:
    if re.fullmatch(r"\d+", value) is None or int(value) < minimum:
        raise FigureInputError(f"{field} must be an integer >= {minimum}")
    return int(value)


def _number(value: str, field: str, *, allow_missing: bool = False) -> float:
    if allow_missing and value.strip().lower() in {"", "nan"}:
        return math.nan
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FigureInputError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise FigureInputError(f"{field} must be finite")
    return result


def load_figure_inputs(results_csv: Path, receipt_path: Path) -> tuple[FigureRow, ...]:
    """Verify computed bytes, receipt metadata and paired support identities."""
    if not results_csv.is_file() or not receipt_path.is_file():
        raise FigureInputError("explicit results CSV and receipt files are required")
    result_bytes = results_csv.read_bytes()
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FigureInputError("receipt must contain valid JSON") from exc
    if not isinstance(receipt, dict):
        raise FigureInputError("receipt must be a JSON object")
    if receipt.get("schema_version") != 1 or receipt.get("artifact_kind") != (
        "paired_inference_component"
    ):
        raise FigureInputError("receipt is not a paired inference component receipt")
    for field, expected in (("analysis_scope", SCOPE), ("evidence_status", EVIDENCE_STATUS)):
        if receipt.get(field) != expected:
            raise FigureInputError(f"receipt {field} must be {expected}")
    if _require_hash(receipt.get("results_sha256"), "results_sha256") != (
        hashlib.sha256(result_bytes).hexdigest()
    ):
        raise FigureInputError("result bytes do not match receipt results_sha256")
    input_sha256 = _require_hash(receipt.get("input_sha256"), "input_sha256")
    inputs = receipt.get("inputs")
    if (
        not isinstance(inputs, list)
        or not inputs
        or not all(isinstance(item, dict) for item in inputs)
    ):
        raise FigureInputError("receipt inputs must contain the recorded source inventories")
    if (
        hashlib.sha256(
            json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        != input_sha256
    ):
        raise FigureInputError("receipt inputs do not match input_sha256")
    config_sha256 = _require_hash(receipt.get("config_sha256"), "config_sha256")
    config = receipt.get("config")
    if not isinstance(config, dict):
        raise FigureInputError("receipt config must be a JSON object")
    try:
        config_bytes = json.dumps(
            config, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except ValueError as exc:
        raise FigureInputError("receipt config contains non-finite values") from exc
    if hashlib.sha256(config_bytes).hexdigest() != config_sha256:
        raise FigureInputError("receipt config does not match config_sha256")

    with io.StringIO(result_bytes.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise FigureInputError("computed result table is missing required columns")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise FigureInputError("computed result table has duplicate columns")
        records = list(reader)
    if type(receipt.get("row_count")) is not int or receipt["row_count"] != len(records):
        raise FigureInputError("receipt row_count does not match computed rows")
    expected_rows = {(site, estimand) for site in SITES for estimand in SITE_ESTIMANDS}
    expected_rows.add(("cross_site", "cross_site_index_heterogeneity"))
    seen: set[tuple[str, str]] = set()
    rows = []
    for record in records:
        if None in record or any(value is None for value in record.values()):
            raise FigureInputError("computed result table has malformed rows")
        key = (record["site"], record["estimand"])
        if key not in expected_rows or key in seen:
            raise FigureInputError(f"unexpected or duplicate result row: {key}")
        seen.add(key)
        if record["analysis_scope"] != SCOPE or record["evidence_status"] != EVIDENCE_STATUS:
            raise FigureInputError("all figure rows must retain provisional retrospective status")
        if record["interval_method"] != "union_calendar_moving_block_percentile":
            raise FigureInputError("unsupported component interval_method")
        if record["interval_adjustment"] != "bonferroni":
            raise FigureInputError("unsupported component interval_adjustment")
        if not record["interval_family"] or not record["classification"]:
            raise FigureInputError("interval_family and classification are required")
        if "confirm" in record["classification"].lower():
            raise FigureInputError("a provisional result cannot be classified as confirmed")
        n_hours = _integer(record["n_hours"], "n_hours")
        n_days = _integer(record["n_days"], "n_days")
        if n_days > n_hours:
            raise FigureInputError("n_days cannot exceed n_hours")
        requested = _integer(record["resamples_requested"], "resamples_requested", 1)
        valid = _integer(record["resamples_valid"], "resamples_valid")
        if valid > requested:
            raise FigureInputError("resamples_valid cannot exceed resamples_requested")
        family_size = _integer(record["family_size"], "family_size", 1)
        confidence_level = _number(record["confidence_level"], "confidence_level")
        if not 0 < confidence_level < 1:
            raise FigureInputError("confidence_level must lie between zero and one")
        missing = record["classification"] == "not_estimable"
        estimate = _number(record["estimate"], "estimate", allow_missing=missing)
        low = _number(record["ci_low"], "ci_low", allow_missing=missing)
        high = _number(record["ci_high"], "ci_high", allow_missing=missing)
        if missing:
            if not record["failure_reason"] or not (math.isnan(low) and math.isnan(high)):
                raise FigureInputError("not_estimable rows need a reason and absent interval")
        elif n_hours == 0 or n_days == 0 or valid == 0 or record["failure_reason"]:
            raise FigureInputError("estimable rows require support, valid resamples and no failure")
        bound = 1 if record["estimand"] in {"rq1_index", "rq1_count"} else 2
        for value in (estimate, low, high):
            if math.isfinite(value) and not -bound <= value <= bound:
                raise FigureInputError("result value exceeds its estimand bounds")
        if math.isfinite(low) and low > high:
            raise FigureInputError("interval bounds are reversed")
        rows.append(
            FigureRow(
                *key,
                estimate,
                low,
                high,
                n_hours,
                n_days,
                confidence_level,
                family_size,
                record["classification"],
                record["failure_reason"],
                _require_hash(record["observation_sha256"], "observation_sha256"),
                _require_hash(record["resample_sha256"], "resample_sha256"),
            )
        )
    if seen != expected_rows:
        raise FigureInputError("computed results must contain all nine RQ/site rows")
    if len({row.resample_sha256 for row in rows}) != 1:
        raise FigureInputError("all rows must identify the shared resampling plan")
    for site in SITES:
        paired = [row for row in rows if row.site == site and row.estimand in SITE_ESTIMANDS[:3]]
        if len({(row.n_hours, row.n_days, row.observation_sha256) for row in paired}) != 1:
            raise FigureInputError("RQ1 and RQ3 rows must identify identical paired support")
        by_estimand = {row.estimand: row for row in paired}
        index = by_estimand["rq1_index"].estimate
        count = by_estimand["rq1_count"].estimate
        advantage = by_estimand["rq3_paired_advantage"].estimate
        if all(math.isfinite(value) for value in (index, count, advantage)) and not math.isclose(
            advantage, index - count, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise FigureInputError("RQ3 estimate must equal index minus count correlation")
    associations = {row.site: row.estimate for row in rows if row.estimand == "rq1_index"}
    heterogeneity = next(row.estimate for row in rows if row.site == "cross_site")
    if all(math.isfinite(value) for value in (*associations.values(), heterogeneity)) and not (
        math.isclose(
            heterogeneity,
            associations["Shinjuku"] - associations["Sukiyabashi"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise FigureInputError("heterogeneity must use Shinjuku minus Sukiyabashi")
    return tuple(rows)


def _forest(rows: tuple[FigureRow, ...], title: str, xlabel: str) -> Figure:
    fig, ax = plt.subplots(figsize=(10.0, max(3.2, 1.0 + len(rows) * 0.8)))
    labels = []
    for position, row in enumerate(rows):
        label = row.site.replace("cross_site", "Shinjuku minus Sukiyabashi")
        if row.estimand in {"rq1_index", "rq1_count"}:
            label += " / " + ("Index" if row.estimand == "rq1_index" else "Count")
        labels.append(label + f"\n{row.n_hours} hours; {row.n_days} days")
        if row.classification == "not_estimable":
            ax.text(
                0.02,
                position,
                "Not estimable: " + row.failure_reason.replace("_", " "),
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=8,
            )
            continue
        color = "#286d9f" if row.site == SITES[0] else "#b55b27"
        ax.plot([row.ci_low, row.ci_high], [position, position], color=color, linewidth=2)
        ax.plot(row.estimate, position, "o", color=color, markersize=6)
        ax.annotate(
            f"{row.estimate:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}] "
            f"({1 - (1 - row.confidence_level) / row.family_size:.2%} Bonferroni interval)\n"
            + row.classification.replace("_", " "),
            (row.estimate, position),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=8,
            ha="center",
        )
    ax.set_yticks(range(len(rows)), labels)
    ax.set_ylim(len(rows) - 0.5, -0.65)
    ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_title(title + " — provisional analysis components", pad=16)
    ax.grid(axis="x", alpha=0.2)
    if all(row.classification == "not_estimable" for row in rows):
        ax.set_xlim(-1, 1)
    else:
        ax.margins(x=0.2)
    fig.text(
        0.5,
        0.015,
        "Retrospective two-site evaluation; source and measurement validation pending.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def build_figures(rows: tuple[FigureRow, ...]) -> dict[str, Figure]:
    """Build four RQ displays using only supplied computed rows."""
    definitions = (
        (("rq1_index", "rq1_count"), "RQ1: Ambient NOx associations", "Spearman correlation"),
        (
            ("rq3_paired_advantage",),
            "RQ3: Paired association advantage over counts",
            "Index correlation minus count correlation",
        ),
        (
            ("rq2_wind_contrast",),
            "RQ2: Wind moderation contrast",
            "Favourable minus opposing index correlation",
        ),
        (
            ("cross_site_index_heterogeneity",),
            "Cross-site association heterogeneity (descriptive)",
            "Shinjuku minus Sukiyabashi index correlation",
        ),
    )
    figures = {}
    for name, (estimands, title, xlabel) in zip(FIGURE_NAMES, definitions, strict=True):
        selected = tuple(
            sorted(
                (row for row in rows if row.estimand in estimands),
                key=lambda row: (SITES.index(row.site) if row.site in SITES else 2, row.estimand),
            )
        )
        figures[name] = _forest(selected, title, xlabel)
    return figures


def _check_output_destination(out_dir: Path) -> None:
    if out_dir.exists() or out_dir.is_symlink():
        raise FigureInputError("output directory must be new; existing artifacts are immutable")
    ancestor = out_dir.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ancestor,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        root = Path(result.stdout.strip()).resolve()
        relative = out_dir.resolve().relative_to(root)
        if not relative.parts or relative.parts[0] != "build":
            raise FigureInputError("repository figure outputs must use a new directory under build")
        tracked = subprocess.run(
            ["git", "ls-files", "--", relative.as_posix()],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        if tracked.stdout:
            raise FigureInputError("output destination contains tracked artifacts")


def generate_figures(results_csv: Path, receipt_path: Path, out_dir: Path) -> list[Path]:
    """Write new component figures and a receipt without replacing artifacts."""
    if not results_csv.is_file() or not receipt_path.is_file():
        raise FigureInputError("explicit results CSV and receipt files are required")
    result_hash = _sha256(results_csv)
    receipt_hash = _sha256(receipt_path)
    rows = load_figure_inputs(results_csv, receipt_path)
    _check_output_destination(out_dir)
    out_dir = out_dir.resolve()
    figures = build_figures(rows)
    try:
        if result_hash != _sha256(results_csv) or receipt_hash != _sha256(receipt_path):
            raise FigureInputError("figure input bytes changed during generation")
        out_dir.mkdir(parents=True, exist_ok=False)
        paths = []
        for name, fig in figures.items():
            path = out_dir / name
            with path.open("xb") as output:
                fig.savefig(output, format="png", dpi=180)
            paths.append(path)
        figure_receipt = {
            "schema_version": 1,
            "artifact_kind": "computed_component_figures",
            "analysis_scope": SCOPE,
            "evidence_status": EVIDENCE_STATUS,
            "results_sha256": result_hash,
            "analysis_receipt_sha256": receipt_hash,
            "result_row_count": len(rows),
            "artifacts": [{"path": path.name, "sha256": _sha256(path)} for path in paths],
        }
        with (out_dir / RECEIPT_NAME).open("x", encoding="utf-8") as stream:
            json.dump(figure_receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    finally:
        for fig in figures.values():
            plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    """Render an explicitly selected computed table and its receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = generate_figures(args.results_csv, args.receipt, args.out_dir)
    except (FigureInputError, OSError) as exc:
        parser.error(str(exc))
    print(f"Wrote {len(paths)} provisional component figures to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
