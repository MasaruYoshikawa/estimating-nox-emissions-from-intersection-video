"""Count-agreement and common-rate model diagnostics for the thesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import posixpath
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vehicle_emission.baselines.vtcpfm import VTCPFMModel
from vehicle_emission.emissions.movestar import BASE_DIR, MOVEStarModel
from vehicle_emission.vehicle_classes import emission_vehicle_type_for_project_class

XML_NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
CLASS_NAMES = {0: "Bus", 1: "Car", 2: "Heavy truck", 3: "Light truck"}
CLASS_LABELS = {
    "Bus": "Bus",
    "Car": "Car",
    "Heavy Track": "Heavy truck",
    "Light Track": "Light truck",
}
ENTRY_LABELS = {
    "Enter from Left": "Left",
    "Enter from Right": "Right",
    "Enter from Top": "Top",
    "Enter from bottom": "Bottom",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sheet_cells(path: Path, sheet_name: str) -> list[dict[str, str]]:
    """Read original XLSX cell values without evaluating cached formulas."""
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = [
                "".join(node.itertext())
                for node in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall(
                    "s:si", XML_NS
                )
            ]
        relationships = {
            node.attrib["Id"]: node.attrib["Target"]
            for node in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        sheets = ET.fromstring(archive.read("xl/workbook.xml")).findall("s:sheets/s:sheet", XML_NS)
        selected = [sheet for sheet in sheets if sheet.attrib["name"] == sheet_name]
        if len(selected) != 1:
            raise ValueError(f"expected one {sheet_name} sheet")
        target = relationships[selected[0].attrib[REL_ID]]
        target = target.lstrip("/") if target.startswith("/") else posixpath.join("xl", target)
        target = posixpath.normpath(target)
        if not target.startswith("xl/"):
            raise ValueError("worksheet must be inside the workbook")
        rows = []
        for row in ET.fromstring(archive.read(target)).findall("s:sheetData/s:row", XML_NS):
            cells = {}
            for cell in row.findall("s:c", XML_NS):
                if cell.find("s:f", XML_NS) is not None:
                    continue
                column = re.sub(r"\d+$", "", cell.attrib["r"])
                value = cell.find("s:v", XML_NS)
                if cell.attrib.get("t") == "inlineStr":
                    inline = cell.find("s:is", XML_NS)
                    cells[column] = "".join(inline.itertext()) if inline is not None else ""
                elif value is not None and value.text is not None:
                    cells[column] = (
                        shared[int(value.text)] if cell.attrib.get("t") == "s" else value.text
                    )
            rows.append(cells)
    return rows


def count_records(path: Path, site: str, kind: str) -> pd.DataFrame:
    """Extract manual/automated detail pairs, excluding summary and formula cells."""
    labels = CLASS_LABELS if kind == "classification" else ENTRY_LABELS
    if kind not in {"classification", "entering"}:
        raise ValueError("count kind must be classification or entering")
    rows = read_sheet_cells(path, "Classification" if kind == "classification" else "Entering")
    clip = None
    records = []
    for row in rows:
        label = row.get("A", "").strip()
        if re.fullmatch(r"Video\d+", label):
            clip = int(label.removeprefix("Video"))
        if label not in labels:
            continue
        if clip is None:
            raise ValueError("count pair has no video identity")
        try:
            manual, automated = float(row["B"]), float(row["C"])
        except (KeyError, ValueError) as exc:
            raise ValueError("manual and automated detail counts must be numeric") from exc
        if not all(math.isfinite(x) and x >= 0 and x.is_integer() for x in (manual, automated)):
            raise ValueError("detail counts must be non-negative integers")
        records.append(
            {
                "site": site,
                "kind": kind,
                "clip": clip,
                "category": labels[label],
                "manual_count": int(manual),
                "automated_count": int(automated),
                "signed_difference": int(automated - manual),
                "absolute_difference": int(abs(automated - manual)),
            }
        )
    if not records:
        raise ValueError("no detailed count pairs")
    frame = pd.DataFrame(records)
    if frame.duplicated(["site", "kind", "clip", "category"]).any():
        raise ValueError("duplicate count pair")
    for _, group in frame.groupby("clip"):
        if set(group.category) != set(labels.values()):
            raise ValueError("each video must contain all four detail categories")
    return frame


def count_summary(records: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Aggregate absolute discrepancies before computing count agreement."""
    output = records.groupby(groups, as_index=False, sort=True).agg(
        manual_count=("manual_count", "sum"),
        automated_count=("automated_count", "sum"),
        signed_difference=("signed_difference", "sum"),
        absolute_difference=("absolute_difference", "sum"),
        detail_pairs=("manual_count", "size"),
        clips=("clip", "nunique"),
    )
    output["agreement_percent"] = np.where(
        output.manual_count.gt(0),
        100 * (1 - output.absolute_difference / output.manual_count),
        np.nan,
    )
    return output


def model_parameters() -> pd.DataFrame:
    """Return actual class mapping, road-load constants and fitted NOx parameters."""
    model = VTCPFMModel()
    output = []
    for class_id, name in CLASS_NAMES.items():
        native = emission_vehicle_type_for_project_class(class_id)
        coeff_type = {1: 2, 2: 3}.get(native, native)
        row = model.df_coeff.loc[model.df_coeff.VehicleType.eq(coeff_type)].iloc[0]
        quadratic, linear, intercept = model.coefficients[native][3]
        output.append(
            {
                "class_id": class_id,
                "vehicle_class": name,
                "native_vehicle_type": native,
                "moves_source_type_id": int(row["Source TypeID"]),
                "A": float(row.A),
                "B": float(row.B),
                "C": float(row.C),
                "mass_tonnes": float(row.M),
                "power_divisor_tonnes": float(row.f),
                "alpha_0_g_per_hour": float(intercept),
                "alpha_1_g_per_hour_per_vsp": float(linear),
                "alpha_2_g_per_hour_per_vsp_squared": float(quadratic),
                "idle_nox_g_per_hour": model.idle_rates[native][3],
            }
        )
    return pd.DataFrame(output)


def response_grid(vsp_values: np.ndarray, speeds_kmh: tuple[float, ...]) -> pd.DataFrame:
    """Evaluate actual rate rules at explicit model states, not observed trajectories."""
    if (
        vsp_values.ndim != 1
        or vsp_values.size == 0
        or not speeds_kmh
        or not np.isfinite(vsp_values).all()
        or not all(np.isfinite(s) and s > 0 for s in speeds_kmh)
    ):
        raise ValueError("response states must be finite with positive speeds")
    discrete, continuous = MOVEStarModel(), VTCPFMModel()
    output = []
    for p in model_parameters().to_dict("records"):
        native = int(p["native_vehicle_type"])
        for speed_kmh in speeds_kmh:
            speed = speed_kmh / 3.6
            for vsp in vsp_values:
                acceleration = (
                    vsp * p["power_divisor_tonnes"]
                    - p["A"] * speed
                    - p["B"] * speed**2
                    - p["C"] * speed**3
                ) / (p["mass_tonnes"] * speed)
                modes = discrete.calculate_operating_modes(
                    np.array([speed]), np.array([acceleration]), np.array([vsp])
                )
                rates = discrete.ems_tables[native]
                index = int(np.flatnonzero(modes)[0])
                polynomial = float(np.polyval(continuous.coefficients[native][3], vsp))
                fitted = max(0.0, polynomial) if vsp > 0 else continuous.idle_rates[native][3]
                output.append(
                    {
                        "class_id": p["class_id"],
                        "vehicle_class": p["vehicle_class"],
                        "speed_kmh": speed_kmh,
                        "vsp_kw_per_tonne": float(vsp),
                        "acceleration_mps2": acceleration,
                        "operating_mode": int(rates.iloc[index, 0]),
                        "movestar_nox_mg_per_second": float(rates.iloc[index, 3]) / 3.6,
                        "continuous_nox_mg_per_second": fitted / 3.6,
                        "polynomial_before_clipping_g_per_hour": polynomial,
                    }
                )
    return pd.DataFrame(output)


def quadratic_identity(coefficients: np.ndarray, vsp_values: np.ndarray) -> dict[str, float]:
    """Evaluate the quadratic mean/variance identity on a positive, unclipped branch."""
    if coefficients.shape != (3,) or vsp_values.ndim != 1 or vsp_values.size < 2:
        raise ValueError("three coefficients and at least two states are required")
    if not np.isfinite(coefficients).all() or not np.isfinite(vsp_values).all():
        raise ValueError("quadratic states and coefficients must be finite")
    values = np.polyval(coefficients, vsp_values)
    if not (vsp_values > 0).all() or not (values >= 0).all():
        raise ValueError("identity diagnostic requires positive, unclipped states")
    mean_response = float(np.mean(values))
    response_at_mean = float(np.polyval(coefficients, np.mean(vsp_values)))
    curvature_variance = float(coefficients[0] * np.var(vsp_values, ddof=0))
    return {
        "mean_vsp": float(np.mean(vsp_values)),
        "population_variance_vsp": float(np.var(vsp_values, ddof=0)),
        "mean_polynomial_g_per_hour": mean_response,
        "polynomial_at_mean_g_per_hour": response_at_mean,
        "curvature_variance_g_per_hour": curvature_variance,
        "identity_residual_g_per_hour": mean_response - response_at_mean - curvature_variance,
    }


def worked_hour(raw_directory: Path, hourly_panel: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay the first chronological Sukiyabashi hour as a worked aggregation example."""
    panel = pd.read_csv(hourly_panel)
    hour = panel.loc[panel.site.eq("Sukiyabashi")].sort_values("monitor_start_utc").iloc[0]
    videos = pd.read_csv(raw_directory / "video_master.csv")
    emissions = pd.read_csv(raw_directory / "video_emissions.csv")
    starts = pd.to_datetime(
        videos.recording_datetime_utc, utc=True, format="ISO8601", errors="raise"
    )
    videos = videos.assign(monitor_start_utc=starts.dt.floor("h").dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    clips = videos.loc[
        videos.source_site.eq("Sukiyabashi") & videos.monitor_start_utc.eq(hour.monitor_start_utc)
    ].sort_values("recording_datetime_utc")
    selected = clips.merge(emissions, on=["source_site", "video_id"], validate="one_to_many")
    if set(selected.estimation_method) != {"MOVEStar", "VT-CPFM"}:
        raise ValueError("worked hour requires both recognised model methods")
    if selected.duplicated(["video_id", "estimation_method"]).any():
        raise ValueError("worked hour contains duplicate method records")
    numeric = selected[
        ["vehicle_count", "average_speed_kmh", "nox_grams", "nox_hybrid_adjusted_grams"]
    ]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or not numeric.ge(0).all().all():
        raise ValueError("worked hour requires finite non-negative numeric records")
    summaries = []
    for method, group in selected.groupby("estimation_method", sort=True):
        if len(group) != len(clips):
            raise ValueError("worked hour lacks a model record for a clip")
        raw = float(group.nox_grams.sum())
        adjusted = float(group.nox_hybrid_adjusted_grams.sum())
        expected_column = "movestar_nox_grams" if method == "MOVEStar" else "vt_cpfm_nox_grams"
        if not np.isclose(adjusted, hour[expected_column], rtol=0, atol=1e-8):
            raise ValueError("worked hour does not replay its recorded panel index")
        if int(group.vehicle_count.sum()) != int(hour.vehicle_count):
            raise ValueError("worked hour does not replay its profile volume")
        summaries.append(
            {
                "site": "Sukiyabashi",
                "monitor_start_utc": hour.monitor_start_utc,
                "estimation_method": method,
                "recorded_clips": len(group),
                "recorded_profiles": int(group.vehicle_count.sum()),
                "mean_clip_speed_kmh": float(group.average_speed_kmh.mean()),
                "raw_nox_index_grams": raw,
                "adjusted_nox_index_grams": adjusted,
                "ambient_nox_ppm": float(hour.ambient_nox_ppm),
                "precipitation_mm": float(hour.source_interval_precipitation_mm),
            }
        )
    columns = [
        "video_id",
        "recording_datetime_utc",
        "estimation_method",
        "vehicle_count",
        "average_speed_kmh",
        "nox_grams",
        "nox_hybrid_adjusted_grams",
    ]
    return selected[columns].reset_index(drop=True), pd.DataFrame(summaries)


def _response_plot(grid: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True)
    for ax, (class_id, name) in zip(axes.flat, CLASS_NAMES.items(), strict=True):
        group = grid.loc[grid.class_id.eq(class_id)]
        for speed, rows in group.groupby("speed_kmh", sort=True):
            ax.step(
                rows.vsp_kw_per_tonne,
                rows.movestar_nox_mg_per_second,
                where="post",
                label=f"MOVEStar, {speed:g} km/h",
                linewidth=1.4,
            )
        rows = group.loc[group.speed_kmh.eq(group.speed_kmh.min())]
        ax.plot(
            rows.vsp_kw_per_tonne,
            rows.continuous_nox_mg_per_second,
            label="VT-CPFM",
            color="#207668",
            linewidth=1.6,
        )
        ax.set(title=name, xlabel="VSP (kW/tonne)", ylabel="NOx rate (mg/s)")
        ax.title.set_fontsize(12)
        ax.xaxis.label.set_fontsize(11)
        ax.yaxis.label.set_fontsize(11)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def run(repository_root: Path, raw_directory: Path, hourly_panel: Path, output: Path) -> None:
    """Calculate a fresh, source-bound set of measurement and model diagnostics."""
    if output.exists():
        raise FileExistsError("diagnostic output directory must be fresh")
    workbook_paths = {
        site: repository_root / "results" / site / "manual_vs_cv_counting_validation.xlsx"
        for site in ("Sukiyabashi", "Shinjuku")
    }
    inputs = {
        **{f"workbook_{s}": p for s, p in workbook_paths.items()},
        "video_master": raw_directory / "video_master.csv",
        "video_emissions": raw_directory / "video_emissions.csv",
        "hourly_panel": hourly_panel,
    }
    all_records = pd.concat(
        [
            count_records(path, site, kind)
            for site, path in workbook_paths.items()
            for kind in ("classification", "entering")
        ],
        ignore_index=True,
    )
    parameters = model_parameters()
    grid = response_grid(np.arange(-5.0, 35.0001, 0.25), (25.0, 60.0))
    model = VTCPFMModel()
    identities = pd.DataFrame(
        [
            {
                "class_id": class_id,
                "vehicle_class": name,
                **quadratic_identity(
                    model.coefficients[emission_vehicle_type_for_project_class(class_id)][3],
                    np.array([3.0, 5.0, 7.0]),
                ),
            }
            for class_id, name in CLASS_NAMES.items()
        ]
    )
    example_clips, example_summary = worked_hour(raw_directory, hourly_panel)
    output.mkdir(parents=True)
    frames = {
        "count_detail_pairs.csv": all_records,
        "count_site_summary.csv": count_summary(all_records, ["site", "kind"]),
        "count_category_summary.csv": count_summary(all_records, ["site", "kind", "category"]),
        "count_clip_summary.csv": count_summary(all_records, ["site", "kind", "clip"]),
        "model_parameters.csv": parameters,
        "model_response_states.csv": grid,
        "quadratic_aggregation_check.csv": identities,
        "worked_hour_clips.csv": example_clips,
        "worked_hour_summary.csv": example_summary,
    }
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False, float_format="%.17g", lineterminator="\n")
    _response_plot(grid, output / "fig_class_response_functions.png")
    emission_dir = Path(BASE_DIR)
    code_paths = [
        Path(__file__),
        emission_dir / "movestar.py",
        emission_dir.parent / "baselines" / "vtcpfm.py",
        emission_dir.parent / "vehicle_classes.py",
    ]
    model_inputs = [
        emission_dir / "VehicleSrcCoeff.csv",
        *sorted(emission_dir.glob("EmsRate_*.csv")),
    ]
    receipt = {
        "status": "recorded_count_agreement_and_constructed_model_diagnostics",
        "count_columns": {"manual": "B", "automated": "C"},
        "count_metric": "100 * (1 - sum(abs(automated - manual)) / sum(manual))",
        "zero_manual_total": "agreement_unavailable",
        "response_grid": {
            "vsp_min": -5,
            "vsp_max": 35,
            "vsp_step": 0.25,
            "speeds_kmh": [25, 60],
            "scope": "explicit_model_states_not_observed_occupancy",
        },
        "quadratic_identity_states": [3, 5, 7],
        "inputs": {
            key: {
                "path": (
                    str(p.resolve().relative_to(repository_root.resolve()))
                    if p.resolve().is_relative_to(repository_root.resolve())
                    else str(p.resolve())
                ),
                "sha256": _hash(p),
            }
            for key, p in inputs.items()
        },
        "model_input_sha256": {p.name: _hash(p) for p in model_inputs},
        "calculator_sha256": {p.name: _hash(p) for p in code_paths},
        "outputs": {p.name: _hash(p) for p in sorted(output.iterdir())},
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> None:
    """Run source-workbook and common-rate diagnostics without changing inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--raw-source-directory", type=Path, required=True)
    parser.add_argument("--hourly-panel", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run(args.repository_root, args.raw_source_directory, args.hourly_panel, args.output_directory)


if __name__ == "__main__":
    main()
