# Estimating NOx Emissions from Intersection Traffic Video

Code, verification suite, and preserved numerical outputs accompanying the
Master's thesis:

> *Video-Based Vehicle NOx Emission Estimation at Two Tokyo Intersections Evaluated Against Ambient Air Quality Monitoring*
> Masaru Yoshikawa, Unitec Institute of Technology (November 2026)

[![CI](https://github.com/MasaruYoshikawa/estimating-nox-emissions-from-intersection-video/actions/workflows/ci.yml/badge.svg)](https://github.com/MasaruYoshikawa/estimating-nox-emissions-from-intersection-video/actions/workflows/ci.yml)

---

## Overview

This repository implements an end-to-end comparative pipeline evaluating whether
video-derived microscopic vehicle $\text{NO}_x$ indices provide stronger hourly
rank associations with urban ambient air quality monitoring than unweighted and
class-weighted traffic counts.

- **Sites & Period**: Sukiyabashi (Ginza, paired with Kasumigaseki station) and
  Shinjuku O-Guard West (paired with Shinjuku Gyoen station), with the Setagaya
  residential general-ambient station providing the regional background series;
  May–June 2026 (`1,148` dry-weather hours, `09:00`–`20:00 JST` hourly starts
  across `61` calendar dates).
- **Computer Vision**: Fine-tuned **RT-DETR** four-class detector (car, light
  truck, bus, heavy truck) + **ByteTrack** multi-object tracker + virtual
  entry-line crossing counters + single-chord pixel-to-metre speed scaling.
- **Emission Models (Common MOVES Rate Basis)**:
  1. **MOVEStar**: Discrete speed–VSP operating-mode lookup calculation.
  2. **Continuous MOVES-derived model (VT-CPFM-inspired)**: Class-specific
     quadratic polynomials fitted to the same MOVES 5.0 $\text{NO}_x$ rates,
     with an explicit idle branch and non-negative clipping.
- **Statistical Evaluation**: Paired Spearman rank correlations and contrasts
  under shared 1-day cluster and 7-day calendar-block resampling (`20,000`
  core draws; ten-estimand Bonferroni family adjustment), fixed class-weighted
  entry benchmarks, staged rank-residual projections, between-clip speed
  volatility (`CV_speed`), and calendar-shift diagnostics.

---

## Repository Structure

```text
├── src/vehicle_emission/       # Detection, tracking, calibration, MOVEStar & continuous models, paired bootstrap inference
├── scripts/                    # CLI helpers for figure refresh, ordering plots, and precision design
├── tests/                      # Unit, integration, contract, and deterministic figure/diagnostic tests
├── data/
│   ├── airpollution_history/   # Hourly ambient NOx monitor records (Soramame-kun)
│   ├── weather_history/        # Hourly JMA meteorological records
│   ├── calibration/            # Camera pixel-to-metre landmark calibration configs
│   └── manifests/              # Pipeline & analysis configurations
├── results/                    # Preserved thesis numeric bundles, SHA-256 receipts, figures, and manual validation audits
├── docs/                       # Reproduction, robustness, and numeric source-export guides
├── pyproject.toml              # Package metadata and pinned CPU/vision/dev dependencies
├── environment.yml             # Portable Conda Python 3.11 environment spec
├── requirements-dev.lock       # Resolved CPU/test dependency lockfile
├── CITATION.cff                # Machine-readable citation metadata
└── LICENSE                     # MIT License
```

---

## Installation

Python **3.11** is required. Create and activate an isolated environment:

```bash
conda env create -f environment.yml
conda activate vehicle-emission
```

Or with `venv` + `pip`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip==25.0.1
pip install -r requirements-dev.lock
pip install --no-deps -e .
```

*(Optional GPU inference dependencies for video detection/tracking: `pip install -e .[vision]`.)*

---

## Quick Verification & Smoke Runs

Run linting and the automated test suite:

```bash
ruff check src tests
black --check src tests
pytest
```

Run the synthetic smoke pipelines and deterministic diagnostics (writing to a
fresh directory under `build/`):

```bash
PYTHONPATH=src python -m vehicle_emission.reproduction smoke --output-dir build/pipeline_smoke
PYTHONPATH=src python -m vehicle_emission.inference.v4_smoke --output-directory build/v4_controlled_smoke
PYTHONPATH=src python -m vehicle_emission.inference.model_fit_diagnostics --output-directory build/model_fit_replay
```

Reproduce the manuscript figures directly from the preserved bundles in `results/`:

```bash
PYTHONPATH=src python -m scripts.refresh_thesis_figures \
  --extension-directory results/recorded_hourly_extension_20261109 \
  --measurement-directory results/measurement_model_extension_20261109 \
  --ordering-directory results/model_ordering_20261109 \
  --profile-image results/thesis_figure_refresh_20261117/fig_model_profile_response_checked.png \
  --output-directory build/thesis_figure_refresh
```

See [docs/thesis-reproduction.md](docs/thesis-reproduction.md) and
[docs/kinematics-robustness.md](docs/kinematics-robustness.md) for the full
re-analysis commands from an exported numeric bundle (`SOURCE_BUNDLE`).

---

## Data Availability

- **Included in this repository**: Hourly ambient $\text{NO}_x$ and JMA
  meteorological histories (`data/airpollution_history/`, `data/weather_history/`),
  landmark camera calibration parameters (`data/calibration/`), EPA MOVES 5.0
  extracted lookup tables (`src/vehicle_emission/emissions/`), manual 12-clip
  count audits (`results/*/manual_vs_cv_counting_validation.xlsx`), and all
  derived thesis numeric result bundles with SHA-256 provenance receipts (`results/`).
- **Excluded from version control**: Raw intersection video recordings
  (`*.mp4`), trained neural-network checkpoint weights (`*.pt`), and full
  per-trajectory SQLite databases (`*.db`) are excluded due to file size and
  privacy/broadcast terms of the public livestreams. All statistical comparisons,
  paired bootstrap intervals, sensitivity checks, and publication figures can be
  audited and replayed from the preserved numeric bundles without raw video.

---

## License

Released under the [MIT License](LICENSE).

## Citation

If you use this codebase or preserved outputs, please use GitHub's **"Cite this
repository"** button (`CITATION.cff`) or cite the Master's thesis:

```bibtex
@mastersthesis{yoshikawa2026estimating_nox_video,
  title   = {Video-Based Vehicle NOx Emission Estimation at Two Tokyo Intersections Evaluated Against Ambient Air Quality Monitoring},
  author  = {Masaru Yoshikawa},
  school  = {Unitec Institute of Technology},
  year    = {2026},
  month   = Nov,
  type    = {Master's thesis},
  address = {Auckland, New Zealand}
}
```
