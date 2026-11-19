# Estimating NOx Emissions from Intersection Traffic Video

Code and numerical outputs for Masaru Yoshikawa's Master's thesis at Unitec
Institute of Technology, November 2026:

> *Video-Based Vehicle NOx Emission Estimation at Two Tokyo Intersections
> Evaluated Against Ambient Air Quality Monitoring*

[![CI](https://github.com/MasaruYoshikawa/estimating-nox-emissions-from-intersection-video/actions/workflows/ci.yml/badge.svg)](https://github.com/MasaruYoshikawa/estimating-nox-emissions-from-intersection-video/actions/workflows/ci.yml)

## Study

The study compares video-derived relative NOx emission indices with traffic
counts and hourly ambient observations at Sukiyabashi and Shinjuku O-Guard West
in May–June 2026. The receptor stations are Kasumigaseki and National Shinjuku;
Setagaya supplies a regional comparison series. The models are the discrete
MOVEStar calculation and a continuous MOVES-derived, VT-CPFM-inspired NOx
adaptation. Both use the same rate basis.

The quality-corrected dry-weather cohort contains 590 Sukiyabashi and 550
Shinjuku hours across 61 calendar dates, with hourly starts from 09:00 to
20:00 JST.

Statistical comparisons use identical hours and shared calendar resamples:
20,000 core draws, one-/seven-day blocks and a ten-estimand adjusted interval
family. Descriptive extensions use 2,000 draws. Sample sizes and monitoring
quality decisions are recorded in each result bundle.

## Install

Use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==25.0.1
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
Alternatively, use `conda env create -f environment.yml` followed by
`conda activate vehicle-emission`. Optional video-inference dependencies are
installed with `python -m pip install -e '.[vision]'`.

## Recalculate the published statistics

After installation, run:

```bash
make PYTHON=python replay-statistics
```

`make PYTHON=python all` is the same command. It recalculates the paired
statistics and descriptive tables from the included quality-corrected hourly
panel into `build/panel_replay/`. No video, database, detector checkpoint or
network request is needed. Output directories must be fresh; use
`REPLAY_OUTPUT=build/panel_replay_02` for another run.

The [calculation guide](docs/thesis-reproduction.md) distinguishes this public
statistical replay from source reconstruction and explains the output files.

## Verify the software

```bash
python -m ruff check src tests
python -m black --check src tests
python -m pytest
```

Run a controlled pipeline smoke or model-response diagnostic separately:

```bash
make PYTHON=python pipeline-smoke
make PYTHON=python pipeline-v4-smoke
make PYTHON=python thesis-model-fit
```

These are software/constructed-state checks, not new empirical findings.

## Reproduce figures

The current figures can be regenerated without private source data:

```bash
make PYTHON=python thesis-figure-refresh
```

See [results/README.md](results/README.md) for current and historical bundles.
Historical receipts and output bytes are retained; new calculations use new
directories.

The portable inventory is `results/publication_manifest_20261119.json`. Check
the package's recorded file identities with
`python -m scripts.publication_manifest --check`.

## Included data and source-dependent tools

The repository includes hourly monitoring and weather histories, prepared
analysis panels, extracted emission-rate tables, aggregate manual count audits,
recorded scalar calibration parameters and result receipts. See
[data/README.md](data/README.md) for sources, units, time conventions and quality
handling, and [data/calibration/README.md](data/calibration/README.md) for the
camera parameters actually recorded for this study.

Raw videos, trained weights, per-trajectory databases and the numeric source
export are not included. Source-export, source-analysis, calibrated-speed and
detector-log commands are optional tools for users who hold their required
inputs; they are not fresh-copy reproduction steps. Their inputs are described
in the [source-export guide](docs/source-export.md) and
[speed/robustness guide](docs/kinematics-robustness.md).

Legacy unverified examples are isolated in `tests/fixtures/legacy/`. Historical
fixed-value writers are outside the installed package in `archive/legacy/`.
Neither supplies the current statistical results.

## Attribution and citation

Project-authored software is distributed under the [MIT License](LICENSE).
Third-party model and data attribution is recorded in [NOTICE.md](NOTICE.md);
the project licence does not replace providers' terms.

Use GitHub's **Cite this repository** button or the following thesis reference:

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

For code reproducibility, record the exact commit used. The existing
`v1.0.0-thesis` tag identifies the initial publication snapshot, not subsequent
updates on `main`.
