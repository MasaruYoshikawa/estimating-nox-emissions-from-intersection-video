# Reproduce the thesis calculations

## Published-panel replay

Install the pinned environment described in the root README, then run:

```bash
make PYTHON=python replay-statistics
```

This recalculates the statistical tables from the included
`results/monitor_quality_reanalysis_20261119/extension_hourly_panel.csv`.
The panel contains the recorded model aggregates, counts/classes, monitoring
observations, weather controls and between-clip speed summaries. No database,
raw video, trained weights or external download is required for this replay.

The output directory is `build/panel_replay/`. Use a different
`REPLAY_OUTPUT` for subsequent runs. The command refuses to overwrite an
existing result directory. The receipt binds the input panel, settings,
calculator code and output files. Historical receipts retain their original
paths and hashes; they are not rewritten to describe this new run.

An explicit command can also compare recalculated statistical CSVs with the
published bundle:

```bash
python -m vehicle_emission.inference.panel_replay \
  --panel results/monitor_quality_reanalysis_20261119/extension_hourly_panel.csv \
  --output-directory build/panel_replay_checked \
  --expected-bundle results/monitor_quality_reanalysis_20261119
```

Defaults retain seed 42, 20,000 core draws, 2,000 descriptive extension draws
and separate one-/seven-day calendar blocks. Core intervals retain the
ten-estimand Bonferroni family. Descriptive extension intervals are pointwise;
they are not additional members of that core family. Parse numeric panels
with round-trip float precision and round only for presentation.

For a fast execution smoke, use the following separate output. Its reduced
draw count is not a replay of the published interval endpoints:

```bash
make PYTHON=python replay-statistics REPLAY_OUTPUT=build/panel_smoke \
  REPLAY_CORE_RESAMPLES=50 REPLAY_EXTENSION_RESAMPLES=50
```

## Figures and available diagnostics

The quality correction excludes eight receptor site-hours identified as
missing by the provider and leaves nine missing Setagaya timestamps unavailable
in the regional comparisons (18 site-hour rows). Valid numerical observations,
including genuine zeros, remain unchanged. The paired sample is 590/550 hours;
the common residual sample is 578/539. Source responses and exact interval
mappings are preserved in `results/monitor_quality_sources_20261119/`,
`results/setagaya_quality_sources_20261119/` and
`results/monitor_quality_preparation_20261119/`.

```bash
make PYTHON=python thesis-figure-refresh
make PYTHON=python thesis-weighted-audit
make PYTHON=python thesis-model-fit
```

These commands use included numeric bundles. The figure refresh reads the
preserved panel/statistical tables and constructed-profile image; it does not
re-estimate intervals. Model-fit diagnostics calculate the twenty construction
points, a fixed VSP response grid and symmetric Mode 40 rate-input substitutions.
The weighted count audit recomputes clip–class discrepancies from the included
manual/CV comparison tables. Original output directories remain unchanged.

[results/README.md](../results/README.md) lists current and archived bundles.
[data/README.md](../data/README.md) describes station identities, units, quality
flags and time alignment.

## Source reconstruction: additional inputs required

The following tools are included for the source host. They are not fresh-copy
reproduction steps because the database, numeric source export, source tracks
and development logs are not published here.

| Calculation | Additional inputs |
|---|---|
| `source-export` | Existing closed source SQLite snapshot |
| `source-reconcile` | Numeric export with its manifest |
| `thesis-analysis`, `thesis-extension` | Numeric export, prepared all-weather panel and matching calculation bundle |
| `thesis-diagnostics`, `thesis-timing-weights`, `thesis-recording-support` | Numeric export and matching hourly/measurement bundles |
| `source-analysis` | Explicit v4 analysis configuration and its declared source/measurement evidence |
| `calibrated-speeds`, `robustness` | Explicit configuration and declared tracks, timing, geometry or profile inputs |

Supply the corresponding Make variables, for example:

```bash
make PYTHON=python source-reconcile SOURCE_BUNDLE=build/raw_source_export \
  SOURCE_RECONCILIATION_OUTPUT=build/source_reconciliation
```

Source reconstruction is described in [source-export.md](source-export.md).
The optional perspective-aware speed route and reference-based evaluation are
described in [kinematics-robustness.md](kinematics-robustness.md). The scalar
parameters used for the recorded thesis aggregates are documented separately
in [data/calibration/](../data/calibration/README.md).

The detector-development summary command additionally requires `args.yaml`
and `results.csv` for both `runs/rtdetr_finetune-4/` and
`runs/rtdetr_shinjuku/`, plus `data/eval_dataset/coco_eval.yaml`. These logs are
not published; the preserved summary is in
`results/profile_measurement_checks_20261117/`. Development validation does
not establish independent test membership or which checkpoint produced a
particular source aggregate.

## Model and rate identities

The pinned environment uses Python 3.11, NumPy 2.1.3, pandas 2.2.3, SciPy
1.14.1 and Matplotlib 3.9.4. Video dependencies are separately pinned in the
`vision` extra. Environment files remain the installation source of truth.

The recorded rate database is MOVES 5.0 (`movesdb20241112`). Project class
IDs are bus 0, car 1, heavy truck 2 and light truck 3. Their native emission
types are 6, 1, 13 and 2; their MOVES source types are 42, 21, 62 and 31.
The continuous comparator is the study's MOVES-derived NOx adaptation, not
the original published VT-CPFM fuel formulation.

EPA's [VSP-bin FAQ](https://www.epa.gov/moves/how-do-i-output-rates-vsp-bin)
distinguishes default input rates from outputs with run-time adjustments.
[EPA-420-R-24-015](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P101CU1O.TXT),
Table 2-16, describes the shared 2010–2018 HHD/urban-bus rate-development basis.
The bus/heavy-truck tables differ at Mode 40 NOx. The symmetric substitution
diagnostic assesses rate-input response; replaying the extraction itself
requires the original MOVES database. Zero placeholder columns in the
eight-column extraction are not measured pollutant observations.

The fixed hybrid scenario combines 83% conventional output with 17% of a
distance-based 0.013 g/km component for passenger cars. The stock proportion
is based on [AIRIA's March 2025 hybrid stock](https://www.airia.or.jp/publish/file/HVEV_2025.pdf)
and [total stock bulletin](https://www.airia.or.jp/news/v19mrm0000002med.html).
The factor is the listed JC08 certification/regulatory value in the 2015
[Prius](https://toyota.jp/pages/contents/prius/003_p_007/pdf/spec/prius_ecology_201504.pdf)
and [Aqua](https://toyota.jp/pages/contents/aqua/001_p_007/pdf/spec/aqua_ecology_201511.pdf)
specifications, not a measured local hybrid-fleet mean.
