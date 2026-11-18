# Thesis calculation guide

This guide holds implementation details separately from the reader-facing
thesis. Scientific definitions and numerical comparisons remain in the report.

## Environment and source identities

The numerical environment uses Python 3.11, NumPy 2.1.3, pandas 2.2.3,
SciPy 1.14.1 and Matplotlib 3.9.4. Video-processing dependencies include
OpenCV 4.10.0.84, PyTorch 2.5.1, torchvision 0.20.1 and Ultralytics 8.3.220.
The environment specifications remain the installation source of truth.
The recorded processing workstation used Windows, an Intel Core i7-12700K,
32 GB RAM and an NVIDIA GeForce RTX 3080.

- MOVES release database: `movesdb20241112` (MOVES 5.0).
- Bus and combination-truck lookup files: `EmsRate_6.csv` and
  `EmsRate_13.csv`.
- Native class indices: bus 6, car 1, heavy truck 13, light truck 2.
- Corresponding MOVES source types: bus 42, car 21, heavy truck 62,
  light truck 31.
- Ambient station identifiers: Kasumigaseki 13101520; Shinjuku Gyoen 13104010.
- Bootstrap seed: 42. Core intervals use 20,000 draws; descriptive extensions
  use 2,000. Preserve the separate interval families and block lengths.

## Reproduction commands

Use the received numeric export as `SOURCE_BUNDLE`. No full database transfer
or repeated video processing is needed for the recorded-hour calculation.
Every output directory must be fresh.

```bash
make PYTHON=.venv/bin/python thesis-analysis \
  SOURCE_BUNDLE=build/raw_source_export \
  REANALYSIS_OUTPUT=build/recorded_hourly_run

make PYTHON=.venv/bin/python thesis-extension \
  SOURCE_BUNDLE=build/raw_source_export \
  EXTENSION_BASE=build/recorded_hourly_run \
  EXTENSION_OUTPUT=build/information_run

make PYTHON=.venv/bin/python thesis-diagnostics \
  SOURCE_BUNDLE=build/raw_source_export \
  DIAGNOSTIC_PANEL=build/recorded_hourly_run/recorded_hourly_panel.csv \
  DIAGNOSTIC_OUTPUT=build/measurement_model_run

make PYTHON=.venv/bin/python thesis-ordering \
  ORDERING_SOURCE=build/information_run \
  ORDERING_OUTPUT=build/ordering_run

make PYTHON=.venv/bin/python thesis-figures \
  FIGURE_EXTENSION=build/information_run \
  FIGURE_MEASUREMENT=build/measurement_model_run \
  FIGURE_OUTPUT=build/figure_run
```

The prepared all-weather input is
`results/v4_all_weather/phase5_hourly_paired_dataset_all_weather.csv`.
The thesis uses these preserved result directories:

- `results/recorded_hourly_reanalysis_20261109/`
- `results/recorded_hourly_extension_20261109/`
- `results/measurement_model_extension_20261109/`
- `results/model_ordering_20261109/`
- `results/thesis_figure_extension_20261109/`
- `results/recording_support_sensitivity_20261117/`
- `results/weighted_count_audit_20261117/`

Their receipts bind source files, calculation code and outputs by SHA-256.
The measurement and information receipts also identify lookup tables. Read
numeric panels with the round-trip float parser and retain full output
precision; round only for display. Keep `estimation_method` distinct when
reading both model outputs, and select a single speed summary per clip.

## Recording schedule and weighted-count audit

These calculations reuse the received export and preserved extension bundles.
The local received source is
`build/source_export_20261022/raw_source_export/`; use the actual export path
on another machine. Every output path must be fresh.

```bash
make PYTHON=.venv/bin/python thesis-recording-support \
  SOURCE_BUNDLE=build/source_export_20261022/raw_source_export \
  RECORDING_SUPPORT_OUTPUT=build/recording_support_replay_01

make PYTHON=.venv/bin/python thesis-weighted-audit \
  WEIGHTED_AUDIT_OUTPUT=build/weighted_count_replay_01
```

The recording restriction requires six records, six distinct starts and one
start per ten-minute clock bin. It retains 579/553 core and 577/552
count-complete hours. It measures the start schedule, not continuous video
coverage. Both populations retain the full 61-date calendar, seed 42 and
2,000 shared one-/seven-day draws; direct population changes are calculated
within those draws. Intervals are pointwise descriptive.

The weighted audit uses the same 12 clips and four-class manual count pairs
as the existing measurement bundle. It sums fixed weight times absolute
clip--class discrepancy before aggregation and divides by weighted manual
counts. Classification cells are not joint class-specific entering events.
Both new bundles bind input receipts, CSVs and calculator bytes; their READMEs
define every output. The preserved core cohort and its intervals are unchanged.

The reporting corrections follow the implemented source: the residual sample
loses one missing wind-speed hour per site, ByteTrack's 0.80 setting is an
association-cost limit, and the braking rule uses sharp current deceleration
or its two preceding moderate decelerations. Repository history preserves
that rule from the original engine addition; the received clip-aggregate
export does not identify the historical emission-engine version or contain
per-second operating states. Independent speed errors require paired physical
references, and observed state-occupancy summaries require those state records.
These are the two separate source-dependent additions, not prerequisites to
the recorded-hour comparisons.

## Rate-source verification

EPA's [VSP-bin FAQ](https://www.epa.gov/moves/how-do-i-output-rates-vsp-bin)
distinguishes default input rates from outputs with run-time adjustments.
[EPA-420-R-24-015](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P101CU1O.TXT),
Table 2-16 (printed p. 56), describes the shared 2010–2018 HHD/urban-bus
rate-development basis and bus-specific tampering/malmaintenance factors.
Common base rates therefore do not establish copying. The two checked-in
tables differ only at Mode 40 NOx; their first tracked versions already have
that difference. The query below records source selection, but verifying its
exact result still requires the original MOVES database. Zero placeholder
columns in the extraction are not independent matching pollutant measurements.

The Sensors publisher lists the first author's surname as `Rodriguez Valido`;
the prose and bibliography now use that published spelling consistently.

## Timing, fit and timestamped-profile checks

Fresh available-record diagnostics are in
`results/model_fit_diagnostics_20261117/`,
`results/timing_weight_sensitivity_20261117/` and
`results/profile_measurement_checks_20261117/`. They do not replace the
recorded-hour results. Reproduce into fresh paths:

```bash
make PYTHON=.venv/bin/python thesis-model-fit \
  MODEL_FIT_OUTPUT=build/model_fit_replay_01
make PYTHON=.venv/bin/python thesis-timing-weights \
  SOURCE_BUNDLE=build/source_export_20261022/raw_source_export \
  TIMING_WEIGHTS_OUTPUT=build/timing_weights_replay_01
.venv/bin/python -m vehicle_emission.measurement.development_logs \
  --root . --output build/detector_development_replay_01.json
```

The timing join uses exact site/video IDs for the 6,965 core clips. Above
305 seconds, the retained scalar callable uses approximately `N=d*FPS` and
`dt=600/N`: assigned time stretches by `600/d`, while uncapped step speed is
`d/600` of the media-clock calculation. Two clips use the short-duration rule.
Playable duration alone is not wall-clock coverage. All three fixed-speed
count scenarios give identical weights and paired intervals on 591/554 hours.

Fit diagnostics retain all twenty construction points, raw and applied-rate
residuals, the fixed VSP grid, roots and both Mode 40 substitutions. They do
not estimate observed class/state contributions or corrected hourly effects.

`emissions.profiles.calculate_timestamped_emissions` accepts the corrected
production route's `speed_seconds_v4.csv` or `speed_profiles_v4.csv`. It returns
run, support and exclusion tables, splits missing/partial seconds and explicit
profile/zone boundaries, and integrates both models on identical complete
seconds. Singleton runs retain each model's zero-boundary-acceleration rule.
The archived emission adapter instead discards second labels before model
evaluation; it and all historical results remain unchanged. Actual per-second
records are required to quantify observed gaps and run a corrected replay.

Detector summaries retain final-epoch development-validation metrics, input
hashes and logged training lineage. Neither the logs nor local evaluation
path aliases establish independent test membership or the checkpoint used for
each historical clip. Those exact production identities remain source inputs,
not inferred from model filenames.

## Closest-study verification

The closest-work comparison separates movement accuracy, agreement with an
emission model, measured ambient prediction and index--ambient association.
The CMEM citation now matches the linked official guide: Scora and Barth,
version 3.01, June 2006. No new bibliography identity or journal-publication
status is inferred from a preprint.

## Presentation-only figure refresh

To reproduce the current figure styling from preserved numerical bundles,
without rerunning inference:

```bash
make PYTHON=.venv/bin/python thesis-figure-refresh \
  FIGURE_REFRESH_OUTPUT=build/thesis_figure_refresh_run_01
```

The output directory must be fresh. The runner verifies the source receipts
and uses round-trip CSV parsing. It rerenders twelve plots and copies the
existing constructed-profile image without model execution. The receipt binds
inputs, plotting code, descriptive tables and images. The published refresh is
`results/thesis_figure_refresh_20261117/`; the thesis uses its dated image copies.
Earlier numerical bundles and image versions remain unchanged. The historical
scratch entry point now delegates to this same fresh-output command.

## Hybrid scenario sources

The recorded adjustment remains 83% conventional output plus 17% of a
distance-based 0.013 g/km NOx component for passenger cars. The parameters are
unchanged; their interpretation follows these primary sources:

- [AIRIA hybrid/electric stock](https://www.airia.or.jp/publish/file/HVEV_2025.pdf):
  13,945,137 hybrids at 31 March 2025, including plug-in hybrids across vehicle
  categories.
- [AIRIA total stock bulletin](https://www.airia.or.jp/news/v19mrm0000002med.html):
  82,699,840 motor vehicles including light motor vehicles at the same date;
  published 10 June 2025. Their ratio is 16.86%, rounded to the fixed 17%
  scenario weight. This is not a measured local passenger-car proportion.
- [Toyota Prius, April 2015](https://toyota.jp/pages/contents/prius/003_p_007/pdf/spec/prius_ecology_201504.pdf)
  and [Toyota Aqua, November 2015](https://toyota.jp/pages/contents/aqua/001_p_007/pdf/spec/aqua_ecology_201511.pdf):
  0.013 g/km NOx is the listed certification-level or applicable regulatory
  value under JC08 for DAA-ZVW30 and DAA-NHP10, respectively, not an observed
  mean rate for the local hybrid fleet.

The raw/adjusted comparison quantifies this scenario's influence on matched
hourly associations. It does not estimate a local hybrid share.

## Database structure

The source SQLite database links `video_master` and
`vehicle_classification_master` with `vehicle_tracks`, `crossing_events`,
`entering_counts`, `vehicle_speeds` and `video_emissions`. Environmental
records are in `air_pollution_history` and `weather_history`. Site/video
identities and UTC timestamps define the joins. The implementation uses
foreign-key constraints and `ON DELETE CASCADE` for dependent records.

Manual count pairs come from the `Classification` and `Entering` worksheets,
not their cached summary cells.

## MOVES extraction query

The heavy-truck extraction below retains the recorded source-bin and age-group
configuration. The bus extraction uses its corresponding source-bin identifier.
The eight-column format includes fields beyond the NOx endpoint; zero-valued
placeholder columns preserve the original lookup layout.

```sql
SELECT era.opModeID,
       SUM(CASE WHEN era.polProcessID = 101
                THEN era.meanBaseRate ELSE 0 END) AS CO,
       0 AS HC,
       SUM(CASE WHEN era.polProcessID = 301
                THEN era.meanBaseRate ELSE 0 END) AS NOx,
       SUM(CASE WHEN era.polProcessID = 11201
                THEN era.meanBaseRate ELSE 0 END) AS PM25_Ele,
       0 AS PM25_Org,
       MAX(er.Energy) AS Energy,
       0 AS CO2
FROM (
    SELECT opModeID, polProcessID, meanBaseRate
    FROM emissionratebyage
    WHERE sourceBinID = '1020147150000000000'
      AND ageGroupID = 3
      AND polProcessID IN (101, 301, 11201)
) era
LEFT JOIN (
    SELECT opModeID,
           SUM(CASE WHEN polProcessID IN (9101, 9190)
                    THEN meanBaseRate ELSE 0 END) AS Energy
    FROM emissionrate
    WHERE sourceBinID = '1020147150000000000'
    GROUP BY opModeID
) er ON era.opModeID = er.opModeID
GROUP BY era.opModeID
ORDER BY era.opModeID;
```

## Checks and document build

```bash
make PYTHON=.venv/bin/python test
make PYTHON=.venv/bin/python lint
```

