# Calibrated speeds and robustness checks

These optional tools require source tracks, timing and geometry that are not
included in the public statistical package. They calculate fresh numeric outputs without modifying historical
speed/emission rows or archived results. Run them on the processing host or a
small reproducible source export. Real geometry, recording timing and
independent reference observations are inputs, not values inferred from the
ambient correlations.

## Production speeds

```bash
make PYTHON=.venv/bin/python calibrated-speeds \
  SPEED_DATABASE=<source.db> SPEED_CONFIG=<production-speed-v4.json> \
  SPEED_OUTPUT=build/<fresh-speed-run>
```

The corrected entry point reads SQLite tracks in read-only mode. It uses
bounding-box bottom centres, perspective-aware geometry and explicit elapsed
source-frame times. It does not use a guessed FPS, compressed-video duration,
global pixel scale, zero-filled missing observation or silent speed cap.
The former scalar callable remains a historical compatibility API, not the
corrected production route.

Each video must match its declared site, camera epoch, calibration, source
identity, resolution and recording start. A planar homography uses verified
image/metric control points and a valid convex footprint. The bounded-zone
alternative uses independently calibrated local scales in disjoint convex
zones. Invalid boxes, unavailable timing, frame/time gaps and zone crossings
cannot produce valid steps. Inspect exclusions rather than filling them.

The calculation records interval speeds, duration-weighted second bins and
separate fully covered contiguous one-second profiles. Partial seconds remain
partial; acceleration is not interpolated across missing observations.
Producing these records does not establish independent speed accuracy.
Use [reference-based measurement](#reference-based-measurement) for
reference-based errors and per-camera/zone support.

The production JSON has `schema_version: "production_speed_v4"`, an explicit
positive `maximum_step_seconds`, `calibrations` and `sources` lists.

| Declaration | Required fields |
|---|---|
| Calibration | `site`, `camera_epoch`, `calibration_id`, `width`, `height`, `route`, `geometry_evidence_path`, `geometry_evidence_sha256` |
| Homography route | `image_points`, `world_points_metres`, `image_polygon` |
| Bounded-zone route | `zones`, each with `id`, `image_polygon`, `scale_metres_per_pixel` |
| Source video | `video_id`, `source_id`, `site`, `camera_epoch`, `calibration_id`, `width`, `height`, `recording_start_utc`, `source_path`, `source_sha256`, `timing_source_id`, `timing_path`, `timing_sha256`, `timing_basis` |

Paths resolve relative to the configuration file; hashes identify exact bytes.
`source_path` must identify the selected database video's actual path.
`timing_basis` is `source_elapsed_seconds`; the timing CSV contains unique
`frame_id` and strictly increasing `elapsed_seconds`. Supply source-clock
evidence, not a value inferred from encoded-video FPS. Valid steps require
consecutive frame IDs and elapsed time no greater than the declared maximum.

The fresh bundle contains `speed_steps_v4.csv`, `speed_seconds_v4.csv`,
`speed_profiles_v4.csv`, `speed_exclusions_v4.csv` and
`speed_samples_v4.csv`. The last table adapts actual valid intervals to the
hourly baseline's fields and uses collision-safe source/track identities.
Its validity flags describe geometry/step validity, not independent reference
acceptance. The bundle also contains the selected numeric source tracks and
timing, configuration and a hash-bound production receipt.

```bash
make PYTHON=.venv/bin/python calibrated-speeds-replay \
  SPEED_OUTPUT=build/<speed-run> \
  SPEED_REPLAY_OUTPUT=build/<fresh-speed-replay>
```

Numerical replay recomputes the exported speeds from the frozen numeric
records and calibration declarations. It does not require SQLite or raw
video and does not claim to reverify the recorded raw-source or physical
geometry evidence. Original production verifies those source-file identities;
independent calibration acceptance remains a separate measurement check.

## Reference-based measurement

This calculation requires object-matched annotations/predictions, entry-event
references, independent speed references and their calibration segments. The
configuration declares `schema_version: "measurement_validation_config_v4"`,
all nine named inputs with repository-relative paths and SHA-256 values,
matching/bootstrap parameters and the fitting-source video hashes.

```bash
python -m vehicle_emission.measurement.evaluate \
  --config data/manifests/measurement-validation.json \
  --output-directory build/reference_measurement
```

Supply the configuration and referenced observations locally; they are not
included study measurements. The required fields are enforced by
`measurement/evaluate.py` and illustrated with controlled inputs in
`tests/test_measurement_evaluation.py`. Pre-entered MAE/F1 summaries cannot
replace reference/prediction pairs. The legacy schema examples in
`tests/fixtures/legacy/` are test inputs only.

## Optional robustness bundle

```bash
make PYTHON=.venv/bin/python robustness \
  ROBUSTNESS_CONFIG=<robustness-config.json> \
  ROBUSTNESS_OUTPUT=build/<fresh-diagnostic-run>
make PYTHON=.venv/bin/python robustness-replay \
  ROBUSTNESS_OUTPUT=build/<fresh-diagnostic-run>
```

The configuration has exactly these top-level fields:

```json
{
  "schema_version": 1,
  "analysis_scope": "locked_retrospective_two_site",
  "status": "optional_diagnostics_only",
  "predeclaration_id": "<recorded-diagnostic-specification>",
  "speed": null,
  "temporal": null
}
```

Enable only the sections whose inputs are available. Input references contain
a repository-relative `path` and exact-byte `sha256`. Preparation receipts
record the declaration; they do not prove that it preceded outcome access.

### Speed-error sensitivity

Replace `speed: null` with `profiles: {path, sha256}` and a `scenarios` list.
Each scenario contains `scenario_id`, `multiplier` and `additive_mps` and uses
`v_new = multiplier * v + additive_mps`. Supply the range from independent
measurement errors; no empirical error bounds are built in.

Profile CSV fields are `site`, `source_id`, `profile_id`, `class_id`,
`second_index` and `speed_mps`. Class IDs follow the project adapter: bus 0,
car 1, heavy truck 2, light truck 3. Seconds must be contiguous within each
profile; explicit missing speeds split valid runs and remain in support.
Optional `duration_seconds` must be 1, and optional `ts_utc` must contain
contiguous canonical UTC seconds. Negative perturbed speeds reject the
calculation rather than being clipped.

The tool reruns MOVEStar acceleration, VSP, operating modes and NOx using the
tracked coefficient/rate tables. `speed_profiles.csv` records support;
`speed_runs.csv` records recalculated run diagnostics;
`speed_sensitivity.csv` records baseline/scenario totals and differences for
each profile. These are profile-level sensitivity calculations, not hourly
correlations, confidence intervals or accepted speed-error estimates.

### Dependence-preserving temporal controls

Replace `temporal: null` with `hourly_rows: {path, sha256}`,
`calendar_start_date`, `calendar_end_date`, `day_offsets` and `acf_lags_hours`.
Dates are inclusive Tokyo calendar dates; offsets are explicit signed nonzero
whole days smaller than the declared calendar. Positive offsets pair a target
hour with earlier predictor days, wrapping at the declared calendar boundary.
ACF lags are positive integer hours with no wrap.

Use the v4 pipeline's actual eligible hourly rows, including site, monitor
start/end UTC, local calendar date, actual-target status, eligibility, ambient
NOx, emission index and count. The full common calendar retains missing hours
and days. Both predictors move together under the same transformation at both
sites; each observed/control comparison uses an identical retained mask.

`temporal_contrasts.csv` reports observed and transformed Spearman associations
and index-minus-count contrasts. `temporal_support.csv` retains every calendar
position and exclusion; `temporal_day_mapping.csv` records source/destination
dates and wrap boundaries. `temporal_acf.csv` counts actual time-separated
pairs and calculates pairwise Pearson autocorrelation.

These are descriptive controls, not IID shuffles or automatically valid null
tests. Keep every declared offset and its support. Circular seams and
non-stationarity remain interpretation limits; do not select a favourable
offset or use controls to change the primary lag.

## Replay and source boundaries

The robustness receipt binds the exact configuration, input CSVs, calculator
code, coefficient tables and calculated outputs. Replay checks identities and
recomputes the outputs read-only. Both diagnostic sections can be unrequested;
missing optional inputs do not remove primary observations or stop other work.

Fresh outputs are engineering evidence until applied to verified source
measurements with recorded settings. They do not independently establish
detector lineage, calibration acceptance or a source-reproduced thesis result.
