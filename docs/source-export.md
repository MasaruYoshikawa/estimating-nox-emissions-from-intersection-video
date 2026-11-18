# Export existing database records

When the processing database is not available on the analysis machine, run this
command on its host. The export copies small numeric tables; it does not copy
videos, bounding-box tracks or the full database, rerun detection, calculate
correlations, or modify the source.

Close database writers first, then use a fresh output path:

```bash
make PYTHON=.venv/bin/python source-export \
  SOURCE_DATABASE=data/results.db SOURCE_OUTPUT=build/raw_source_export
```

Alternatively, from the repository in its Python environment:

```bash
python -m vehicle_emission.source_export --database data/results.db \
  --output-directory build/raw_source_export
```

Copy the resulting folder, not the large archive. Its size depends on the
number of recorded rows; no transfer-size estimate is assumed. It contains:

- Recorded video paths, start timestamps, durations and other video metadata.
- All stored emission methods, keeping unadjusted and hybrid-adjusted NOx
  separate. The emission table's `vehicle_count` is a track count, not an
  entering-event count.
- Raw entering-count and crossing-event rows. `entering_count_audit.csv`
  selects direction totals or class sums, never both; duplicates, invalid
  counts and total/class disagreements remain unavailable. Missing records
  are not replaced with zero. Arithmetic agreement is not measurement validation.
- Stored monitor/weather records without shifting, interpolation or replacing
  missing values. These are the complete stored tables, not independently
  assigned station roles or verified original feed observations.
- The class map when present, queries, table definitions, row counts and
  exporter/output hashes.
  A missing optional table is explicitly marked unavailable in the manifest.

The command preserves timestamp text, including unresolved legacy timestamps;
SQL NULL is an empty CSV field. It does not infer clocks from filenames or
crossing events from track counts. Source size/mtime are recorded, not presented
as a database content hash. A read-only SQLite transaction and before/after
file checks reject detected source changes. Nonempty WAL files and journals
are rejected rather than ignored. Do not delete sidecars to make the command
run; close the writer normally or use an existing closed snapshot. SQLite may
create a zero-byte WAL and shared-memory reader cache even for a read-only
connection. Their sizes are reported and the exporter does not delete them.

This is a **raw source export, not an analysis-ready or validated result**.
Actual valid recording intervals, feed interval semantics, checkpoint/calibration
identities and their production use still require their existing source records.
In particular, stored start/duration does not prove valid-second coverage, and
clip totals must not be spread uniformly into fabricated second-level inputs.
Engineering reconciles this bundle through the existing hourly and calibrated-
speed tools; only specifically missing source facts need follow-up.

The [calibrated-speed route](kinematics-robustness.md#production-speeds) still
needs the selected bounding-box tracks on the host, an explicit source
`frame_id`/`elapsed_seconds` timing table, verified source video identity/start,
and the corresponding homography or bounded-zone geometry. Its existing
numeric replay bundle can be transferred instead of all tracks/videos. The
hourly route separately needs actual valid recording intervals and source-
derived count/emission increments; first/last detections or a stored clip
duration cannot supply those validity facts.

## Verify a received bundle

Use a fresh output directory. This reads the bundle without changing it:

```bash
make PYTHON=.venv/bin/python source-reconcile \
  SOURCE_BUNDLE=build/raw_source_export \
  SOURCE_RECONCILIATION_OUTPUT=build/source_reconciliation
```

The four outputs are `verification.json`, `summary.json`,
`video_reconciliation.csv` and `stored_media_hourly.csv`. The command checks
all copied table hashes, headers, row counts and site/video references, and
recalculates the supplied count audit from raw rows. A missing or inconsistent
direction makes the whole video's recorded count unavailable, not a partial
sum. Source-role labels describe recorded names/paths; they do not select the
final cohort.

Stored media duration is intersected with half-open clock hours and overlaps
are merged. These are descriptive **media-duration unions, not valid recording
coverage**. Cross-hour clip emission totals remain intact; no clip total is
spread into fabricated second-level measurements. Outputs remain preparation
evidence, not empirical results or accepted measurement validation.

An optional replay compares original monitor and weather cells and binds their
hashes. Supply the monitor-label convention explicitly:

```bash
python -m vehicle_emission.source_reconciliation \
  --source-directory build/raw_source_export \
  --output-directory build/source_reconciliation_with_originals \
  --monitor-source Sukiyabashi=data/airpollution_history/AirPollution_Sukiyabashi.csv \
  --monitor-source Shinjuku=data/airpollution_history/AirPollution_Shinjuku.csv \
  --monitor-source Setagaya=data/airpollution_history/AirPollution_Setagaya.csv \
  --weather-source data/weather_history/weather_history.csv \
  --soramame-hour-convention end-labelled-hour
```

This checks the supplied convention against the original-to-database mapping;
it does not independently verify the camera clock or instrument support.
Monitor interval starts and weather observation instants are different:
previous-hour precipitation at weather label `t+1h` corresponds to monitor
interval `[t,t+1h)`. Wind and temperature retain their own observation support.
Environmental mismatches are reported, not silently corrected or treated as
transfer corruption. No alignment, eligibility threshold or inference setting
is frozen by this command.
