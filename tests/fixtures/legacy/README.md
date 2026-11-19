# Legacy examples

These files test schema handling and rejection of pre-entered summaries. They
are not measured speed-validation evidence or the recorded-hour cohort.

- `speed_validation_v2.csv`: pre-entered summary metrics; the speed evaluator
  must reject these instead of treating them as measured prediction/reference
  pairs.
- `two_site_coverage_v2.csv`: unverified historical coverage example, not the
  study's eligible-hour inventory.
- `site_calibration_v2.json`: historical homography schema example, not the
  scalar calibration used for the recorded thesis aggregates.

The original file bytes are retained. Actual recorded scalar parameters are in
`data/calibration/recorded_scalar_calibration.json`.
