# Recording-schedule sensitivity

The current recorded-hour panel is compared with a metadata-defined subset:
exactly six source clips, six distinct recorded start times, and one start in
each ten-minute clock bin (00–09, 10–19, …, 50–59). This checks the recorded
start schedule, not continuous media coverage or valid recording duration.
The original panel and analysis bundles are unchanged.

- `recording_support_clips.csv`: source clip identity, recorded UTC start,
  assigned hour and ten-minute clock bin.
- `recording_support_hours.csv`: source and panel clip counts, distinct starts
  and bins, the restricted-support flag, and entry-count completeness.
- `recording_support_summary.csv`: hours, site-days, clips and complete
  entry-count hours in each cohort.
- `recording_support_inference.csv`: Spearman correlations with ambient NOx,
  paired VT-CPFM-minus-comparator differences, and direct
  restricted-minus-current differences.
- `receipt.json`: settings and SHA-256 bindings for the inputs, calculators
  and four CSV outputs.

## Populations and intervals

`recorded_core` uses the current or restricted site-hours. Profile count is
the stored `vehicle_count`, not an entering-event count.
`complete_entry_counts` additionally requires all three existing count
baselines: entering count, bus/heavy entering count and fixed-weighted entries.
The VT-CPFM and weighted-entry correlations use identical hours within each
such comparison.

Every cohort and estimand shares the original 61-date May–June calendar,
including dates without eligible hours. The calculation uses 2,000 nonwrapping
moving-block draws at one- and seven-day lengths, seed 42. Subset filtering
occurs after each joint calendar draw; repeated dates retain their sampling
multiplicity. Direct support differences are calculated within each shared
draw, not by subtracting interval endpoints.

`estimate`, `ci_low` and `ci_high` give the point estimate and pointwise 95%
percentile interval. `current_hours`, `restricted_hours` and corresponding
site-day columns identify the paired populations. `resamples_valid` and
`resamples_invalid` retain draw validity; any invalid draw withholds that
interval. These are separate descriptive sensitivities, not a replacement
primary cohort or multiplicity family.
