# Fixed-weight class-count audit

This descriptive audit uses the existing manual and automated classification
counts for seven Sukiyabashi clips and five Shinjuku clips. Each clip retains
all four vehicle classes, including zero-count cells. The weights are the
existing fixed reference-rate ratios at 25 km/h and zero acceleration, with
the car rate normalised to one.

## Tables and units

- `weighted_count_cells.csv`: one clip--class pair, its fixed weight, raw counts
  and weighted counts/differences.
- `weighted_count_clips.csv`: clip totals and relative discrepancies.
- `weighted_count_classes.csv`: site--class totals and each class's share of
  weighted manual counts and absolute cell discrepancy.
- `weighted_count_sites.csv`: site totals, clip/cell denominators and relative
  discrepancies.

Raw counts are vehicles. Weighted counts are car-reference-rate equivalents,
not measured emissions. Signed differences are automated minus manual.
`weighted_absolute_cell_difference` sums `weight * abs(automated - manual)`
before aggregation. The cell table calls this term
`weighted_absolute_difference`. Absolute weighted clip, class and site net
differences instead take the absolute value after summing signed differences.
These columns show how opposite errors cancel at different aggregation levels.

`relative_weighted_absolute_cell_difference` divides absolute weighted cell
discrepancy by the matching weighted manual total. It is a proportion, not a
percentage. An empty relative/share cell denotes a zero denominator; a recorded
zero count remains zero.

The observation unit is a clip--class count cell, not a matched vehicle or
class-specific entering event. The available clips are not a random sample, so
the audit reports descriptive discrepancies without sampling intervals.
`receipt.json` records input, calculator and CSV hashes.
