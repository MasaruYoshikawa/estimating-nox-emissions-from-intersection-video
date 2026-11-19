# Data sources and conventions

## Environmental histories

`airpollution_history/` preserves received hourly CSV exports from the Ministry
of the Environment's [Soramame service](https://soramame.env.go.jp/). The
Sukiyabashi comparison uses Kasumigaseki station **13101520**; Shinjuku uses
the National Shinjuku roadside station **13104540**.
The original hourly values were matched to provider records and station
metadata. The Setagaya regional series matches Tokyo station
**0110** (national code **13112010**), not Setagaya Seijo. Hibiya and Nakano
histories are also retained but are not the two receptor outcomes.

The source `Date` is day/month/year in Japan Standard Time. `Time` values 1–24
label interval ends: date 1 May, hour 10 represents 09:00–10:00 JST. Prepared
panels instead use UTC interval-start and interval-end timestamps, with an
explicit JST calendar date. Concentrations are in ppm as named in the source
columns. Tokyo API NOx values are in ppb and require division by 1,000 to
compare with these CSVs.

The original exports retain their received values, including zero-coded
records. They do not retain provider status fields. Use the quality-corrected
analysis panel for current inference, not a fresh analysis of these original
values without the quality mapping. Corrections use exact station/interval
provider flags; a genuine numeric zero is distinct from missing data. Missing
regional observations affect only background-dependent comparisons. Soramame
source formats are described in its
[API manual](https://soramame.env.go.jp/data/apiManual/3.html) and
[download specification](https://soramame.env.go.jp/data/legend/download/format.html).
The
[Tokyo API specification](https://www.taiki.kankyo.metro.tokyo.lg.jp/taikikankyo5g/apiguid/index.html)
defines `欠測` as missing and an empty value as uncollected. It also explains
the distinction between provisional and final observations and states Tokyo's
ownership of API data.

`weather_history/weather_history.csv` is adapted from
[JMA past-weather observations](https://www.data.jma.go.jp/stats/etrn/index.php).
The source spelling of headings is retained. Temperature is in degrees Celsius,
precipitation in mm, wind speed in m/s and wind direction in named compass
sectors. Rainfall is aligned to the same monitoring interval by its end label;
instantaneous weather controls follow the declared interval-start convention.
Missing values remain unavailable, not zero. JMA content use and attribution
are described in its [website terms](https://www.jma.go.jp/jma/en/copyright.html).

## Analysis panels

The current quality-corrected panel is
`results/monitor_quality_reanalysis_20261119/extension_hourly_panel.csv`.
Its receipt and quality evidence bind exact source versions. The replay command
uses this prepared panel without contacting a provider.

One row is one site and UTC hour. Key fields are:

| Fields | Meaning |
|---|---|
| `site`, `monitor_start_utc`, `monitor_end_utc`, `local_calendar_date` | Site and exact hourly interval, with JST calendar date |
| `ambient_nox_ppm`, `setagaya_nox_ppm` | Receptor and regional ambient NOx, ppm |
| `movestar_nox_grams`, `vt_cpfm_nox_grams` | Hybrid-scenario-adjusted recorded clip aggregates, g; the latter is the continuous adaptation |
| corresponding `_raw` fields | Recorded model aggregates before the hybrid adjustment |
| `vehicle_count` | Tracked profile count, not unique entering vehicles |
| `captured_clip_entering_count`, `entering_class_0`–`3` | Captured-clip entry events and class totals: bus, car, heavy truck, light truck |
| `fixed_rate_weighted_entries` | Class-weighted entries at the fixed steady 25 km/h reference |
| `mean_speed_mps` | Mean recorded clip-speed proxy, m/s |
| `between_clip_speed_cv` | Sample SD divided by mean across stored clip mean speeds |
| `temperature`, `wind_speed` | Weather controls, degrees Celsius and m/s |
| `residual_complete` | Common complete-case membership for all residual stages |

Original November panels and their receipts remain in their dated directories.
Do not mix intervals or results from different bundles.

## Other inputs

- [Recorded scalar calibration](calibration/README.md) documents the image
  chord, distance, scale and reference resolution actually recorded per site.
- `src/vehicle_emission/emissions/` contains coefficients and lookup rates from
  the U.S. EPA's [MOVES model](https://www.epa.gov/moves). The extraction and
  model definitions are in the [calculation guide](../docs/thesis-reproduction.md).
- The manual-count workbooks in `results/Sukiyabashi/` and `results/Shinjuku/`
  are study reference counts; they are not object-matched detection or
  independent speed references.
- `tests/fixtures/legacy/` contains unverified historical schema examples,
  separated from study measurements.

Provider data and third-party model material retain their own terms; the
software's MIT licence does not relicense those sources. See
[NOTICE.md](../NOTICE.md) for model attribution.
