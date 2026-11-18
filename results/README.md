# Preserved Result Bundles

This directory contains the derived numerical outputs, SHA-256 receipts,
validation workbooks, and rendered figures supporting the Master's thesis and
companion publications.

## Primary & Sensitivity Bundles

| Directory | Contents |
|---|---|
| `recorded_hourly_reanalysis_20261109/` | Core hourly panel (593 Sukiyabashi / 555 Shinjuku dry-weather hours), eligibility filters, and temporal shift controls. |
| `recorded_hourly_extension_20261109/` | Core 20,000-draw paired intervals, fixed class-weighted entering counts, staged rank residuals, speed-CV strata, raw/adjusted comparisons, and cross-site contrasts. |
| `measurement_model_extension_20261109/` | Manual classification and entry-line validations, prescribed response states, fitted continuous parameters, and worked hour diagnostics. |
| `model_ordering_20261109/` | Model-to-model rank concordance, ratios, signed rank disagreements, and diurnal profile data. |
| `model_fit_diagnostics_20261117/` | Rate-fit residuals, clipping/extrapolation checks, and Mode 40 bus/heavy-truck swap sensitivity. |
| `recording_support_sensitivity_20261117/` | Six-start metadata restriction (579/553 hours) and paired restricted-minus-original contrasts. |
| `timing_weight_sensitivity_20261117/` | Clip playable-duration clock comparisons, profile-to-entry ratios, and 15/25/40 km/h fixed-weight sensitivity. |
| `weighted_count_audit_20261117/` | Fixed-weight manual class-count discrepancy audit. |
| `profile_measurement_checks_20261117/` | Detector final-epoch development validation summaries. |
| `thesis_figure_refresh_20261117/` | Final manuscript figures reproduced from the verified bundles above. |
| `v4_all_weather/` | All-weather hourly paired dataset used as an input for dry-weather vs. all-weather checks. |

## Manual Validation Workbooks

- `Sukiyabashi/manual_vs_cv_counting_validation.xlsx`
- `Shinjuku/manual_vs_cv_counting_validation.xlsx`
