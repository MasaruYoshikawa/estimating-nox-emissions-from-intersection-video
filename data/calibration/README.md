# Recorded camera calibration

`recorded_scalar_calibration.json` documents the single measured image chord
and road distance recorded for each camera, the recorded pixels-per-metre
parameter and its reference resolution. A video's effective scale multiplies
that parameter by its width divided by the reference width.

The stored scale is preserved at its recorded precision; it is not recomputed
from rounded landmark measurements. This is a scalar geometry record, not a
road-plane homography or a measurement of independent speed errors.

The optional perspective-aware production-speed tools require their own
measured control points, valid zones and source-frame timing. The legacy
homography schema example lives under `tests/fixtures/legacy/`, not here.
