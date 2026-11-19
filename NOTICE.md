# Model and data attribution

## MOVEStar

`src/vehicle_emission/emissions/movestar.py` adapts the Python v1.4 formulation
from [MOVESTAR-Fuel-and-Emission-Model](https://github.com/ziranw/MOVESTAR-Fuel-and-Emission-Model).
Its Python source header credits Kinjal Dand and Ziran Wang. The upstream
project credits Ziran Wang, Guoyuan Wu and George Scora, University of
California, Riverside, and requests citation of:

Wang, Z., Wu, G., and Scora, G. (2020). *MOVESTAR: An Open-Source Vehicle Fuel
and Emission Model based on USEPA MOVES*. [arXiv:2008.04986](https://arxiv.org/abs/2008.04986).

The upstream repository did not provide an explicit licence file when checked
on 19 November 2026. This project's MIT licence covers project-authored work;
it does not assert a new licence over upstream material.

## Environmental and rate data

Ambient histories derive from Japan's Ministry of the Environment
[Soramame monitoring service](https://soramame.env.go.jp/). Monitoring-quality
checks use its original hourly data/status records and the Tokyo Metropolitan Government's
[hourly observation API](https://www.taiki.kankyo.metro.tokyo.lg.jp/taikikankyo5g/apiguid/index.html).
Weather histories are adapted from the
[Japan Meteorological Agency](https://www.data.jma.go.jp/stats/etrn/index.php).
Emission-rate inputs are extracted from the U.S. EPA's
[MOVES 5.0](https://www.epa.gov/moves).

Source identities, transformations and provider terms are linked in
[data/README.md](data/README.md). Prepared panels and figures are study-derived
products, not official provider products.
