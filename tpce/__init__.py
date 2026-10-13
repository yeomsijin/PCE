"""Own-own PCE and main-text TC, with historical time-induced TPCE APIs.

Use own_own_costs for the current PCE definition. estimate and TPCEConfig
retain the time-induced definition solely for historical compatibility.
"""

from .model import TPCEConfig, Calibration, TPCEResult, estimate, prepare, calibrate
from .entropy import entropy_excess, induced_probability, direction

__all__ = [
    "TPCEConfig",
    "Calibration",
    "TPCEResult",
    "estimate",
    "prepare",
    "calibrate",
    "entropy_excess",
    "induced_probability",
    "direction",
]

from .own_own import (
    own_own_costs,
    DEFAULT_BANDWIDTH_METHOD,
    h1_pair_bandwidth,
    H1_METHOD,
)

__all__ += [
    "own_own_costs",
    "DEFAULT_BANDWIDTH_METHOD",
    "h1_pair_bandwidth",
    "H1_METHOD",
]

from .tc import estimate_tc, TCResult, METHOD as DEFAULT_TC_METHOD

__all__ += ["estimate_tc", "TCResult", "DEFAULT_TC_METHOD"]
