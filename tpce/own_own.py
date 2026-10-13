"""Active own-own H0/H1 PCE with symmetric pairwise bandwidth.

The historical time-induced API in model.py is retained for archived experiments.
Input lifetimes must already use the same reference scale in both spaces.
"""

import numpy as np

from .bandwidth import mixture_moments, persistence_variance_bandwidth
from .entropy import entropy_excess

DEFAULT_BANDWIDTH_METHOD = "pair_persistence_variance"
H1_METHOD = "pce_own_own_h1_pair_coordinate_persistence_variance_v1"


def h1_pair_bandwidth(x, y):
    """Compute diagonal H1 sigma_axis**2 = 2 Var_Q(axis).

    Axes are birth/lifetime. Each nonempty PD has equal mixture mass; bars
    within a PD are lifetime weighted. Exactly constant axes are omitted
    from the kernel. Both empty PDs return zero widths.
    """
    diagrams = []
    for values in (x, y):
        values = np.asarray(values, dtype=float)
        if (
            values.ndim != 2
            or values.shape[1] != 2
            or not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise ValueError("Expected finite nonnegative H1 (birth,lifetime) diagrams")
        diagrams.append(values)
    mean, variance, nonempty, empty = mixture_moments(diagrams, lifetime_axis=1)
    sigma = np.sqrt(2) * np.sqrt(variance)
    if not np.isfinite(sigma).all():
        raise ValueError("Nonfinite H1 pair bandwidth")
    return sigma, {
        "nonempty_diagrams": nonempty,
        "empty_diagrams": empty,
        "mixture_mean": mean.tolist(),
        "mixture_variance": variance.tolist(),
        "axes": ["birth", "lifetime"],
        "constant_axes": np.flatnonzero(sigma == 0).tolist(),
        "zero_variance_policy": "omit exactly constant support axes; no floor",
        "formula": "sigma_axis^2=2 Var_Q(axis); equal nonempty PD mass; lifetime weights within each PD",
    }


def own_own_costs(
    x,
    y,
    *,
    bandwidth_method=DEFAULT_BANDWIDTH_METHOD,
    bandwidth=None,
    response_scale=5.0,
):
    """Return [E(X,Y), E(Y,X)], shared sigma and calibration metadata.

    Pass 1D lifetimes for H0, or (bars,2) birth/lifetime arrays for H1.
    H0 unidentifiable adaptive bandwidth raises ValueError. H1 constant
    axes have Gaussian factor one, without an epsilon floor.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    h1 = x.ndim == 2 or y.ndim == 2
    if h1:
        if x.ndim == 1 and not x.size:
            x = x.reshape(0, 2)
        if y.ndim == 1 and not y.size:
            y = y.reshape(0, 2)
        if any(d.ndim != 2 or d.shape[1] != 2 for d in (x, y)):
            raise ValueError("Both H1 diagrams must have shape (bars,2)")
    if bandwidth_method == DEFAULT_BANDWIDTH_METHOD:
        if bandwidth is not None:
            raise ValueError(
                "Pair bandwidth is computed from the two PDs; use fixed mode for an override"
            )
        sigma, metadata = (
            h1_pair_bandwidth(x, y) if h1 else persistence_variance_bandwidth([x, y])
        )
    elif bandwidth_method == "fixed":
        if bandwidth is None:
            raise ValueError("Explicit positive bandwidth required in fixed mode")
        sigma = np.asarray(bandwidth, dtype=float) if h1 else float(bandwidth)
        metadata = {"method": "fixed"}
        if h1 and sigma.shape != (2,):
            raise ValueError("H1 fixed bandwidth requires birth and lifetime widths")
    else:
        raise ValueError("Unknown own-own bandwidth method")
    kwargs = (
        {"zero_bandwidth_policy": "constant_axis"}
        if h1 and bandwidth_method == DEFAULT_BANDWIDTH_METHOD
        else {}
    )
    cost = np.array(
        [
            entropy_excess(x, y, sigma, response_scale, **kwargs),
            entropy_excess(y, x, sigma, response_scale, **kwargs),
        ]
    )
    if h1:
        metadata = dict(metadata, homology="H1", estimator_method=H1_METHOD)
    return cost, sigma, dict(metadata, method=bandwidth_method)
