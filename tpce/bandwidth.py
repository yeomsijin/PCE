"""Shared equal-diagram, lifetime-weighted mixture moments.

Callers validate diagrams and decide how to handle zero variance. Keeping that
policy outside this helper preserves the different H0 and H1 conventions.
"""

import numpy as np


def mixture_moments(diagrams, *, lifetime_axis):
    """Return mean, variance, nonempty count and empty count.

    Use lifetime_axis=None for H0 lifetimes, or 1 for H1 birth/lifetime rows.
    Center before squaring to retain precision for large coordinate offsets.
    """
    means = []
    variances = []
    empty = 0
    for values in diagrams:
        lifetime = values if lifetime_axis is None else values[:, lifetime_axis]
        values = values[lifetime > 0]
        lifetime = lifetime[lifetime > 0]
        if not len(values):
            empty += 1
            continue
        weights = lifetime / lifetime.max()
        weights /= weights.sum()
        origin = values[0]
        center = weights @ (values - origin)
        means.append(origin + center)
        variances.append(weights @ ((values - origin - center) ** 2))
    if not means:
        shape = () if lifetime_axis is None else (2,)
        return np.full(shape, np.nan), np.zeros(shape), 0, empty
    means = np.asarray(means)
    origin = means[0]
    center = np.mean(means - origin, axis=0)
    variance = np.mean(np.asarray(variances) + (means - origin - center) ** 2, axis=0)
    return origin + center, variance, len(means), empty


def persistence_variance_bandwidth(lifetime_diagrams):
    """sigma²=E[(L-L')²]=2 Var_Q(L), Q = equal mixture of PD p_D.

    Each nonempty PD gets equal mass; internally p_D is proportional to
    positive lifetime. Empty/zero-mass PDs cannot define p_D: condition the
    mixture on nonempty PDs and record the omission. No epsilon fallback.
    Two-pass mixture variance retains within- AND between-diagram variation.
    """
    diagrams = []
    for values in lifetime_diagrams:
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Expected finite nonnegative lifetimes")
        diagrams.append(values)
    mean, variance, nonempty, empty = mixture_moments(diagrams, lifetime_axis=None)
    if not nonempty:
        raise ValueError(
            "Unidentifiable H0 bandwidth: no positive-persistence calibration diagrams"
        )
    sigma = float(np.sqrt(2) * np.sqrt(variance))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(
            "Unidentifiable H0 bandwidth: zero/nonfinite persistence variance"
        )
    return sigma, dict(
        nonempty_diagrams=nonempty,
        empty_diagrams=empty,
        mixture_mean=float(mean),
        mixture_variance=float(variance),
    )
