"""TPCE: induced probability on the target PD support, not KL(p_X || p_Y)."""

import numpy as np


def _validated(target, explanation, bandwidth, eta, zero_bandwidth_policy="error"):
    a = np.asarray(target, dtype=float)
    b = np.asarray(explanation, dtype=float)
    if a.ndim == 1:
        a = np.column_stack([np.zeros(len(a)), a])
    if b.ndim == 1:
        b = np.column_stack([np.zeros(len(b)), b])
    for d in (a, b):
        if d.ndim != 2 or d.shape[1] != 2 or not np.isfinite(d).all() or np.any(d < 0):
            raise ValueError("PD coordinates must be finite nonnegative birth,lifetime")
    sig = np.atleast_1d(np.asarray(bandwidth, dtype=float))
    if zero_bandwidth_policy not in ("error", "constant_axis"):
        raise ValueError("Unknown zero bandwidth policy")
    if sig.shape not in [(1,), (2,)] or not np.isfinite(sig).all() or np.any(sig < 0):
        raise ValueError("Positive bandwidth(s) required")
    if np.any(sig == 0):
        if zero_bandwidth_policy == "error":
            raise ValueError("Positive bandwidth(s) required")
        # A zero-variance axis is constant on the union of positive-mass PD
        # supports. Its Gaussian factor equals 1 for every positive width,
        # so omit it exactly; never insert an arbitrary epsilon bandwidth.
        points = np.concatenate([a[a[:, 1] > 0], b[b[:, 1] > 0]], axis=0)
        axes = [1] if len(sig) == 1 else [0, 1]
        for axis, s in zip(axes, sig):
            if s == 0 and len(points) and np.any(points[:, axis] != points[0, axis]):
                raise ValueError(
                    "Zero bandwidth requires a constant positive-mass support axis"
                )
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("Positive eta required")
    return a[a[:, 1] > 0], b[b[:, 1] > 0], sig


def _response(at, diagram, sig):
    axes = [1] if len(sig) == 1 else [0, 1]
    if np.any(sig == 0):
        axes = np.asarray(axes)[sig > 0]
        sig = sig[sig > 0]
    delta = (at[:, None, axes] - diagram[None, :, axes]) / sig
    return np.exp(-0.5 * np.sum(delta * delta, axis=2)) @ (
        diagram[:, 1] / (1 + diagram[:, 1])
    )


def induced_probability(
    target, explanation, bandwidth, eta=5.0, *, zero_bandwidth_policy="error"
):
    """Return target p, induced log-q on target support, unexplained mass.
    q_i=p_i exp(-c_i), c_i=(F_B(u_i)-F_A(u_i))²/(2 eta²).
    p assigns zero mass to the extra unexplained event. q is normalized ONLY
    after including that extra event; do not renormalize q on target bars.
    Empty/zero-total-persistence targets use a unit-mass empty event
    and the established mass-coarsening induced probability.
    """
    a, b, sig = _validated(target, explanation, bandwidth, eta, zero_bandwidth_policy)
    if not len(a):
        cost = entropy_excess(
            a, b, sig, eta, zero_bandwidth_policy=zero_bandwidth_policy
        )
        return np.ones(1), np.array([-cost]), float(-np.expm1(-cost))
    p = a[:, 1] / a[:, 1].sum()
    c = (_response(a, b, sig) - _response(a, a, sig)) ** 2 / (2 * eta * eta)
    return p, np.log(p) - c, float(p @ (-np.expm1(-c)))


def explanation_discrepancy(target, explanation, bandwidth):
    """Return F_explanation - F_target on positive target bars.

    Useful for fitting one global response scale in a fixed-bandwidth paper
    experiment. Empty targets have no discrepancy coordinates.
    """
    a, b, sig = _validated(target, explanation, bandwidth, 1.0)
    return _response(a, b, sig) - _response(a, a, sig)


def entropy_excess(
    target,
    explanation,
    bandwidth,
    eta=5.0,
    *,
    empty_policy="mass_coarsening",
    zero_bandwidth_policy="error",
):
    """Established empty-PD extension, for both H0 and H1.
    Both empty: 0. Empty target: coarsen the reverse induced retained mass.
    Zero-persistence bars are ignored and therefore equivalent to empty PD.
    """
    if empty_policy != "mass_coarsening":
        raise ValueError("TPCE uses mass_coarsening for empty PDs")
    a, b, sig = _validated(target, explanation, bandwidth, eta, zero_bandwidth_policy)
    if len(a):
        p = a[:, 1] / a[:, 1].sum()
        return float(
            p @ ((_response(a, b, sig) - _response(a, a, sig)) ** 2) / (2 * eta * eta)
        )
    if not len(b):
        return 0.0
    p = b[:, 1] / b[:, 1].sum()
    c = _response(b, b, sig) ** 2 / (2 * eta * eta)
    minimum = float(c.min())
    lost = float(p @ (-np.expm1(-(c - minimum))))
    if lost < 0.5:
        return float(minimum - np.log1p(-lost))
    return float(minimum - np.log(p @ np.exp(-(c - minimum))))


def direction(delta_x_given_y, delta_y_given_x):
    """Positive convention X->Y. Zero/nonfinite denominator stays undefined."""
    a, b = np.broadcast_arrays(
        np.asarray(delta_x_given_y, float), np.asarray(delta_y_given_x, float)
    )
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("Negative entropy excess")
    den = a + b
    valid = np.isfinite(a) & np.isfinite(b) & (den > 0)
    return np.divide(b - a, den, out=np.full_like(den, np.nan), where=valid)
