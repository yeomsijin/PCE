"""Default TC: main-text neighborhoods of the mapping's SOURCE space.

For causal i->j, select neighbors in j and estimate M_{j->i} forward.
The PCA threshold is 99%, without whitening, ridge, or a dimension floor of2.
The historical inverse-on-cause-neighborhood implementation remains under
experiments.tpce_reference.tc and is not this default.
"""

from dataclasses import dataclass
import numpy as np
from .geometry import neighbor_indices

METHOD = "tc_main_text_source_neighbors_v1"


def mapping_logs(source, destination):
    """Forward log expansion of paired patches, shape (batch,k,m).

    source must be a neighborhood in the mapping's domain; destination uses
    exactly the same time indices. This function does not choose neighbors.
    """
    a = np.asarray(source, dtype=float)
    b = np.asarray(destination, dtype=float)
    if a.ndim != 3 or b.ndim != 3 or a.shape[:2] != b.shape[:2] or a.shape[1] < 2:
        raise ValueError("Expected paired (batch,k,m) patches, k>=2")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Patches must be finite")
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    _, sa, va = np.linalg.svd(a, full_matrices=False)
    _, sb, vb = np.linalg.svd(b, full_matrices=False)

    def dimensions(s):
        energy = s * s
        total = energy.sum(1, keepdims=True)
        fraction = np.divide(
            energy.cumsum(1), total, out=np.ones_like(s), where=total > 0
        )
        return np.minimum((fraction < 0.99).sum(1) + 1, s.shape[1])

    dims = np.maximum(dimensions(sa), dimensions(sb))
    if np.any(dims > min(va.shape[1], vb.shape[1])):
        raise ValueError("Incompatible PCA dimensions between spaces")
    logs = np.empty(len(a))
    for d in np.unique(dims):
        mask = dims == d
        aa = a[mask] @ va[mask, :d, :].swapaxes(1, 2)
        bb = b[mask] @ vb[mask, :d, :].swapaxes(1, 2)
        _, _, v = np.linalg.svd(np.concatenate([aa, bb], axis=2), full_matrices=False)
        p = v[:, :d, :].swapaxes(1, 2)
        p_source = p[:, :d, :]
        p_target = p[:, d:, :]
        values = []
        for src, dst in zip(p_source, p_target):
            try:
                matrix = np.linalg.solve(src.T, dst.T).T  # P_target inv(P_source)
                values.append(
                    np.log(
                        np.maximum(1.0, np.linalg.svd(matrix, compute_uv=False))
                    ).sum()
                )
            except np.linalg.LinAlgError:
                values.append(float("inf"))
        logs[mask] = values
    return logs, dims


def logs_from_neighbors(spaces, neighbors, batch=64):
    """Return (logs,dims), shape (2,n), indexed by CAUSAL direction.

    row0: log e_{1->0}, for C_{0->1}; indices come from neighbors[1].
    row1: log e_{0->1}, for C_{1->0}; indices come from neighbors[0].
    neighbors must contain each space's OWN neighbor index arrays.
    """
    if len(spaces) != 2 or len(neighbors) != 2 or batch < 1:
        raise ValueError("Two spaces/neighborhood arrays and positive batch required")
    nn = [np.asarray(n) for n in neighbors]
    if (
        any(n.ndim != 2 or not np.issubdtype(n.dtype, np.integer) for n in nn)
        or nn[0].shape != nn[1].shape
    ):
        raise ValueError("Two equally shaped integer neighbor arrays required")
    if len(spaces[0]) != len(spaces[1]) or any(
        np.any(n < 0) or np.any(n >= len(spaces[0])) for n in nn
    ):
        raise ValueError("Spaces must be time aligned; indices must be in range")
    n = len(nn[0])
    logs = np.empty((2, n))
    dims = np.empty((2, n), dtype=int)
    for cause in (0, 1):
        effect = 1 - cause
        for start in range(0, n, batch):
            ids = nn[effect][start : start + batch]
            logs[cause, start : start + batch], dims[cause, start : start + batch] = (
                mapping_logs(spaces[effect][ids], spaces[cause][ids])
            )
    return logs, dims


@dataclass
class TCResult:
    logs: np.ndarray
    dimensions: np.ndarray
    neighbors: tuple
    C_local: np.ndarray
    C: np.ndarray
    alpha_local: np.ndarray
    alpha: float
    method: str = METHOD


def estimate_tc(spaces, anchors, k=20, theiler_full_window=0, batch=64):
    """TC for two already-preprocessed, time-aligned reconstructed spaces.

    C uses 1/(1+mean log expansion), not mean(C_local).
    alpha positive means space0->space1. Both logs zero => alpha NaN, C=1.
    No chance-level correction is silently applied.
    """
    if len(spaces) != 2 or k < 2:
        raise ValueError("Two spaces and k>=2 required")
    spaces = [np.asarray(s, dtype=float) for s in spaces]
    nn = tuple(neighbor_indices(s, anchors, k, theiler_full_window) for s in spaces)
    logs, dims = logs_from_neighbors(spaces, nn, batch)
    den = logs.sum(0)
    valid = np.isfinite(logs).all(0) & (den > 0)
    alpha_local = np.full(len(den), np.nan)
    alpha_local[valid] = (logs[1, valid] - logs[0, valid]) / den[valid]
    total = logs.sum()
    alpha = (
        float((logs[1].sum() - logs[0].sum()) / total)
        if np.isfinite(total) and total > 0
        else float("nan")
    )
    return TCResult(
        logs, dims, nn, 1 / (1 + logs), 1 / (1 + logs.mean(1)), alpha_local, alpha
    )
