"""Forward delay embedding and the frozen reference-code Theiler convention."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import ndtr


def embed(data, dimension, delay):
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all():
        raise ValueError("Expected finite (N,2) observations X,Y")
    n = len(data) - (dimension - 1) * delay
    if n <= 0:
        raise ValueError("Insufficient observations for embedding")
    return [
        np.column_stack([data[j * delay : j * delay + n, c] for j in range(dimension)])
        for c in (0, 1)
    ]


def kde_quantile(data, gridsize=2049):
    data = np.asarray(data, dtype=float)
    if (
        data.ndim != 2
        or data.shape[1] != 2
        or len(data) < 2
        or not np.isfinite(data).all()
    ):
        raise ValueError("Expected finite (N,2) data")
    q = np.empty_like(data)
    for j in (0, 1):
        z = data[:, j]
        h = z.std(ddof=1) * len(z) ** (-0.2)
        if not np.isfinite(h) or h <= 0:
            raise ValueError("Constant/degenerate observation")
        grid = np.linspace(z.min(), z.max(), gridsize)
        cdf = np.empty(gridsize)
        for st in range(0, gridsize, 64):
            cdf[st : st + 64] = ndtr((grid[st : st + 64, None] - z[None, :]) / h).mean(
                1
            )
        q[:, j] = np.interp(z, grid, cdf)
    return q


def neighbor_indices(points, anchors, k, full_window):
    """k neighbors without anchor. Interior excludes +/-ceil(full_window/2)
    with the exact asymmetric/boundary sequence of the frozen R port.
    full_window=68 excludes [a-34,a+34]; this is NOT a radius68.
    """
    points = np.asarray(points, dtype=float)
    anchors = np.asarray(anchors)
    if points.ndim != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Invalid points")
    if (
        anchors.ndim != 1
        or not len(anchors)
        or not np.issubdtype(anchors.dtype, np.integer)
        or np.any(anchors < 0)
        or np.any(anchors >= len(points))
    ):
        raise ValueError("Invalid anchors")
    for name, value in [("k", k), ("full_window", full_window)]:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < (1 if name == "k" else 0)
        ):
            raise ValueError("Invalid " + name)
    n = len(points)
    half = int(np.ceil(full_window / 2))
    count = min(n, k + full_window + 1)
    candidates = cKDTree(points).query(points[anchors], k=count)[1]
    if candidates.ndim == 1:
        candidates = candidates[:, None]
    result = []
    for anchor, row in zip(anchors, candidates):
        i = int(anchor) + 1
        if i - half < 1:
            low, high = 1, full_window + 1
        elif i - half + full_window + 1 > n:
            low, high = n - full_window + 1, n
        else:
            low, high = i - half, i - half + full_window
        row = row[row != anchor][: k + full_window]
        selected = row[((row + 1) < low) | ((row + 1) > high)][:k]
        if len(selected) != k:
            raise ValueError("Insufficient eligible neighbors")
        result.append(selected)
    return np.asarray(result)
