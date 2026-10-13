"""Finite VR H0 lifetimes via float64 Euclidean minimum spanning trees.

Includes zero-length edges for coincident points, matching the established
k-1 padded-bar convention. No threshold or perturbation is introduced.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

METHOD = "float64_euclidean_mst_v1"


def batch_lifetimes(clouds, scale=1.0):
    clouds = np.asarray(clouds, dtype=np.float64)
    if clouds.ndim != 3 or not np.isfinite(clouds).all():
        raise ValueError("Expected finite batch of point clouds")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Positive scale required")
    n, k, _ = clouds.shape
    if k < 1:
        raise ValueError("Point cloud must contain a point")
    distances = np.array([squareform(pdist(c, metric="euclidean")) for c in clouds])
    used = np.zeros((n, k), bool)
    used[:, 0] = True
    best = distances[:, 0, :].copy()
    best[:, 0] = np.inf
    edges = np.empty((n, k - 1), dtype=np.float64)
    rows = np.arange(n)
    for j in range(k - 1):
        ix = np.argmin(best, axis=1)
        edges[:, j] = best[rows, ix]
        used[rows, ix] = True
        best = np.minimum(best, distances[rows, ix, :])
        best[used] = np.inf
    return np.sort(edges, axis=1) / scale


def lifetimes(cloud, scale=1.0):
    return batch_lifetimes(np.asarray(cloud)[None, :, :], scale)[0]


def all_h0(space, neighbors, ks, scale):
    return {k: batch_lifetimes(space[neighbors[k]], scale) for k in ks}
