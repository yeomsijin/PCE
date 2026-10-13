"""Full Euclidean Vietoris-Rips H0/H1 over F2, with no point subsampling."""

import numpy as np
from ripser import ripser


def diagrams(cloud, scale, h1_min_persistence=1e-10):
    cloud = np.asarray(cloud, dtype=float)
    if cloud.ndim != 2 or len(cloud) < 2 or not np.isfinite(cloud).all():
        raise ValueError("Invalid point cloud")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid PD scale")
    if not np.isfinite(h1_min_persistence) or h1_min_persistence < 0:
        raise ValueError("Invalid H1 threshold")
    d = ripser(cloud, maxdim=1, coeff=2)["dgms"]
    h0 = d[0][np.isfinite(d[0][:, 1]), 1].astype(float) / scale
    # Ripser omits zero-length bars at coincident points. They carry zero weight,
    # but restore them for fixed-cardinality sorted-H0 calibration.
    if len(h0) > len(cloud) - 1:
        raise RuntimeError("Unexpected finite H0 count")
    h0 = np.r_[np.zeros(len(cloud) - 1 - len(h0)), h0]
    h1 = d[1].astype(float)
    if not np.isfinite(h1).all():
        raise RuntimeError("Unexpected essential H1 in full Rips filtration")
    h1[:, 1] -= h1[:, 0]
    h1 /= scale
    h1 = h1[h1[:, 1] > h1_min_persistence]
    return h0, h1
