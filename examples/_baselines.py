"""Seven symmetric baselines for Figure 2; no PCE estimator duplication."""

import numpy as np
import gudhi as gd

EPS = 1e-12


def birth_persistence_to_birth_death(D_bp):
    """
    Convert birth-persistence coordinates to birth-death coordinates.

    Input:
        D_bp = [(birth, persistence)]

    Output:
        D_bd = [(birth, death)]
    """
    D_bp = np.asarray(D_bp, dtype=float)

    if len(D_bp) == 0:
        return np.empty((0, 2), dtype=float)

    birth = D_bp[:, 0]
    pers = D_bp[:, 1]
    death = birth + pers

    return np.column_stack([birth, death])


def bottleneck_distance(D1, D2):
    """
    Bottleneck distance between persistence diagrams.

    Inputs:
        D1, D2 are in birth-persistence coordinates.

    Internally:
        converted to birth-death coordinates.

    Uses:
        GUDHI bottleneck_distance.
    """
    X = birth_persistence_to_birth_death(D1)
    Y = birth_persistence_to_birth_death(D2)

    return float(
        gd.bottleneck_distance(
            X,
            Y,
            e=0.0,
        )
    )


def wasserstein_distance(D1, D2, p=2):
    """Exact order-p assignment with Linfinity ground cost and diagonal matches."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    x = birth_persistence_to_birth_death(D1)
    y = birth_persistence_to_birth_death(D2)
    n, m = len(x), len(y)
    cost = np.full((n + m, n + m), np.inf)
    cost[:n, :m] = cdist(x, y, metric="chebyshev") ** p
    cost[np.arange(n), m + np.arange(n)] = (D1[:, 1] / 2) ** p
    cost[n + np.arange(m), np.arange(m)] = (D2[:, 1] / 2) ** p
    cost[n:, m:] = 0.0
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].sum() ** (1 / p))


def betti_curve_vector(
    D,
    t_range=(0.0, 1.8),
    resolution=256,
):
    D = np.asarray(D, dtype=float)
    t = np.linspace(t_range[0], t_range[1], resolution)

    if len(D) == 0:
        return np.zeros_like(t)

    curve = np.zeros_like(t, dtype=float)

    for b, p in D:
        death = b + p
        curve += ((t >= b) & (t <= death)).astype(float)

    return curve


def persistence_landscape_vector(
    D,
    t_range=(0.0, 1.8),
    resolution=256,
    n_layers=5,
):
    D = np.asarray(D, dtype=float)
    t = np.linspace(t_range[0], t_range[1], resolution)

    if len(D) == 0:
        return np.zeros(n_layers * resolution)

    tents = []

    for b, p in D:
        death = b + p
        tent = np.maximum(
            0.0,
            np.minimum(t - b, death - t),
        )
        tents.append(tent)

    tents = np.asarray(tents)
    sorted_vals = np.sort(tents, axis=0)[::-1]

    layers = np.zeros((n_layers, resolution), dtype=float)
    n_available = min(n_layers, sorted_vals.shape[0])

    layers[:n_available] = sorted_vals[:n_available]

    return layers.ravel()


def persistence_silhouette_vector(
    D,
    t_range=(0.0, 1.8),
    resolution=100,
    weight_power=0.0,
):
    D = np.asarray(D, dtype=float)
    t = np.linspace(t_range[0], t_range[1], resolution)

    if len(D) == 0:
        return np.zeros_like(t)

    # Uniform silhouette weights in Appendix B.2 / Table 4.
    tents = np.maximum(
        0.0,
        np.minimum(
            t[None, :] - D[:, 0, None], (D[:, 0] + D[:, 1])[:, None] - t[None, :]
        ),
    )
    if weight_power != 0.0:
        raise ValueError("Figure 2 uses uniform silhouette weights")
    return tents.mean(axis=0)


def persistence_image_vector(
    D,
    x_range=(0.0, 1.4),
    y_range=(0.0, 1.4),
    resolution=32,
    sigma=0.08,
    weight_power=1.0,
):
    D = np.asarray(D, dtype=float)

    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)

    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    img = np.zeros_like(Xg, dtype=float)

    if len(D) == 0:
        return img.ravel()

    for b, p in D:
        w = max(p, EPS) ** weight_power

        img += w * np.exp(-0.5 * (((Xg - b) / sigma) ** 2 + ((Yg - p) / sigma) ** 2))

    return img.ravel()


def persistence_scale_space_kernel(D1, D2, sigma):
    """Persistence scale-space kernel in birth--death coordinates."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("PSSK sigma must be positive and finite.")

    X = birth_persistence_to_birth_death(D1)
    Y = birth_persistence_to_birth_death(D2)

    if len(X) == 0 or len(Y) == 0:
        return 0.0

    Y_reflected = Y[:, ::-1]
    direct_d2 = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    reflected_d2 = np.sum(
        (X[:, None, :] - Y_reflected[None, :, :]) ** 2,
        axis=2,
    )

    return float(
        np.sum(
            np.exp(-direct_d2 / (8.0 * sigma)) - np.exp(-reflected_d2 / (8.0 * sigma))
        )
        / (8.0 * np.pi * sigma)
    )


def persistence_scale_space_distance(D1, D2, sigma):
    k11 = persistence_scale_space_kernel(D1, D1, sigma)
    k22 = persistence_scale_space_kernel(D2, D2, sigma)
    k12 = persistence_scale_space_kernel(D1, D2, sigma)
    squared_distance = max(k11 + k22 - 2.0 * k12, 0.0)
    return float(np.sqrt(squared_distance))


def median_scale_space_sigma(diagrams, max_pairs=250000, seed=7):
    """Global median heuristic: sigma = median(||u-v||^2) / 8."""
    pooled = np.vstack(
        [birth_persistence_to_birth_death(D) for D in diagrams if len(D) > 0]
    )

    pooled = pooled[np.all(np.isfinite(pooled), axis=1)]
    n = len(pooled)
    if n < 2:
        raise ValueError("At least two finite diagram points are required.")

    n_pairs = n * (n - 1) // 2
    if n_pairs <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n - 1, size=max_pairs)
        j = j + (j >= i)

    squared_distances = np.sum((pooled[i] - pooled[j]) ** 2, axis=1)
    squared_distances = squared_distances[
        np.isfinite(squared_distances) & (squared_distances > EPS)
    ]
    if len(squared_distances) == 0:
        raise ValueError("Cannot calibrate PSSK sigma from zero distances.")

    return float(np.median(squared_distances) / 8.0)
