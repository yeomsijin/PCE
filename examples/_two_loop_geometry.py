"""Original Figure 1 point generator, extracted without notebook execution."""

import numpy as np

CASE_ORDER = ["reference", "clean", "large loop", "small loop", "noise"]


def loop_points(center, radius, n, noise, rng):
    if n <= 0:
        return np.empty((0, 2))

    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    t = t + rng.normal(0, 0.015, size=n)

    pts = np.column_stack(
        [
            center[0] + radius * np.cos(t),
            center[1] + radius * np.sin(t),
        ]
    )

    pts += rng.normal(0, noise, size=pts.shape)
    return pts


def thick_loop_points(
    center,
    radius,
    n_base,
    base_noise,
    thickness=0.0,
    layers=0,
    points_per_layer=12,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    parts = [loop_points(center, radius, n_base, base_noise, rng)]

    if thickness <= 0 or layers <= 0 or points_per_layer <= 0:
        return np.vstack(parts)

    for k in range(1, int(layers) + 1):
        sign = -1 if k % 2 == 1 else 1
        frac = k / max(layers, 1)
        rk = radius + sign * thickness * frac

        parts.append(
            loop_points(
                center=center,
                radius=rk,
                n=int(points_per_layer),
                noise=base_noise + 0.25 * thickness,
                rng=rng,
            )
        )

    return np.vstack(parts)


def mini_circle(center, radius, n=8, noise=0.004, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    t = t + rng.normal(0, 0.025, size=n)

    pts = np.column_stack(
        [
            center[0] + radius * np.cos(t),
            center[1] + radius * np.sin(t),
        ]
    )

    pts += rng.normal(0, noise, size=pts.shape)
    return pts


def make_mini_circle_bundle(centers, radii, n_each=8, noise=0.004, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if len(centers) == 0:
        return np.empty((0, 2))

    return np.vstack(
        [
            mini_circle(
                center=np.asarray(c, dtype=float),
                radius=float(r),
                n=n_each,
                noise=noise,
                rng=rng,
            )
            for c, r in zip(centers, radii)
        ]
    )


def diffuse_points(n, box=(-2.2, 2.2, -1.6, 1.6), rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if n <= 0:
        return np.empty((0, 2))

    xmin, xmax, ymin, ymax = box
    return np.column_stack(
        [
            rng.uniform(xmin, xmax, n),
            rng.uniform(ymin, ymax, n),
        ]
    )


def default_geometry():
    return {
        "large_center": (-1.25, 0.0),
        "large_radius": 0.80,
        "small_center": (1.20, 0.0),
        "small_radius": 0.42,
        "noise_centers": [
            (-0.45, -1.05),
            (0.10, 1.05),
            (0.60, -1.05),
            (1.65, 0.95),
            (-1.80, 0.95),
            (1.85, -0.90),
            (-1.55, -1.10),
            (0.95, 1.15),
        ],
        "noise_radii": [0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115],
        "off_noise_centers": [
            (-1.85, -0.95),
            (-0.85, 1.10),
            (0.95, 1.10),
            (1.85, -1.05),
            (-0.20, -1.20),
            (1.45, -1.20),
        ],
        "off_noise_radii": [0.020, 0.025, 0.030, 0.035, 0.040, 0.045],
    }


CASE_DEFAULTS = {
    "reference": {
        "use_large": True,
        "use_small": True,
        "large_n": 90,
        "small_n": 70,
        "large_noise": 0.014,
        "small_noise": 0.012,
        "large_thickness": 0.0,
        "small_thickness": 0.0,
        "thickness_layers": 0,
        "thickness_points": 12,
        "noise_micro_n": 6,
        "off_micro_n": 0,
        "micro_noise": 0.004,
        "diffuse_n": 0,
    },
    "clean": {
        "use_large": True,
        "use_small": True,
        "large_n": 90,
        "small_n": 70,
        "large_noise": 0.010,
        "small_noise": 0.009,
        "large_thickness": 0.0,
        "small_thickness": 0.0,
        "thickness_layers": 0,
        "thickness_points": 12,
        "noise_micro_n": 0,
        "off_micro_n": 0,
        "micro_noise": 0.004,
        "diffuse_n": 0,
    },
    "large loop": {
        "use_large": True,
        "use_small": False,
        "large_n": 95,
        "small_n": 0,
        "large_noise": 0.011,
        "small_noise": 0.040,
        "large_thickness": 0.0,
        "small_thickness": 0.0,
        "thickness_layers": 0,
        "thickness_points": 12,
        "noise_micro_n": 0,
        "off_micro_n": 2,
        "micro_noise": 0.004,
        "diffuse_n": 6,
    },
    "small loop": {
        "use_large": False,
        "use_small": True,
        "large_n": 0,
        "small_n": 80,
        "large_noise": 0.050,
        "small_noise": 0.009,
        "large_thickness": 0.0,
        "small_thickness": 0.0,
        "thickness_layers": 0,
        "thickness_points": 12,
        "noise_micro_n": 0,
        "off_micro_n": 2,
        "micro_noise": 0.004,
        "diffuse_n": 6,
    },
    "noise": {
        "use_large": False,
        "use_small": False,
        "large_n": 0,
        "small_n": 0,
        "large_noise": 0.080,
        "small_noise": 0.080,
        "large_thickness": 0.0,
        "small_thickness": 0.0,
        "thickness_layers": 0,
        "thickness_points": 12,
        "noise_micro_n": 6,
        "off_micro_n": 0,
        "micro_noise": 0.004,
        "diffuse_n": 10,
    },
}


def make_case_cloud(case, cfg=None, seed=7):
    if case not in CASE_DEFAULTS:
        raise ValueError(f"Unknown case: {case}")

    if cfg is None:
        cfg = {}

    rng = np.random.default_rng(seed)
    geom = default_geometry()

    p = {**CASE_DEFAULTS[case], **cfg}
    parts = []

    if p["use_large"] and p["large_n"] > 0:
        parts.append(
            thick_loop_points(
                center=geom["large_center"],
                radius=geom["large_radius"],
                n_base=int(p["large_n"]),
                base_noise=float(p["large_noise"]),
                thickness=float(p["large_thickness"]),
                layers=int(p["thickness_layers"]),
                points_per_layer=int(p["thickness_points"]),
                rng=rng,
            )
        )

    if p["use_small"] and p["small_n"] > 0:
        parts.append(
            thick_loop_points(
                center=geom["small_center"],
                radius=geom["small_radius"],
                n_base=int(p["small_n"]),
                base_noise=float(p["small_noise"]),
                thickness=float(p["small_thickness"]),
                layers=int(p["thickness_layers"]),
                points_per_layer=int(p["thickness_points"]),
                rng=rng,
            )
        )

    n_micro = int(p["noise_micro_n"])
    if n_micro > 0:
        parts.append(
            make_mini_circle_bundle(
                centers=geom["noise_centers"][:n_micro],
                radii=geom["noise_radii"][:n_micro],
                n_each=8,
                noise=float(p["micro_noise"]),
                rng=rng,
            )
        )

    n_off = int(p["off_micro_n"])
    if n_off > 0:
        parts.append(
            make_mini_circle_bundle(
                centers=geom["off_noise_centers"][:n_off],
                radii=geom["off_noise_radii"][:n_off],
                n_each=8,
                noise=float(p["micro_noise"]),
                rng=rng,
            )
        )

    if int(p["diffuse_n"]) > 0:
        parts.append(diffuse_points(int(p["diffuse_n"]), rng=rng))

    parts = [x for x in parts if len(x) > 0]

    if len(parts) == 0:
        return np.empty((0, 2))

    return np.vstack(parts)


def make_all_clouds(tuning=None, seed=7):
    if tuning is None:
        tuning = {}

    return {
        case: make_case_cloud(
            case,
            cfg=tuning.get(case, {}),
            seed=seed + 100 * i,
        )
        for i, case in enumerate(CASE_ORDER)
    }
