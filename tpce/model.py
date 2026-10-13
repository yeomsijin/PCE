"""Historical time-induced TPCE; current PCE lives in own_own.py."""

from dataclasses import dataclass, asdict, field
import numpy as np
from scipy.spatial.distance import pdist, cdist
from .geometry import embed, kde_quantile, neighbor_indices
from .persistence import diagrams
from .entropy import entropy_excess, direction
from .bandwidth import persistence_variance_bandwidth


@dataclass(frozen=True)
class TPCEConfig:
    embedding_dimension: int = 5
    delay: int = 17
    neighbors: int = 20
    theiler_full_window: int = None
    normalization: str = "kde_cdf"
    calibration_k: int = None
    bandwidth_method: str = "persistence_variance"
    calibration_anchors: int = 64
    scale_sample_size: int = 256
    h1_min_persistence: float = 1e-10
    response_scale: float = 5.0

    def __post_init__(self):
        for name in (
            "embedding_dimension",
            "delay",
            "neighbors",
            "calibration_anchors",
            "scale_sample_size",
        ):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(name + " must be a positive integer")
        if self.calibration_anchors < 2 or self.scale_sample_size < 2:
            raise ValueError("At least two calibration samples required")
        if self.calibration_k is not None and (
            isinstance(self.calibration_k, bool)
            or not isinstance(self.calibration_k, int)
            or self.calibration_k < 1
        ):
            raise ValueError("calibration_k must be positive or None")
        if self.bandwidth_method not in ("persistence_variance", "legacy_local_std"):
            raise ValueError("Unknown bandwidth method")
        v = self.theiler_full_window
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
            raise ValueError("Invalid full Theiler window")
        if self.normalization not in ("none", "kde_cdf"):
            raise ValueError("normalization must be none or kde_cdf")
        if not np.isfinite(self.h1_min_persistence) or self.h1_min_persistence < 0:
            raise ValueError("Invalid H1 floor")
        if not np.isfinite(self.response_scale) or self.response_scale <= 0:
            raise ValueError("Invalid response scale")

    @property
    def window(self):
        return (
            (self.embedding_dimension - 1) * self.delay
            if self.theiler_full_window is None
            else self.theiler_full_window
        )


@dataclass(frozen=True)
class Calibration:
    scales: tuple
    h0: float
    birth: float = None
    lifetime: float = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if (
            len(self.scales) != 2
            or not np.isfinite(self.scales).all()
            or min(self.scales) <= 0
        ):
            raise ValueError("Positive X,Y scales required")
        if not np.isfinite(self.h0) or self.h0 <= 0:
            raise ValueError("Positive H0 bandwidth required")
        if (self.birth is None) != (self.lifetime is None):
            raise ValueError("H1 bandwidths must both be given or unavailable")
        if self.birth is not None and (
            not np.isfinite([self.birth, self.lifetime]).all()
            or min(self.birth, self.lifetime) <= 0
        ):
            raise ValueError("Positive H1 bandwidths required")

    @classmethod
    def from_dict(cls, d):
        return cls(
            tuple(d["scales"]),
            d["h0"],
            d.get("birth"),
            d.get("lifetime"),
            d.get("metadata", {}),
        )


def prepare(data, config):
    data = np.asarray(data, float)
    if data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all():
        raise ValueError("Expected finite (N,2) observations")
    q = kde_quantile(data) if config.normalization == "kde_cdf" else data
    return embed(q, config.embedding_dimension, config.delay)


def _calibrate_legacy(spaces, config):
    """Own-neighborhood PDs only, fixed calibration_k independent of test k.
    No fallback to arbitrary scales/bandwidths. H1 stays unavailable when its
    calibration has no variation; this does not mean zero H1 excess.
    """
    n = len(spaces[0])
    scales = []
    for s in spaces:
        ix = np.linspace(
            0, len(s) - 1, min(config.scale_sample_size, len(s)), dtype=int
        )
        d = pdist(s[ix])
        positive = d[d > 0]
        if not len(positive):
            raise ValueError("Unidentifiable PD scale: constant calibration sample")
        scales.append(float(np.median(positive)))
    aa = np.linspace(0, n - 1, min(config.calibration_anchors, n), dtype=int)
    diff = []
    allh1 = []
    for s, scale in zip(spaces, scales):
        nn = neighbor_indices(
            s,
            aa,
            100 if config.calibration_k is None else config.calibration_k,
            config.window,
        )
        group = []
        for a, ix in zip(aa, nn):
            h0, h1 = diagrams(s[np.r_[a, ix]], scale, config.h1_min_persistence)
            group.append(np.sort(h0))
            allh1.append(h1)
        dist = cdist(s[aa], s[aa])
        dist[abs(aa[:, None] - aa[None, :]) <= int(np.ceil(config.window / 2))] = np.inf
        eligible = np.isfinite(dist).any(1)
        pairs = np.unique(
            np.sort(
                np.column_stack([np.flatnonzero(eligible), dist[eligible].argmin(1)]),
                axis=1,
            ),
            axis=0,
        )
        v = (np.array(group)[pairs[:, 0]] - np.array(group)[pairs[:, 1]]).ravel()
        diff.extend([v, -v])
    values = np.concatenate(diff)
    h0 = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    if not np.isfinite(h0) or h0 <= 0:
        raise ValueError("Unidentifiable H0 bandwidth: no calibration variation")
    points = np.concatenate(allh1)
    band = []
    for axis in [0, 1]:
        v = np.sort(points[:, axis])
        v = v[np.linspace(0, len(v) - 1, min(256, len(v)), dtype=int)]
        d = abs(v[:, None] - v[None, :])[np.triu_indices(len(v), 1)]
        d = d[d > 0]
        band.append(float(np.median(d)) if len(d) else None)
    if None in band:
        band = [None, None]
    return Calibration(
        tuple(scales),
        h0,
        *band,
        metadata=dict(
            method="legacy_local_std",
            calibration_k=100 if config.calibration_k is None else config.calibration_k,
        ),
    )


def calibrate(spaces, config):
    """Fit once, freeze for all costs: equal-PD, persistence-weighted H0 variance.

    All four clouds (X, Y|X, Y, X|Y) at fixed anchors and the evaluation k.
    Explicit calibration_k overrides k; explicit Calibration bypasses this.
    Existing per-space reference-distance preprocessing is unchanged.
    H1 retains its previous own-PD coordinate-distance rule; no H1 theory change.
    """
    if config.bandwidth_method == "legacy_local_std":
        return _calibrate_legacy(spaces, config)
    n = len(spaces[0])
    scales = []
    for s in spaces:
        ix = np.linspace(
            0, len(s) - 1, min(config.scale_sample_size, len(s)), dtype=int
        )
        positive = pdist(s[ix])
        positive = positive[positive > 0]
        if not len(positive):
            raise ValueError("Unidentifiable PD scale: constant calibration sample")
        scales.append(float(np.median(positive)))
    aa = np.linspace(0, n - 1, min(config.calibration_anchors, n), dtype=int)
    k = config.neighbors if config.calibration_k is None else config.calibration_k
    neighbors = [neighbor_indices(s, aa, k, config.window) for s in spaces]
    lifetime_diagrams = []
    own_h1 = []
    for target in (0, 1):
        for anchor, indices in zip(aa, neighbors[target]):
            rows = np.r_[anchor, indices]
            for space in (target, 1 - target):
                h0, h1 = diagrams(
                    spaces[space][rows], scales[space], config.h1_min_persistence
                )
                lifetime_diagrams.append(h0)
                if space == target:
                    own_h1.append(h1)
    h0, metadata = persistence_variance_bandwidth(lifetime_diagrams)
    metadata.update(
        method="persistence_variance",
        formula="sigma^2=2 Var_Q(lifetime)",
        mixture="equal nonempty PD mass; within PD p=lifetime/sum(lifetime)",
        clouds=["X", "Y|X", "Y", "X|Y"],
        calibration_k=k,
        points_per_cloud=k + 1,
        anchors=aa.tolist(),
        reference_scales=scales,
        empty_policy="condition mixture on nonempty positive-persistence diagrams",
        H1_method="legacy own-PD positive coordinate-distance medians",
    )
    points = np.concatenate(own_h1)
    band = []
    for axis in (0, 1):
        v = np.sort(points[:, axis])
        v = v[np.linspace(0, len(v) - 1, min(256, len(v)), dtype=int)]
        d = abs(v[:, None] - v[None, :])[np.triu_indices(len(v), 1)]
        d = d[d > 0]
        band.append(float(np.median(d)) if len(d) else None)
    if None in band:
        band = [None, None]
    return Calibration(tuple(scales), h0, *band, metadata=metadata)


def paired_costs(spaces, anchor, indices_by_target, calibration, config):
    """cost[H,target X/Y]. Always reference=target, never own-cross."""
    values = np.full((2, 2), np.nan)
    counts = np.empty((2, 2), int)
    h0_degenerate = np.zeros(2, bool)
    for target in [0, 1]:
        rows = np.r_[anchor, indices_by_target[target]]
        a0, a1 = diagrams(
            spaces[target][rows], calibration.scales[target], config.h1_min_persistence
        )
        b0, b1 = diagrams(
            spaces[1 - target][rows],
            calibration.scales[1 - target],
            config.h1_min_persistence,
        )
        values[0, target] = entropy_excess(
            a0, b0, calibration.h0, config.response_scale
        )
        if calibration.birth is not None:
            values[1, target] = entropy_excess(
                a1,
                b1,
                (calibration.birth, calibration.lifetime),
                config.response_scale,
                empty_policy="mass_coarsening",
            )
        h0_degenerate[target] = not np.any(a0 > 0)
        counts[target] = len(a1), len(b1)
    return values, counts, h0_degenerate


@dataclass
class TPCEResult:
    anchors: np.ndarray
    cost: np.ndarray  # H,target X/Y,time
    S_PCE: np.ndarray  # H,time
    denominator: np.ndarray
    S_R: np.ndarray  # H, ratio of summed costs, not mean of ratios
    neighbors: np.ndarray  # target X/Y,time,k
    h1_counts: np.ndarray  # target X/Y,target/explanation,time
    h0_degenerate: (
        np.ndarray
    )  # target X/Y,time; empty-PD extension used, not missing cost
    calibration: Calibration
    config: TPCEConfig


def estimate(data, config=None, *, anchors=None, calibration=None):
    config = TPCEConfig() if config is None else config
    spaces = prepare(data, config)
    n = len(spaces[0])
    aa = np.arange(n) if anchors is None else np.asarray(anchors)
    if (
        aa.ndim != 1
        or not len(aa)
        or not np.issubdtype(aa.dtype, np.integer)
        or np.any(aa < 0)
        or np.any(aa >= n)
        or len(np.unique(aa)) != len(aa)
    ):
        raise ValueError("Unique valid embedding-start anchors required")
    cal = calibrate(spaces, config) if calibration is None else calibration
    if not isinstance(cal, Calibration):
        raise ValueError("Use a validated Calibration object")
    nn = np.array(
        [neighbor_indices(s, aa, config.neighbors, config.window) for s in spaces]
    )
    costs = np.empty((2, 2, len(aa)))
    counts = np.empty((2, 2, len(aa)), int)
    degenerate = np.empty((2, len(aa)), bool)
    for i, a in enumerate(aa):
        costs[:, :, i], counts[:, :, i], degenerate[:, i] = paired_costs(
            spaces, a, nn[:, i], cal, config
        )
    S = direction(costs[:, 0], costs[:, 1])
    den = costs.sum(1)
    # Missing cost invalidates the aggregate, rather than silently discarding observations.
    total = costs.sum(-1)
    SR = direction(total[:, 0], total[:, 1])
    return TPCEResult(aa.copy(), costs, S, den, SR, nn, counts, degenerate, cal, config)
