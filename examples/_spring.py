"""Original Figure 2 RK4, raw reconstructions and alpha-complex H1."""

import numpy as np
import gudhi as gd

EPS = 1e-12


def remove_small_persistence(D, min_persistence=0.0):
    D = np.asarray(D, dtype=float)

    if len(D) == 0:
        return np.empty((0, 2), dtype=float)

    return D[D[:, 1] > min_persistence]


def birth_death_to_birth_persistence(D_bd):
    """
    Convert birth-death coordinates to birth-persistence coordinates.

    Input:
        D_bd = [(birth, death)]

    Output:
        D_bp = [(birth, persistence)]
    """
    D_bd = np.asarray(D_bd, dtype=float)

    if len(D_bd) == 0:
        return np.empty((0, 2), dtype=float)

    D_bd = D_bd[np.isfinite(D_bd[:, 1])]

    if len(D_bd) == 0:
        return np.empty((0, 2), dtype=float)

    birth = D_bd[:, 0]
    death = D_bd[:, 1]
    pers = death - birth

    D_bp = np.column_stack([birth, pers])
    D_bp = D_bp[pers > 0]

    return D_bp


def normalize_diagram_by_max_persistence(D):
    """
    Per-diagram max-persistence normalization.

    D is in birth-persistence coordinates.
    After normalization:
        max persistence = 1.
    """
    D = np.asarray(D, dtype=float)

    if len(D) == 0:
        return np.empty((0, 2), dtype=float)

    max_p = np.max(D[:, 1])

    if max_p <= EPS:
        return D

    return D / max_p


def spring_expected_regime(alpha, beta, tol=1e-12):
    """
    Convention:
        alpha controls B -> A
        beta  controls A -> B
    """
    a_on = abs(alpha) > tol
    b_on = abs(beta) > tol

    if not a_on and not b_on:
        return "independent"

    if not a_on and b_on:
        return "A_to_B"

    if a_on and not b_on:
        return "B_to_A"

    return "bidirectional"


def spring_rhs(z, alpha, beta):
    """
    Coupled spring--mass system.

        x'' = -x + 0.7 alpha y
        y'' = -0.7 y + beta x

    State ordering:
        z = (x, x', y, y')

    Convention:
        alpha : B -> A
        beta  : A -> B
    """
    x, vx, y, vy = z

    dx = vx
    dvx = -x + 0.7 * alpha * y

    dy = vy
    dvy = -0.7 * y + beta * x

    return np.array([dx, dvx, dy, dvy], dtype=float)


def rk4_integrate_spring(
    alpha,
    beta,
    initial_state=(1.0, 0.0, 0.0, 1.0),
    dt=0.01,
    n_steps=10000,
):
    """
    Fixed-step RK4 integration.

    Returns:
        traj shape = (n_steps, 4)

    Each row:
        (x, x', y, y')
    """
    z = np.asarray(initial_state, dtype=float)

    if z.shape != (4,):
        raise ValueError("initial_state must have shape (4,).")

    traj = np.empty((n_steps, 4), dtype=float)

    for i in range(n_steps):
        k1 = spring_rhs(z, alpha, beta)
        k2 = spring_rhs(z + 0.5 * dt * k1, alpha, beta)
        k3 = spring_rhs(z + 0.5 * dt * k2, alpha, beta)
        k4 = spring_rhs(z + dt * k3, alpha, beta)

        z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        traj[i] = z

    return traj


def simulate_spring_from_grid_params(
    alpha,
    beta,
    seed=0,
    dt=0.01,
    n_steps=10000,
    sample_every=10,
    transient=0,
    noise_std=0.0,
    initial_state=(1.0, 0.0, 0.0, 1.0),
):
    """
    Kaji-style fixed-step simulation.

    Default:
        dt = 0.01
        n_steps = 10000
        sample_every = 10

    This gives 1000 samples before delay embedding.

    seed is used only if noise_std > 0.
    """
    rng = np.random.default_rng(seed)

    traj = rk4_integrate_spring(
        alpha=alpha,
        beta=beta,
        initial_state=initial_state,
        dt=dt,
        n_steps=n_steps,
    )

    A = traj[:, 0]
    B = traj[:, 2]

    if transient > 0:
        A = A[transient:]
        B = B[transient:]

    A = A[::sample_every]
    B = B[::sample_every]

    if noise_std > 0:
        A = A + noise_std * rng.normal(size=len(A))
        B = B + noise_std * rng.normal(size=len(B))

    return A, B


def takens_embedding_1d(x, embed_dim=4, lag=1):
    """
    Scalar Takens embedding:

        X(t) = (x_t, x_{t+tau}, ..., x_{t+(E-1)tau}) in R^E
    """
    x = np.asarray(x, dtype=float).ravel()

    n = len(x) - (embed_dim - 1) * lag

    if n <= 0:
        raise ValueError(f"Time series too short for embed_dim={embed_dim}, lag={lag}.")

    cols = [x[i * lag : i * lag + n] for i in range(embed_dim)]

    return np.column_stack(cols)


def paper_style_reconstructions(
    A,
    B,
    embed_dim=4,
    lag=1,
    max_points=1000,
    seed=0,
):
    """
    Paper-style reconstruction.

    Individual:
        X_A in R^E:
            (A_t, A_{t+tau}, ..., A_{t+(E-1)tau})

        X_B in R^E:
            (B_t, B_{t+tau}, ..., B_{t+(E-1)tau})

    Joint:
        X_AB in R^E, not R^{2E}:
            (A_t, ..., A_{t+(E/2-1)tau},
             B_t, ..., B_{t+(E/2-1)tau})

    Therefore:
        embed_dim must be even.

    Important:
        X_A, X_B, X_AB are subsampled using the same time indices.
    """
    if embed_dim % 2 != 0:
        raise ValueError(
            "Paper-style joint reconstruction requires even EMBED_DIM. "
            f"Got EMBED_DIM={embed_dim}. Use 4, 6, 8, ..."
        )

    # Kaji et al. use the raw delay reconstructions and normalize
    # each persistence diagram by its maximum lifetime afterward.
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()

    XA = takens_embedding_1d(
        A,
        embed_dim=embed_dim,
        lag=lag,
    )

    XB = takens_embedding_1d(
        B,
        embed_dim=embed_dim,
        lag=lag,
    )

    half_dim = embed_dim // 2

    n_joint = min(len(A), len(B)) - (half_dim - 1) * lag

    if n_joint <= 0:
        raise ValueError(
            f"Time series too short for joint reconstruction: "
            f"embed_dim={embed_dim}, lag={lag}."
        )

    A_part = [A[i * lag : i * lag + n_joint] for i in range(half_dim)]

    B_part = [B[i * lag : i * lag + n_joint] for i in range(half_dim)]

    XAB = np.column_stack(A_part + B_part)

    n = min(len(XA), len(XB), len(XAB))

    XA = XA[:n]
    XB = XB[:n]
    XAB = XAB[:n]

    if max_points is not None and n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        idx = np.sort(idx)

        XA = XA[idx]
        XB = XB[idx]
        XAB = XAB[idx]

    return {
        "A": XA,
        "B": XB,
        "AB": XAB,
    }


def persistence_diagram_from_cloud_alpha(
    X,
    homology_dim=1,
    min_persistence=0.0,
    normalize=True,
    output_squared_values=False,
):
    """
    Compute an alpha-complex persistence diagram using GUDHI.

    Input:
        X : point cloud, shape (n_points, ambient_dim)

    Output:
        birth-persistence coordinates.

    GUDHI stores squared alpha-filtration radii. When
    output_squared_values=False, convert them to the radius convention
    used in the Kaji--Bando--Yaguchi formulation before normalization.
    """
    X = np.asarray(X, dtype=float)

    if len(X) < 3:
        return np.empty((0, 2), dtype=float)

    alpha_complex = gd.AlphaComplex(points=X)
    simplex_tree = alpha_complex.create_simplex_tree()
    simplex_tree.compute_persistence()
    D_bd = simplex_tree.persistence_intervals_in_dimension(homology_dim)

    if not output_squared_values and len(D_bd) > 0:
        D_bd = np.sqrt(np.maximum(D_bd, 0.0))
    D_bp = birth_death_to_birth_persistence(D_bd)

    if normalize:
        D_bp = normalize_diagram_by_max_persistence(D_bp)

    D_bp = remove_small_persistence(
        D_bp,
        min_persistence=min_persistence,
    )

    return D_bp
