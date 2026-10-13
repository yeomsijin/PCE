"""Reproduce the nine panels of Figure 2, using Appendix B.2 settings.

Run: python -m examples.figure2_spring_mass --output example_outputs/figure2
All 81 settings are evaluated by default. --grid-size 2 is a smoke check only.
"""

import argparse
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tpce.entropy import explanation_discrepancy
from ._spring import (
    simulate_spring_from_grid_params,
    paper_style_reconstructions,
    persistence_diagram_from_cloud_alpha,
    spring_expected_regime,
)
from ._common import metrics, save_records, separation_r2
from . import _baselines as baseline

VECTORS = {
    "Betti": baseline.betti_curve_vector,
    "Landscape": baseline.persistence_landscape_vector,
    "Silhouette": baseline.persistence_silhouette_vector,
    "PI": baseline.persistence_image_vector,
}
PANELS = [
    ("PCE", "Entropy excess"),
    ("Boundary", "Unexplained mass"),
    ("Bottleneck", "Bottleneck"),
    ("Wasserstein", "2-Wasserstein"),
    ("Betti", "Betti curve"),
    ("Landscape", "Persistence landscape"),
    ("Silhouette", "Persistence silhouette"),
    ("PSSK", "Persistence scale-space kernel"),
    ("PI", "Persistence image"),
]


def compute_diagrams(task):
    alpha, beta, cache = task
    path = cache / f"a{alpha:.10f}_b{beta:.10f}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as source:
            diagrams = {name: source[name] for name in ("A", "B", "AB")}
    else:
        a, b = simulate_spring_from_grid_params(alpha, beta)
        clouds = paper_style_reconstructions(a, b, embed_dim=4, lag=1, max_points=None)
        assert all(cloud.shape == (997, 4) for cloud in clouds.values())
        diagrams = {
            name: persistence_diagram_from_cloud_alpha(cloud)
            for name, cloud in clouds.items()
        }
        # Replace atomically so interrupted runs cannot leave readable partial caches.
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **diagrams)
        temporary.replace(path)
    if any(not len(d) for d in diagrams.values()):
        raise ValueError(f"Empty H1 at {alpha},{beta}; no invented costs")
    return float(alpha), float(beta), diagrams


def run(output, grid_size=9, workers=4):
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "diagrams"
    cache.mkdir(exist_ok=True)
    grid = np.round(np.linspace(0, 0.8, grid_size), 10)
    entries = []
    pooled = []
    tasks = [(alpha, beta, cache) for alpha in grid for beta in grid]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for alpha, beta, diagrams in pool.map(compute_diagrams, tasks):
            for target, other in [("A", "B"), ("B", "A")]:
                pooled.extend(
                    np.abs(
                        explanation_discrepancy(
                            diagrams[target], diagrams[other], (0.4, 0.15)
                        )
                    )
                )
            entries.append((alpha, beta, diagrams))
            print(
                f"diagrams {len(entries)}/{grid_size**2}: alpha={alpha}, beta={beta}",
                flush=True,
            )
    response_scale = float(np.median(pooled))
    pssk_sigma = baseline.median_scale_space_sigma(
        [d for _, _, ds in entries for d in ds.values()]
    )
    rows = []
    for alpha, beta, ds in entries:
        row = dict(alpha=alpha, beta=beta, regime=spring_expected_regime(alpha, beta))
        # Horizontal is E(B,A), vertical E(A,B), matching Figure 2's labels.
        for side, target, other in [("x", "B", "A"), ("y", "A", "B")]:
            record = metrics(ds[target], ds[other], (0.4, 0.15), response_scale)
            row["PCE_" + side] = record["excess"]
            row["Boundary_" + side] = record["boundary"]
        for side, name in [("x", "A"), ("y", "B")]:
            for metric, fn in [
                ("Bottleneck", baseline.bottleneck_distance),
                ("Wasserstein", baseline.wasserstein_distance),
            ]:
                row[metric + "_" + side] = fn(ds["AB"], ds[name])
            for metric, vector in VECTORS.items():
                a, b = vector(ds["AB"]), vector(ds[name])
                row[metric + "_" + side] = float(
                    np.linalg.norm(a - b) / np.sqrt(len(a))
                )
            row["PSSK_" + side] = baseline.persistence_scale_space_distance(
                ds["AB"], ds[name], pssk_sigma
            )
        rows.append(row)
        print(f"metrics {len(rows)}/{len(entries)}", flush=True)
    save_records(
        output,
        rows,
        dict(
            figure=2,
            grid=grid.tolist(),
            dt=0.01,
            n_steps=10000,
            sample_every=10,
            first_sample_time=0.01,
            last_sample_time=99.91,
            initial_state=[1, 0, 0, 1],
            embedding_dimension=4,
            delay=1,
            points_per_cloud=997,
            complex="GUDHI AlphaComplex H1",
            normalization="each diagram divided by its own maximum lifetime",
            bandwidth=[0.4, 0.15],
            bandwidth_method="fixed",
            response_scale=response_scale,
            response_scale_paper_rounded=2.001779,
            pssk_sigma=pssk_sigma,
            pssk_sigma_paper_rounded=0.016325,
            response_calibration="median absolute discrepancy over both directions and all grid settings",
            paper_reproduction=grid_size == 9,
            diagram_cache="generated only by this example; full fixed settings",
        ),
    )
    labels = np.array([r["regime"] for r in rows])
    oneway = np.isin(labels, ["A_to_B", "B_to_A"])
    summary = {}
    for prefix, _ in PANELS:
        coordinates = np.array([[r[prefix + "_x"], r[prefix + "_y"]] for r in rows])
        summary[prefix] = dict(
            four_regimes_r2=separation_r2(coordinates, labels),
            two_directions_r2=separation_r2(coordinates[oneway], labels[oneway]),
        )
    correct = sum(
        (r["PCE_x"] > r["PCE_y"]) == (r["regime"] == "A_to_B")
        for r in rows
        if r["regime"] in ["A_to_B", "B_to_A"]
    )
    summary["oneway"] = dict(correct=int(correct), total=int(oneway.sum()))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(rows, output / "figure2.png")
    print(json.dumps(summary, indent=2))
    return rows


def plot(rows, path):
    colors = {
        "independent": "tab:blue",
        "A_to_B": "tab:orange",
        "B_to_A": "tab:green",
        "bidirectional": "tab:red",
    }
    labels = {
        "independent": "independent",
        "A_to_B": "A → B",
        "B_to_A": "B → A",
        "bidirectional": "bidirectional",
    }
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), layout="constrained")
    for ax, (prefix, title) in zip(axes.flat, PANELS):
        for regime in ["A_to_B", "B_to_A", "bidirectional", "independent"]:
            selection = [r for r in rows if r["regime"] == regime]
            ax.scatter(
                [r[prefix + "_x"] for r in selection],
                [r[prefix + "_y"] for r in selection],
                color=colors[regime],
                label=labels[regime],
                s=28,
                alpha=0.85,
                zorder=10 if regime == "independent" else 5,
            )
        upper = (
            1.0
            if prefix == "Boundary"
            else 1.08 * max(r[prefix + "_" + side] for r in rows for side in ["x", "y"])
        )
        ax.plot([0, upper], [0, upper], "--", alpha=0.5)
        ax.set(
            xlim=(0, upper),
            ylim=(0, upper),
            title=title,
            xlabel=(
                "E(B,A)"
                if prefix == "PCE"
                else ("unexplained B from A" if prefix == "Boundary" else "d(DAB, DA)")
            ),
            ylabel=(
                "E(A,B)"
                if prefix == "PCE"
                else ("unexplained A from B" if prefix == "Boundary" else "d(DAB, DB)")
            ),
        )
        ax.set_box_aspect(1)
        ax.grid(alpha=0.2)
    handles, legendlabels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legendlabels, loc="outside upper center", ncol=4)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="example_outputs/figure2")
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.output, args.grid_size, args.workers)
