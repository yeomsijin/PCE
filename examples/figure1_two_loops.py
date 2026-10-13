"""Reproduce Figure 1 from the original geometry and fixed paper bandwidths.

Run: python -m examples.figure1_two_loops --output example_outputs/figure1
"""

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from tpce.persistence import diagrams as persistence_diagrams
from tpce.entropy import explanation_discrepancy
from ._common import metrics, save_records
from ._two_loop_geometry import make_all_clouds, CASE_ORDER

# Original final notebook configuration; these are geometry parameters,
# not adjustments to the PCE definition or a runtime search for matching PE.
TUNING = {
    "reference": dict(noise_micro_n=10, micro_noise=0.072, diffuse_n=15),
    "clean": dict(
        large_thickness=0.04,
        small_thickness=0.05987,
        thickness_layers=23,
        thickness_points=13,
    ),
    "large loop": dict(
        large_thickness=0.04673,
        thickness_layers=20,
        thickness_points=30,
        off_micro_n=0,
        diffuse_n=0,
    ),
    "small loop": dict(
        small_thickness=0.01930,
        thickness_layers=25,
        thickness_points=21,
        off_micro_n=0,
        diffuse_n=0,
    ),
    "noise": dict(noise_micro_n=10, micro_noise=0.03057, diffuse_n=20),
}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    clouds = make_all_clouds(TUNING, seed=7)
    diagrams = {}
    for name, cloud in clouds.items():
        _, diagram = persistence_diagrams(cloud, scale=1.0, h1_min_persistence=1e-3)
        diagrams[name] = diagram[np.argsort(-diagram[:, 1])]
    reference = diagrams["reference"]
    bandwidth = (0.5, 0.5)
    residuals = np.concatenate(
        [
            np.abs(explanation_discrepancy(reference, diagrams[name], bandwidth))
            for name in CASE_ORDER[1:]
        ]
    )
    # Preserve the original generator's positive-residual selection rule.
    response_scale = float(np.median(residuals[residuals > 1e-12]))
    records = [
        metrics(reference, diagrams[name], bandwidth, response_scale)
        for name in CASE_ORDER
    ]
    rows = []
    for name, record in zip(CASE_ORDER, records):
        diagram = diagrams[name]
        p = diagram[:, 1] / diagram[:, 1].sum()
        rows.append(
            dict(
                case=name,
                points=len(clouds[name]),
                bars=len(diagram),
                pe=float(-p @ np.log(p)),
                cross_entropy=record["cross_entropy"],
                excess=record["excess"],
                unexplained_mass=record["boundary"],
            )
        )
    save_records(
        output,
        rows,
        dict(
            figure=1,
            seeds=[7, 107, 207, 307, 407],
            bandwidth=list(bandwidth),
            bandwidth_method="fixed",
            response_scale=response_scale,
            response_scale_paper_rounded=0.298499,
            min_persistence=0.001,
            geometry="original final notebook tuning; raw coordinates; full Euclidean VR H1",
            probability="augmented; never renormalized on reference bars",
        ),
    )
    np.savez_compressed(output / "clouds.npz", **clouds)
    np.savez_compressed(output / "diagrams.npz", **diagrams)
    np.savez_compressed(
        output / "probabilities.npz",
        **{f"case{i}": r["retained"] for i, r in enumerate(records)},
    )
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    grid = fig.add_gridspec(
        4, 6, height_ratios=[1, 1, 1, 0.25], width_ratios=[1, 1, 1, 1, 1, 0.04]
    )
    norm = PowerNorm(0.7, vmin=0, vmax=max(r["retained"].max() for r in records))
    names = [
        "X: two loops + noise",
        "Y1: two loops",
        "Y2: large loop",
        "Y3: small loop",
        "Y4: noise",
    ]
    for i, (name, record, row) in enumerate(zip(CASE_ORDER, records, rows)):
        cloud = clouds[name]
        diagram = diagrams[name]
        ax = fig.add_subplot(grid[0, i])
        ax.scatter(*cloud.T, s=4, color="#253142")
        ax.set(
            xlim=(-2.45, 2.45), ylim=(-1.8, 1.8), xticks=[], yticks=[], title=names[i]
        )
        ax.set_box_aspect(1)
        ax = fig.add_subplot(grid[1, i])
        ax.scatter(*diagram.T, s=18, color="#3d4858")
        ax.text(
            0.97,
            0.95,
            f"PE = {row['pe']:.3f}",
            ha="right",
            va="top",
            transform=ax.transAxes,
        )
        ax.set(xlim=(0, 1.35), ylim=(0, 1.5), title="Own H1 diagram")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.set_ylabel("persistence")
        ax = fig.add_subplot(grid[2, i])
        sc = ax.scatter(
            *reference.T, c=record["retained"], cmap="magma", norm=norm, s=28
        )
        ax.text(
            0.97,
            0.95,
            f"PCE = {record['cross_entropy']:.3f}",
            ha="right",
            va="top",
            transform=ax.transAxes,
        )
        ax.set(
            xlim=(0, 1.35), ylim=(0, 1.5), xlabel="birth", title="Induced mass on DX"
        )
        ax.grid(alpha=0.2)
        if i == 0:
            ax.set_ylabel("persistence")
        ax = fig.add_subplot(grid[3, i])
        keep = 1 - record["boundary"]
        ax.barh(0, keep, color="#3f7f78")
        ax.barh(0, record["boundary"], left=keep, color="#df755f")
        ax.set(
            xlim=(0, 1),
            yticks=[],
            xticks=[0, 0.5, 1],
            title=f"retained {keep:.3f}\nunexplained {record['boundary']:.3f}",
        )
    fig.colorbar(sc, cax=fig.add_subplot(grid[2, 5]), label="Probability mass")
    fig.savefig(output / "figure1.png", dpi=180)
    plt.close(fig)
    print(rows)
    print("response_scale", response_scale)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="example_outputs/figure1")
    run(parser.parse_args().output)
