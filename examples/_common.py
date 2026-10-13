"""Paper metrics evaluated through the shared PCE implementation."""

import csv
import json
from pathlib import Path
import numpy as np
from tpce.entropy import entropy_excess, induced_probability


def metrics(target, explanation, bandwidth, response_scale):
    p, logq, boundary = induced_probability(
        target, explanation, bandwidth, response_scale
    )
    pe = float(-p @ np.log(p))
    excess = entropy_excess(target, explanation, bandwidth, response_scale)
    retained = np.exp(logq)
    # Three independent identities: normalization, TV=boundary and CE-PE=excess.
    np.testing.assert_allclose(retained.sum() + boundary, 1.0, atol=2e-14)
    np.testing.assert_allclose(
        (np.abs(retained - p).sum() + boundary) / 2, boundary, atol=2e-14
    )
    np.testing.assert_allclose(-p @ logq - pe, excess, atol=2e-13)
    return dict(
        pe=pe,
        excess=excess,
        cross_entropy=pe + excess,
        boundary=boundary,
        retained=retained,
    )


def save_records(output, rows, settings):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "settings.json").write_text(json.dumps(settings, indent=2))


def separation_r2(values, labels):
    values = np.asarray(values)
    labels = np.asarray(labels)
    total = np.sum((values - values.mean(axis=0)) ** 2)
    within = sum(
        np.sum((values[labels == label] - values[labels == label].mean(axis=0)) ** 2)
        for label in np.unique(labels)
    )
    return float(1 - within / total) if total > 0 else float("nan")
