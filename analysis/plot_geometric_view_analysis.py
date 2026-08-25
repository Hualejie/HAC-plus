#!/usr/bin/env python3
"""Plot Phase 1.5 binary/geometric score comparisons."""

from argparse import ArgumentParser
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCORES = (
    "binary_jaccard",
    "geometric_distance",
    "geometric_direction",
    "geometric_image",
    "geometric_depth",
    "geometric_composite",
)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))

    labels = [name.replace("geometric_", "geo-").replace("binary_jaccard", "binary") for name in SCORES]
    scaling = [metrics["controlled_trends"][name]["scaling_l2"] for name in SCORES]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(labels, [item["supporting_bins"] for item in scaling], color="#4472C4")
    axes[0].set_ylim(0, 10.5)
    axes[0].set_ylabel("Bins in expected direction (of 10)")
    axes[0].set_title("Controlled Scaling consistency")
    axes[1].bar(labels, [item["mean_high_minus_low"] for item in scaling], color="#ED7D31")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("High-score − matched-low Scaling L2")
    axes[1].set_title("Controlled Scaling effect")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "scaling_score_comparison.png", dpi=180)
    plt.close(fig)

    with (output / "pair_samples.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, score, label in zip(axes.flat, SCORES, labels):
        values = np.asarray([float(row[score]) for row in rows])
        ax.hist(values, bins=50, color="#70AD47", alpha=0.85)
        ax.set_title(f"{label}: std={values.std():.4f}")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("View-score dynamic range on a fixed pair sample")
    fig.tight_layout()
    fig.savefig(plots / "score_distributions.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
