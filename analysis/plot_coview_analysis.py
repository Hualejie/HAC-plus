#!/usr/bin/env python3
"""Generate Phase 1 plots from the committed aggregate CSV files."""

from argparse import ArgumentParser
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_neighbor_comparison(output: Path) -> None:
    rows = _read_csv(output / "neighbor_comparison.csv")
    metrics = ["feature_cosine", "feature_l2", "scaling_l2", "offset_l2"]
    kinds = ["coview", "spatial", "random"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(axes.flat, metrics):
        values = [float(next(row["mean"] for row in rows if row["metric"] == metric and row["neighbor_type"] == kind)) for kind in kinds]
        ax.bar(kinds, values, color=["#4472C4", "#ED7D31", "#A5A5A5"])
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Attribute similarity by neighbour family")
    fig.tight_layout()
    fig.savefig(output / "plots" / "neighbor_comparison.png", dpi=180)
    plt.close(fig)


def plot_binned_correlations(output: Path) -> None:
    rows = _read_csv(output / "distance_binned_correlations.csv")
    metrics = ["feature_cosine", "feature_l2", "scaling_l2", "offset_l2"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in metrics:
        selected = sorted((row for row in rows if row["metric"] == metric), key=lambda row: int(row["distance_bin"]))
        ax.plot(
            [int(row["distance_bin"]) for row in selected],
            [float(row["spearman"]) for row in selected],
            marker="o",
            label=metric.replace("_", " "),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Spatial-distance quantile bin")
    ax.set_ylabel("Spearman correlation with CoView score")
    ax.set_title("CoView/attribute trend after spatial-distance binning")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "plots" / "distance_binned_correlations.png", dpi=180)
    plt.close(fig)


def plot_controlled_high_low(output: Path) -> None:
    rows = _read_csv(output / "controlled_high_low.csv")
    metrics = ["feature_cosine", "feature_l2", "scaling_l2", "offset_l2"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(axes.flat, metrics):
        selected = sorted((row for row in rows if row["metric"] == metric), key=lambda row: int(row["distance_bin"]))
        bins = np.asarray([int(row["distance_bin"]) for row in selected])
        ax.plot(bins, [float(row["low_metric_mean"]) for row in selected], marker="o", label="low CoView")
        ax.plot(bins, [float(row["high_metric_mean"]) for row in selected], marker="o", label="high CoView")
        ax.set_title(metric.replace("_", " "))
        ax.grid(alpha=0.25)
    axes.flat[0].legend()
    fig.suptitle("Distance-controlled high/low CoView pairs")
    fig.tight_layout()
    fig.savefig(output / "plots" / "controlled_high_low.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="analysis_output")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    (output / "plots").mkdir(parents=True, exist_ok=True)
    plot_neighbor_comparison(output)
    plot_binned_correlations(output)
    plot_controlled_high_low(output)


if __name__ == "__main__":
    main()
