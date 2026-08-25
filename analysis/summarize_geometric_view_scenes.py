#!/usr/bin/env python3
"""Aggregate Phase 1.5 metrics across multiple HAC++ scenes."""

from argparse import ArgumentParser
import csv
import json
from pathlib import Path


SCORES = (
    "binary_jaccard",
    "geometric_distance",
    "geometric_direction",
    "geometric_image",
    "geometric_depth",
    "geometric_composite",
)
ATTRIBUTES = ("feature_cosine", "feature_l2", "scaling_l2", "offset_l2")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    scene_metrics = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    rows = []
    for metrics in scene_metrics:
        for score in SCORES:
            for attribute in ATTRIBUTES:
                trend = metrics["controlled_trends"][score][attribute]
                rows.append({
                    "scene": metrics["scene"],
                    "score_type": score,
                    "attribute": attribute,
                    "supporting_bins": trend["supporting_bins"],
                    "valid_bins": trend["valid_bins"],
                    "support_fraction": trend["support_fraction"],
                    "mean_high_minus_low": trend["mean_high_minus_low"],
                })
    with (output / "cross_scene_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scaling_lines = []
    for score in SCORES:
        fields = []
        deltas = []
        fractions = []
        for metrics in scene_metrics:
            trend = metrics["controlled_trends"][score]["scaling_l2"]
            matched = metrics["controlled_pool"]["matched_pair_count_by_score"][score]
            fields.append(
                f"{metrics['scene']}: {trend['supporting_bins']}/{trend['valid_bins']}, "
                f"delta={trend['mean_high_minus_low']:.6f}, n={matched:,}"
            )
            deltas.append(trend["mean_high_minus_low"])
            fractions.append(trend["support_fraction"])
        consistent = all(delta < 0 for delta in deltas)
        scaling_lines.append(
            f"| {score} | {'; '.join(fields)} | {sum(fractions)/len(fractions):.3f} "
            f"| {sum(deltas)/len(deltas):.6f} | {'yes' if consistent else 'no'} |"
        )

    summary = f"""# Phase 1.5 Cross-Scene Summary

Scenes: {', '.join(metrics['scene'] for metrics in scene_metrics)}.

| Score | Per-scene Scaling result | Mean support fraction | Mean high-low delta | Negative delta in every scene |
|---|---|---:|---:|---:|
{chr(10).join(scaling_lines)}

The table is descriptive. Scaling uses lower L2 as the expected direction. Feature remains observation-only, and this report does not authorize an entropy-model implementation.
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
