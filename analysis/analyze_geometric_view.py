#!/usr/bin/env python3
"""Phase 1.5 comparison of binary CoView and geometric view context."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, get_combined_args
from analysis.analyze_coview import (
    METRIC_NAMES,
    _controlled_rows,
    _load_model,
    _pair_metrics,
    _write_csv,
)
from scene.coview_context import (
    build_geometric_observation_descriptors,
    build_geometry_observations,
    canonicalize_codec_anchors,
    pair_geometric_view_scores,
    spatial_topk_queries,
)


SCORE_NAMES = (
    "binary_jaccard",
    "geometric_distance",
    "geometric_direction",
    "geometric_image",
    "geometric_depth",
    "geometric_composite",
)


def _trend_summary(rows):
    result = {}
    for metric_name in METRIC_NAMES:
        selected = [row for row in rows if row["metric"] == metric_name]
        expected_positive = metric_name == "feature_cosine"
        supporting = sum(
            (row["high_minus_low"] > 0) if expected_positive else (row["high_minus_low"] < 0)
            for row in selected
        )
        result[metric_name] = {
            "supporting_bins": int(supporting),
            "valid_bins": len(selected),
            "support_fraction": float(supporting / len(selected)) if selected else float("nan"),
            "mean_high_minus_low": (
                float(np.mean([row["high_minus_low"] for row in selected]))
                if selected else float("nan")
            ),
        }
    return result


def _score_statistics(values):
    quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "max": float(quantiles[4]),
        "unique_value_count": int(np.unique(values).size),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    model_params = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--n_features", type=int, default=None)
    parser.add_argument("--log2", type=int, default=None)
    parser.add_argument("--log2_2D", type=int, default=None)
    parser.add_argument("--controlled_k", type=int, default=64)
    parser.add_argument("--controlled_anchor_samples", type=int, default=20_000)
    parser.add_argument("--distance_bins", type=int, default=10)
    parser.add_argument("--distance_match_caliper", type=float, default=None)
    parser.add_argument("--observation_chunk_size", type=int, default=262_144)
    parser.add_argument("--score_batch_size", type=int, default=4096)
    parser.add_argument("--image_sigma", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=str, required=True)
    args = get_combined_args(parser)
    args.n_features = 4 if getattr(args, "n_features", None) is None else args.n_features
    args.log2 = 13 if getattr(args, "log2", None) is None else args.log2
    args.log2_2D = 15 if getattr(args, "log2_2D", None) is None else args.log2_2D

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    dataset, scene, gaussians = _load_model(args, model_params)
    train_cameras = scene.getTrainCameras()
    if not train_cameras:
        raise RuntimeError("scene.getTrainCameras() returned no cameras")

    with torch.no_grad():
        mask = gaussians.get_mask_anchor.to(torch.bool)[:, 0]
        aligned = canonicalize_codec_anchors(
            gaussians.get_anchor,
            mask,
            gaussians.voxel_size,
            attributes={
                "feature": gaussians._anchor_feat,
                "scaling": gaussians.get_scaling,
                "offset": gaussians._offset * gaussians.get_mask,
            },
        )
    xyz = aligned.xyz.detach().cpu().numpy().astype(np.float64, copy=False)
    attributes = {
        name: value.detach().cpu().numpy().astype(np.float32, copy=False)
        for name, value in aligned.attributes.items()
    }
    num_anchors = xyz.shape[0]
    if args.controlled_k >= num_anchors:
        raise ValueError("controlled_k must be smaller than the valid anchor count")

    relation = build_geometry_observations(
        aligned.xyz,
        train_cameras,
        chunk_size=args.observation_chunk_size,
    )
    descriptors = build_geometric_observation_descriptors(
        aligned.xyz,
        train_cameras,
        relation,
    )

    rng = np.random.default_rng(args.seed)
    sample_count = min(args.controlled_anchor_samples, num_anchors)
    sampled_anchor = np.sort(rng.choice(num_anchors, sample_count, replace=False))
    neighbors, distances = spatial_topk_queries(xyz, sampled_anchor, k=args.controlled_k)
    pair_source = np.repeat(sampled_anchor, args.controlled_k)
    pair_target = neighbors.reshape(-1)
    pair_distance = distances.reshape(-1)
    pair_metrics = _pair_metrics(attributes, pair_source, pair_target)
    scores = pair_geometric_view_scores(
        descriptors,
        pair_source,
        pair_target,
        image_sigma=args.image_sigma,
        batch_size=args.score_batch_size,
    )
    caliper = (
        float(gaussians.voxel_size)
        if getattr(args, "distance_match_caliper", None) is None
        else float(args.distance_match_caliper)
    )

    all_controlled_rows = []
    all_correlation_rows = []
    trends = {}
    matched_pair_counts = {}
    for score_name in SCORE_NAMES:
        controlled_rows, correlation_rows, _ = _controlled_rows(
            pair_source,
            pair_distance,
            scores[score_name],
            pair_metrics,
            args.distance_bins,
            caliper,
        )
        for row in controlled_rows:
            row["score_type"] = score_name
        for row in correlation_rows:
            row["score_type"] = score_name
        all_controlled_rows.extend(controlled_rows)
        all_correlation_rows.extend(correlation_rows)
        trends[score_name] = _trend_summary(controlled_rows)
        matched_pair_counts[score_name] = sum(
            int(row["high_count"])
            for row in controlled_rows
            if row["metric"] == "feature_cosine"
        )

    score_stats = {name: _score_statistics(scores[name]) for name in SCORE_NAMES}
    common_count = scores["common_camera_count"]
    score_stats["common_camera_count"] = _score_statistics(common_count)

    # Bounded raw sample contains identical pairs for every score definition.
    raw_count = min(20_000, pair_source.size)
    raw_idx = np.sort(rng.choice(pair_source.size, raw_count, replace=False))
    raw_rows = []
    for idx in raw_idx:
        raw_rows.append({
            "source": int(pair_source[idx]),
            "target": int(pair_target[idx]),
            "spatial_distance": float(pair_distance[idx]),
            **{name: float(scores[name][idx]) for name in (*SCORE_NAMES, "common_camera_count")},
            **{name: float(pair_metrics[name][idx]) for name in METRIC_NAMES},
        })

    comparison = {}
    for score_name in SCORE_NAMES:
        scaling = trends[score_name]["scaling_l2"]
        offset = trends[score_name]["offset_l2"]
        feature = trends[score_name]["feature_cosine"]
        comparison[score_name] = {
            "scaling_supporting_bins": scaling["supporting_bins"],
            "scaling_mean_high_minus_low": scaling["mean_high_minus_low"],
            "offset_supporting_bins": offset["supporting_bins"],
            "feature_cosine_supporting_bins": feature["supporting_bins"],
        }
    binary_scaling = trends["binary_jaccard"]["scaling_l2"]
    geometric_scaling = trends["geometric_composite"]["scaling_l2"]
    geometric_better = (
        geometric_scaling["supporting_bins"] > binary_scaling["supporting_bins"]
        or (
            geometric_scaling["supporting_bins"] == binary_scaling["supporting_bins"]
            and geometric_scaling["mean_high_minus_low"] < binary_scaling["mean_high_minus_low"]
        )
    )

    observed_counts = np.diff(relation.anchor_indptr)
    metrics_json = {
        "phase": "1.5",
        "scene": Path(dataset.source_path).name,
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": int(scene.loaded_iter),
        "train_camera_count": len(train_cameras),
        "valid_codec_anchor_count": int(num_anchors),
        "voxel_size": float(gaussians.voxel_size),
        "controlled_pool": {
            "sampled_anchor_count": int(sample_count),
            "spatial_neighbors_per_anchor": int(args.controlled_k),
            "pair_count": int(pair_source.size),
            "distance_match_caliper": caliper,
            "matched_pair_count_by_score": matched_pair_counts,
        },
        "observation": {
            "edge_count": int(relation.anchor_camera_ids.size),
            "unobserved_anchor_count": int((observed_counts == 0).sum()),
            "mean_cameras_per_anchor": float(observed_counts.mean()),
            "descriptor_fields": ["distance", "view_direction", "image_xy", "depth"],
            "decoder_reconstructable": True,
            "dense_anchor_camera_matrix_persisted": False,
            "dense_anchor_pair_matrix_created": False,
        },
        "kernel": {
            "distance": "exp(-abs(d_i-d_k)/(0.5*(abs(d_i)+abs(d_k))+eps))",
            "direction": "0.5*(clamp(dot(u_i,u_k),-1,1)+1)",
            "image": f"exp(-squared_ndc_distance/{args.image_sigma}^2)",
            "depth": "exp(-abs(z_i-z_k)/(0.5*(abs(z_i)+abs(z_k))+eps))",
            "composite": "equal mean of distance, direction, image, depth",
        },
        "score_statistics": score_stats,
        "controlled_trends": trends,
        "comparison": comparison,
        "geometric_composite_improves_scaling_over_binary": bool(geometric_better),
        "runtime_seconds": float(time.time() - started),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_json, handle, indent=2, allow_nan=True)

    controlled_fields = [
        "score_type", "distance_bin", "distance_min", "distance_max", "pair_count",
        "metric", "low_count", "low_unique_count", "high_count", "source_anchor_count",
        "low_coview_mean", "high_coview_mean", "low_distance_mean", "high_distance_mean",
        "mean_absolute_distance_mismatch", "low_metric_mean", "high_metric_mean",
        "high_minus_low",
    ]
    _write_csv(output_dir / "controlled_score_comparison.csv", all_controlled_rows, controlled_fields)
    _write_csv(
        output_dir / "distance_binned_correlations.csv",
        all_correlation_rows,
        [
            "score_type", "distance_bin", "distance_min", "distance_max", "pair_count",
            "metric", "count", "pearson", "spearman",
        ],
    )
    _write_csv(
        output_dir / "pair_samples.csv",
        raw_rows,
        [
            "source", "target", "spatial_distance", *SCORE_NAMES, "common_camera_count",
            *METRIC_NAMES,
        ],
    )

    rows = []
    for score_name in SCORE_NAMES:
        for metric_name in METRIC_NAMES:
            trend = trends[score_name][metric_name]
            rows.append(
                f"| {score_name} | {metric_name} | {trend['supporting_bins']}/{trend['valid_bins']} "
                f"| {trend['mean_high_minus_low']:.6f} |"
            )
    verdict = (
        "Geometric composite improves the controlled Scaling result over binary Jaccard."
        if geometric_better
        else "Geometric composite does not improve the controlled Scaling result over binary Jaccard."
    )
    summary = f"""# Phase 1.5 Geometric View Analysis: {Path(dataset.source_path).name}

## Scope

- Existing HAC++ checkpoint: `{Path(dataset.model_path).resolve()}` at iteration {scene.loaded_iter}
- {num_anchors:,} codec-aligned anchors and {len(train_cameras)} train cameras; test cameras were not used
- Geometry-only descriptors: distance, unit viewing direction, normalized image coordinate, camera-space depth
- Controlled pool: {pair_source.size:,} spatial-neighbour pairs; same-source matching with distance caliper {caliper}
- No training, renderer, entropy model, model structure, loss, or codec modification was performed by this analysis

## Dynamic range

- Binary Jaccard: mean {score_stats['binary_jaccard']['mean']:.6f}, std {score_stats['binary_jaccard']['std']:.6f}, unique values {score_stats['binary_jaccard']['unique_value_count']:,}
- Geometric composite: mean {score_stats['geometric_composite']['mean']:.6f}, std {score_stats['geometric_composite']['std']:.6f}, unique values {score_stats['geometric_composite']['unique_value_count']:,}

## Controlled high/low results

| Score | Attribute statistic | Expected bins | Mean high − matched-low |
|---|---|---:|---:|
{os.linesep.join(rows)}

## Scaling decision

**{verdict}** Binary Scaling: {binary_scaling['supporting_bins']}/{binary_scaling['valid_bins']} bins, mean delta {binary_scaling['mean_high_minus_low']:.6f}. Geometric composite Scaling: {geometric_scaling['supporting_bins']}/{geometric_scaling['valid_bins']} bins, mean delta {geometric_scaling['mean_high_minus_low']:.6f}.

Feature remains an observation-only metric. Phase 1.5 does not authorize or implement an entropy-model change.

Runtime: {metrics_json['runtime_seconds']:.1f} seconds.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
