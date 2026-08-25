#!/usr/bin/env python3
"""Phase 1 CoView independent-information analysis for an existing HAC++ model."""

from argparse import ArgumentParser
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, get_combined_args
from scene import GaussianModel, Scene
from scene.coview_context import (
    build_geometry_observations,
    canonicalize_codec_anchors,
    coview_topk,
    pair_coview_scores,
    spatial_topk,
)


METRIC_NAMES = ("feature_cosine", "feature_l2", "scaling_l2", "offset_l2")


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _correlations(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"count": int(x.size), "pearson": float("nan"), "spearman": float("nan")}
    return {
        "count": int(x.size),
        "pearson": float(stats.pearsonr(x, y).statistic),
        "spearman": float(stats.spearmanr(x, y).statistic),
    }


def _pair_metrics(
    attributes: Mapping[str, np.ndarray],
    source: np.ndarray,
    target: np.ndarray,
    chunk_size: int = 100_000,
) -> Dict[str, np.ndarray]:
    source = np.asarray(source, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    output = {name: np.empty(source.size, dtype=np.float32) for name in METRIC_NAMES}

    feature = attributes["feature"]
    scaling = attributes["scaling"]
    offset = attributes["offset"].reshape(attributes["offset"].shape[0], -1)
    for start in range(0, source.size, chunk_size):
        end = min(start + chunk_size, source.size)
        src = source[start:end]
        dst = target[start:end]
        feature_src = feature[src]
        feature_dst = feature[dst]
        dot = np.einsum("ij,ij->i", feature_src, feature_dst)
        norm = np.linalg.norm(feature_src, axis=1) * np.linalg.norm(feature_dst, axis=1)
        output["feature_cosine"][start:end] = np.divide(
            dot, norm, out=np.zeros_like(dot), where=norm > 1e-12
        )
        output["feature_l2"][start:end] = np.linalg.norm(feature_src - feature_dst, axis=1)
        output["scaling_l2"][start:end] = np.linalg.norm(scaling[src] - scaling[dst], axis=1)
        output["offset_l2"][start:end] = np.linalg.norm(offset[src] - offset[dst], axis=1)
    return output


def _random_neighbors(num_anchors: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, num_anchors - 1, size=(num_anchors, k), dtype=np.int64)
    source = np.arange(num_anchors, dtype=np.int64)[:, None]
    neighbors = raw + (raw >= source)
    # Duplicate probability is tiny at scene scale, but make the result a true set.
    duplicate_rows = np.flatnonzero(
        np.asarray([np.unique(row).size != k for row in neighbors], dtype=bool)
    )
    for row_idx in duplicate_rows:
        neighbors[row_idx] = rng.choice(
            np.delete(np.arange(num_anchors, dtype=np.int64), row_idx), size=k, replace=False
        )
    return neighbors


def _neighbor_overlap(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.intersect1d(a, b, assume_unique=True).size / first.shape[1]
         for a, b in zip(first, second)],
        dtype=np.float32,
    )


def _distance_bins(distance: np.ndarray, num_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(distance, np.linspace(0.0, 1.0, num_bins + 1)))
    if edges.size < 2:
        return np.zeros(distance.shape, dtype=np.int32), edges
    bin_id = np.searchsorted(edges[1:-1], distance, side="right").astype(np.int32)
    return bin_id, edges


def _metric_means(metrics: Mapping[str, np.ndarray]) -> Dict[str, float]:
    return {name: float(np.nanmean(values)) for name, values in metrics.items()}


def _controlled_rows(
    source: np.ndarray,
    distance: np.ndarray,
    coview_score: np.ndarray,
    metrics: Mapping[str, np.ndarray],
    num_bins: int,
    max_distance_mismatch: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], int]:
    bin_id, edges = _distance_bins(distance, num_bins)
    controlled_rows = []
    correlation_rows = []
    # Correlations use every pair in the spatial-neighbour pool.
    for current_bin in range(max(0, edges.size - 1)):
        selected = bin_id == current_bin
        if selected.sum() < 20:
            continue
        score = coview_score[selected]
        row_base = {
            "distance_bin": current_bin,
            "distance_min": float(edges[current_bin]),
            "distance_max": float(edges[current_bin + 1]),
            "pair_count": int(selected.sum()),
        }
        for metric_name, values in metrics.items():
            corr = _correlations(score, values[selected])
            correlation_rows.append({**row_base, "metric": metric_name, **corr})

    # The high/low experiment is stricter: split scores within each source
    # anchor, then match high to low by nearest distance within that same source.
    matched_low_parts = []
    high_parts = []
    source = np.asarray(source, dtype=np.int64)
    boundaries = np.flatnonzero(np.r_[True, source[1:] != source[:-1], True])
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        local_idx = np.arange(begin, end, dtype=np.int64)
        score = coview_score[local_idx]
        q25, q75 = np.quantile(score, [0.25, 0.75])
        if q25 < q75:
            low_idx = local_idx[score <= q25]
            high_idx = local_idx[score >= q75]
        else:
            unique_scores = np.unique(score)
            if unique_scores.size < 2:
                continue
            low_idx = local_idx[score == unique_scores[0]]
            high_idx = local_idx[score == unique_scores[-1]]
        if low_idx.size == 0 or high_idx.size == 0:
            continue
        low_order = low_idx[np.argsort(distance[low_idx], kind="stable")]
        low_distance = distance[low_order]
        insertion = np.searchsorted(low_distance, distance[high_idx])
        right_pos = np.clip(insertion, 0, low_order.size - 1)
        left_pos = np.clip(insertion - 1, 0, low_order.size - 1)
        choose_right = (
            np.abs(low_distance[right_pos] - distance[high_idx])
            < np.abs(low_distance[left_pos] - distance[high_idx])
        )
        matched_pos = np.where(choose_right, right_pos, left_pos)
        matched_low = low_order[matched_pos]
        within_caliper = np.abs(distance[matched_low] - distance[high_idx]) <= max_distance_mismatch
        if within_caliper.any():
            matched_low_parts.append(matched_low[within_caliper])
            high_parts.append(high_idx[within_caliper])

    if not high_parts:
        return controlled_rows, correlation_rows, 0
    matched_low_idx = np.concatenate(matched_low_parts)
    high_idx = np.concatenate(high_parts)
    matched_bin_id, matched_edges = _distance_bins(distance[high_idx], num_bins)
    valid_controlled_bins = 0
    for current_bin in range(max(0, matched_edges.size - 1)):
        selected = matched_bin_id == current_bin
        if selected.sum() < 20:
            continue
        current_low = matched_low_idx[selected]
        current_high = high_idx[selected]
        row_base = {
            "distance_bin": current_bin,
            "distance_min": float(matched_edges[current_bin]),
            "distance_max": float(matched_edges[current_bin + 1]),
            "pair_count": int(selected.sum()),
        }
        valid_controlled_bins += 1
        for metric_name, values in metrics.items():
            controlled_rows.append({
                **row_base,
                "metric": metric_name,
                "low_count": int(current_low.size),
                "low_unique_count": int(np.unique(current_low).size),
                "high_count": int(current_high.size),
                "source_anchor_count": int(np.unique(source[current_high]).size),
                "low_coview_mean": float(coview_score[current_low].mean()),
                "high_coview_mean": float(coview_score[current_high].mean()),
                "low_distance_mean": float(distance[current_low].mean()),
                "high_distance_mean": float(distance[current_high].mean()),
                "mean_absolute_distance_mismatch": float(
                    np.abs(distance[current_low] - distance[current_high]).mean()
                ),
                "low_metric_mean": float(values[current_low].mean()),
                "high_metric_mean": float(values[current_high].mean()),
                "high_minus_low": float(values[current_high].mean() - values[current_low].mean()),
            })
    return controlled_rows, correlation_rows, valid_controlled_bins


def _load_model(args, model_params: ModelParams):
    dataset = model_params.extract(args)
    is_synthetic = os.path.exists(os.path.join(dataset.source_path, "transforms_train.json"))
    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        n_features_per_level=args.n_features,
        log2_hashmap_size=args.log2,
        log2_hashmap_size_2D=args.log2_2D,
        decoded_version=True,
        is_synthetic_nerf=is_synthetic,
    )
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.eval()
    return dataset, scene, gaussians


def _format_float(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.6f}"


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    model_params = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--n_features", type=int, default=None)
    parser.add_argument("--log2", type=int, default=None)
    parser.add_argument("--log2_2D", type=int, default=None)
    parser.add_argument("--k_values", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--default_k", type=int, default=8)
    parser.add_argument("--controlled_k", type=int, default=64)
    parser.add_argument("--controlled_anchor_samples", type=int, default=20_000)
    parser.add_argument("--distance_bins", type=int, default=10)
    parser.add_argument(
        "--distance_match_caliper",
        type=float,
        default=None,
        help="Maximum high/low spatial-distance mismatch; defaults to one codec voxel.",
    )
    parser.add_argument("--observation_chunk_size", type=int, default=262_144)
    parser.add_argument("--score_block_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=str, default="analysis_output")
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
        codec_aligned = canonicalize_codec_anchors(
            gaussians.get_anchor,
            mask,
            gaussians.voxel_size,
            attributes={
                "feature": gaussians._anchor_feat,
                "scaling": gaussians.get_scaling,
                "offset": gaussians._offset * gaussians.get_mask,
            },
        )
    attributes = {
        name: value.detach().cpu().numpy().astype(np.float32, copy=False)
        for name, value in codec_aligned.attributes.items()
    }
    xyz = codec_aligned.xyz.detach().cpu().numpy().astype(np.float64, copy=False)
    num_anchors = xyz.shape[0]
    if max(args.k_values + [args.controlled_k]) >= num_anchors:
        raise ValueError("requested K must be smaller than the valid codec anchor count")

    relation = build_geometry_observations(
        codec_aligned.xyz,
        train_cameras,
        chunk_size=args.observation_chunk_size,
    )
    max_k = max(args.k_values)
    coview_neighbors, coview_scores, diagnostics = coview_topk(
        relation, k=max_k, score_block_size=args.score_block_size
    )
    spatial_neighbors, spatial_distances = spatial_topk(xyz, k=max_k)

    k_results = {}
    for k in sorted(set(args.k_values)):
        overlap = _neighbor_overlap(coview_neighbors[:, :k], spatial_neighbors[:, :k])
        k_results[str(k)] = {
            "mean_neighbor_overlap": float(overlap.mean()),
            "median_neighbor_overlap": float(np.median(overlap)),
            "mean_coview_score": float(coview_scores[:, :k].mean()),
        }

    k = args.default_k
    if k not in args.k_values:
        raise ValueError("default_k must be included in k_values")
    source = np.repeat(np.arange(num_anchors, dtype=np.int64), k)
    coview_target = coview_neighbors[:, :k].reshape(-1)
    spatial_target = spatial_neighbors[:, :k].reshape(-1)
    random_target = _random_neighbors(num_anchors, k, args.seed).reshape(-1)

    coview_pair_metrics = _pair_metrics(attributes, source, coview_target)
    topk_correlations = {
        metric: _correlations(coview_scores[:, :k].reshape(-1), values)
        for metric, values in coview_pair_metrics.items()
    }

    comparison_rows = []
    comparison_summary = {}
    for kind, target in (
        ("coview", coview_target),
        ("spatial", spatial_target),
        ("random", random_target),
    ):
        metrics = coview_pair_metrics if kind == "coview" else _pair_metrics(attributes, source, target)
        means = _metric_means(metrics)
        comparison_summary[kind] = means
        for metric_name, value in means.items():
            comparison_rows.append({"k": k, "neighbor_type": kind, "metric": metric_name, "mean": value})

    # Controlled pool: sample anchors deterministically, then use their 64-NN pairs.
    rng = np.random.default_rng(args.seed)
    controlled_anchor_count = min(args.controlled_anchor_samples, num_anchors)
    controlled_anchor = np.sort(rng.choice(num_anchors, controlled_anchor_count, replace=False))
    controlled_neighbors_all, controlled_distances_all = spatial_topk(xyz, k=args.controlled_k)
    controlled_source = np.repeat(controlled_anchor, args.controlled_k)
    controlled_target = controlled_neighbors_all[controlled_anchor].reshape(-1)
    controlled_distance = controlled_distances_all[controlled_anchor].reshape(-1)
    controlled_score = pair_coview_scores(relation, controlled_source, controlled_target)
    controlled_metrics = _pair_metrics(attributes, controlled_source, controlled_target)
    distance_match_caliper = (
        float(gaussians.voxel_size)
        if getattr(args, "distance_match_caliper", None) is None
        else float(args.distance_match_caliper)
    )
    controlled_rows, binned_rows, valid_controlled_bins = _controlled_rows(
        controlled_source,
        controlled_distance,
        controlled_score,
        controlled_metrics,
        args.distance_bins,
        distance_match_caliper,
    )

    controlled_trend = {}
    for metric_name in METRIC_NAMES:
        rows = [row for row in controlled_rows if row["metric"] == metric_name]
        expected_positive = metric_name == "feature_cosine"
        supporting = sum(
            (row["high_minus_low"] > 0) if expected_positive else (row["high_minus_low"] < 0)
            for row in rows
        )
        controlled_trend[metric_name] = {
            "supporting_bins": int(supporting),
            "valid_bins": len(rows),
            "support_fraction": float(supporting / len(rows)) if rows else float("nan"),
            "mean_high_minus_low": float(np.mean([row["high_minus_low"] for row in rows])) if rows else float("nan"),
        }
    matched_pair_count = sum(
        int(row["high_count"])
        for row in controlled_rows
        if row["metric"] == "feature_cosine"
    )

    # Bounded deterministic raw sample; full aggregate arrays remain reproducible.
    sample_count = min(20_000, controlled_source.size)
    sample_idx = np.sort(rng.choice(controlled_source.size, sample_count, replace=False))
    pair_sample_rows = []
    for idx in sample_idx:
        pair_sample_rows.append({
            "source": int(controlled_source[idx]),
            "target": int(controlled_target[idx]),
            "spatial_distance": float(controlled_distance[idx]),
            "coview_score": float(controlled_score[idx]),
            **{metric: float(values[idx]) for metric, values in controlled_metrics.items()},
        })

    observed_counts = np.diff(relation.anchor_indptr)
    mean_overlap = k_results[str(k)]["mean_neighbor_overlap"]
    feature_control = controlled_trend["feature_cosine"]
    feature_l2_control = controlled_trend["feature_l2"]
    controlled_support = (
        (feature_control["support_fraction"] >= 0.6 if np.isfinite(feature_control["support_fraction"]) else False)
        or (feature_l2_control["support_fraction"] >= 0.6 if np.isfinite(feature_l2_control["support_fraction"]) else False)
    )
    recommend_phase2 = bool(mean_overlap < 0.8 and controlled_support)

    metrics_json = {
        "scene": Path(dataset.source_path).name,
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": int(scene.loaded_iter),
        "train_camera_count": len(train_cameras),
        "anchor_count_before_mask": int(gaussians.get_anchor.shape[0]),
        "valid_codec_anchor_count": int(num_anchors),
        "voxel_size": float(gaussians.voxel_size),
        "observation": {
            **diagnostics,
            "unobserved_anchor_count": int((observed_counts == 0).sum()),
            "mean_cameras_per_anchor": float(observed_counts.mean()),
            "max_cameras_per_anchor": int(observed_counts.max(initial=0)),
        },
        "k_results": k_results,
        "default_k": k,
        "coview_attribute_correlations": topk_correlations,
        "neighbor_comparison": comparison_summary,
        "controlled_spatial_experiment": {
            "sampled_anchor_count": int(controlled_anchor_count),
            "pair_count": int(controlled_source.size),
            "matched_pair_count": int(matched_pair_count),
            "distance_match_caliper": distance_match_caliper,
            "valid_distance_bins": int(valid_controlled_bins),
            "trend": controlled_trend,
        },
        "phase2_recommendation": {
            "recommend": recommend_phase2,
            "rule": "mean overlap < 0.8 and feature trend supported in >=60% of valid controlled bins",
            "note": "Descriptive Phase 1 gate only; no p-value or fixed correlation threshold is imposed.",
        },
        "runtime_seconds": float(time.time() - started),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_json, handle, indent=2, allow_nan=True)

    _write_csv(
        output_dir / "neighbor_comparison.csv",
        comparison_rows,
        ["k", "neighbor_type", "metric", "mean"],
    )
    _write_csv(
        output_dir / "distance_binned_correlations.csv",
        binned_rows,
        ["distance_bin", "distance_min", "distance_max", "pair_count", "metric", "count", "pearson", "spearman"],
    )
    _write_csv(
        output_dir / "controlled_high_low.csv",
        controlled_rows,
        [
            "distance_bin", "distance_min", "distance_max", "pair_count", "metric",
            "low_count", "low_unique_count", "high_count", "source_anchor_count",
            "low_coview_mean", "high_coview_mean", "low_distance_mean",
            "high_distance_mean", "mean_absolute_distance_mismatch", "low_metric_mean",
            "high_metric_mean", "high_minus_low",
        ],
    )
    _write_csv(
        output_dir / "pair_samples.csv",
        pair_sample_rows,
        ["source", "target", "spatial_distance", "coview_score", *METRIC_NAMES],
    )

    top_corr_lines = "\n".join(
        f"- {name}: Spearman={_format_float(value['spearman'])}, "
        f"Pearson={_format_float(value['pearson'])} (n={value['count']:,})"
        for name, value in topk_correlations.items()
    )
    comparison_lines = "\n".join(
        f"- {kind}: feature cosine={values['feature_cosine']:.6f}, "
        f"feature L2={values['feature_l2']:.6f}, scaling L2={values['scaling_l2']:.6f}, "
        f"offset L2={values['offset_l2']:.6f}"
        for kind, values in comparison_summary.items()
    )
    controlled_lines = "\n".join(
        f"- {name}: expected direction in {value['supporting_bins']}/{value['valid_bins']} bins; "
        f"mean(high-low)={_format_float(value['mean_high_minus_low'])}"
        for name, value in controlled_trend.items()
    )
    k_lines = "\n".join(
        f"- K={key}: mean overlap={value['mean_neighbor_overlap']:.6f}, "
        f"median overlap={value['median_neighbor_overlap']:.6f}, "
        f"mean CoView score={value['mean_coview_score']:.6f}"
        for key, value in k_results.items()
    )
    recommendation = "建议进入 Phase 2" if recommend_phase2 else "暂不建议直接进入 Phase 2"
    summary = f"""# Phase 1 CoView Analysis Summary

## Scope and data

- Scene: `{Path(dataset.source_path).name}`
- Existing model: `{Path(dataset.model_path).resolve()}` (iteration {scene.loaded_iter})
- Cameras: {len(train_cameras)} train cameras; test cameras were not used
- Anchors: {gaussians.get_anchor.shape[0]:,} before valid mask; {num_anchors:,} codec-aligned valid anchors
- Geometry: `round(anchor / voxel_size) * voxel_size`, voxel size {gaussians.voxel_size}
- Canonical order: unchanged HAC++ `calculate_morton_order()` implementation
- Observation: geometry-only frustum projection; no feature/scaling/offset/opacity/rotation, rasterizer, or occlusion

## Spatial / CoView overlap

{k_lines}

The provisional caution range is 0.80–0.90 mean overlap. The default K={k} result is {mean_overlap:.6f}.

## CoView score and attributes (CoView Top-{k} pairs)

{top_corr_lines}

For cosine similarity a positive correlation is the expected direction; for L2 distances a negative correlation is expected.

## CoView / Spatial / Random neighbours (K={k})

{comparison_lines}

## Spatial-distance-controlled high/low CoView experiment

The controlled pool contains {controlled_source.size:,} pairs from {controlled_anchor_count:,} sampled anchors and their {args.controlled_k} nearest spatial neighbours. High/low comparison retained {matched_pair_count:,} same-source matched pairs whose distance mismatch is at most one codec voxel ({distance_match_caliper}). Retained pairs were split into {args.distance_bins} distance-quantile bins.

{controlled_lines}

High/low CoView sets are formed separately within each source anchor's 64-NN pool. Every high-CoView pair is matched to the nearest-distance low-CoView pair from the same source anchor and rejected if the mismatch exceeds the caliper. `controlled_high_low.csv` records the matched mean distances, mean absolute distance mismatch, source count, and unique matched-low count. Per-bin correlations over the full unfiltered pool are in `distance_binned_correlations.csv`.

## Phase 2 recommendation

**{recommendation}.** The descriptive gate used here requires mean Spatial/CoView overlap below 0.80 and the controlled feature cosine or feature-L2 direction to hold in at least 60% of valid distance bins. This is a reporting aid, not a p-value or a fixed scientific significance threshold; cross-scene consistency is still required.

## Reproducibility and outputs

- `metrics.json`: complete aggregate metrics and sparse-matrix diagnostics
- `neighbor_comparison.csv`: neighbour-family attribute means
- `distance_binned_correlations.csv`: correlations after spatial-distance binning
- `controlled_high_low.csv`: controlled high/low CoView comparisons
- `pair_samples.csv`: deterministic bounded raw pair sample
- `plots/`: generated by `analysis/plot_coview_analysis.py`

Runtime: {metrics_json['runtime_seconds']:.1f} seconds. The analysis did not train or modify the model or bitstreams.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
