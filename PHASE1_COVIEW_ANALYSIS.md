# Phase 1: CoView Independent-Information Analysis

## Status

Phase 1 is complete for the existing Deep Blending `playroom` 30k baseline. No training, model-structure, renderer, loss, rate-estimation, or codec code was changed.

The result does **not** support entering Phase 2 with the proposed Jaccard CoView context as a feature entropy context. CoView and spatial neighbour sets are distinct, but after strict spatial-distance matching the expected feature trend is absent: high-CoView pairs have lower feature cosine similarity in all 10 bins and higher feature L2 distance in 9 of 10 bins.

## Implementation scope

The Phase 1 implementation adds:

- `scene/coview_context.py`: codec anchor alignment, geometry-only observation, sparse Camera↔Anchor relation, Jaccard CoView Top-K, and deterministic spatial Top-K;
- `analysis/analyze_coview.py`: existing-checkpoint loading, correlations, neighbour-family comparison, distance-binned analysis, and matched high/low controlled experiment;
- `analysis/plot_coview_analysis.py`: plots generated only from aggregate CSV files;
- `tests/test_coview_context.py`: deterministic and memory-contract tests;
- `analysis_output/`: the reproducible `playroom` result bundle.

The implementation does not import or call the rasterizer. Observation uses only codec-equivalent xyz and train-camera `world_view_transform`, `full_proj_transform`, image dimensions, and near/far planes. Occlusion is deliberately not modelled.

## Codec alignment contract

The analysis mirrors `conduct_encoding()` without altering it:

```python
valid_original_idx = nonzero(get_mask_anchor)
anchor_int = round(anchor[valid] / voxel_size)
codec_order = calculate_morton_order(anchor_int)
codec_xyz = anchor_int[codec_order] * voxel_size
codec_original_idx = valid_original_idx[codec_order]
```

Feature, scaling, and masked offset tensors are reordered by the same `codec_order`. The current mixed-radix `calculate_morton_order()` remains unchanged; it is not replaced with Morton/Z-order.

This baseline's iteration-30000 PLY was saved after the in-process codec round-trip, so it already contains 108,549 decoded valid anchors rather than the 175,883 pre-codec training anchors. Consequently `anchor_count_before_mask == valid_codec_anchor_count` in this run. The explicit `valid_original_idx` and `codec_original_idx` mapping is still retained and tested, and the analysis position/order remains codec-equivalent.

## Sparse observation and Top-K construction

- Camera input is sorted by `(image_name, colmap_id, uid)`, making camera input permutation irrelevant.
- Each camera projects anchor chunks independently; only observed anchor indices are retained on CPU.
- The transposed Anchor→Camera relation is stored as CSR (`anchor_indptr`, `anchor_camera_ids`).
- Identical camera sets are grouped into observation signatures. Jaccard scores are evaluated in `[signature_block, num_signatures]` blocks.
- Top-K tie-break is score descending, then codec anchor index ascending.
- Spatial Top-K uses `scipy.spatial.cKDTree` and explicitly resolves equal-distance boundary ties by codec anchor index.
- No persistent dense `[camera, anchor]` GPU tensor or `[anchor, anchor]` CPU/GPU matrix is constructed.

For `playroom`, the sparse relation contains 2,874,437 edges, 11,456 unique signatures, and one unobserved anchor. Anchors are observed by 26.48 train cameras on average. The largest score block was `[256, 11456]`, versus a forbidden anchor-pair matrix of `[108549, 108549]`.

## Run command

The analysis ran in the existing server environment without retraining:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/fansonglin/data_space/Chenzhenxin/HAC-plus
/home/fansonglin/miniconda3/bin/conda run -n HAC_5090_a100 \
  python analysis/analyze_coview.py \
  -m /home/fansonglin/data_space/Chenzhenxin/HAC-plus/outputs/blending/playroom/0.004 \
  --output /home/fansonglin/data_space/Chenzhenxin/HAC-plus/analysis_output

/home/fansonglin/miniconda3/bin/conda run -n HAC_5090_a100 \
  python analysis/plot_coview_analysis.py --output analysis_output
```

End-to-end analysis runtime, including model and camera loading, was 18.1 seconds. Exactly 196 train cameras were used; the 29 test cameras were loaded by the unchanged `Scene` class but never passed to the observation builder.

## Results

### Spatial/CoView Top-K overlap

| K | Mean overlap | Median overlap | Mean selected CoView score |
|---:|---:|---:|---:|
| 4 | 0.152652 | 0.000000 | 0.995889 |
| 8 | 0.230000 | 0.125000 | 0.993394 |
| 16 | 0.323777 | 0.312500 | 0.988972 |
| 32 | 0.421040 | 0.437500 | 0.981575 |

The K=8 overlap is far below the 0.80–0.90 caution range. CoView is therefore not merely reproducing Euclidean Top-K. However, this distinctness alone is not evidence that it is useful to the entropy model.

### CoView score versus attributes on CoView Top-8 pairs

| Attribute statistic | Pearson | Spearman | Expected direction |
|---|---:|---:|---:|
| Feature cosine | -0.012187 | -0.016579 | positive |
| Feature L2 | 0.063832 | 0.078663 | negative |
| Scaling L2 | -0.310232 | -0.194844 | negative |
| Offset L2 | -0.089942 | -0.051798 | negative |

The feature directions are opposite to the proposed context hypothesis. Scaling and offset show the expected direction in this uncontrolled Top-K view, so they require the controlled experiment below before interpretation.

### CoView, spatial, and random neighbour comparison (K=8)

| Neighbour type | Feature cosine ↑ | Feature L2 ↓ | Scaling L2 ↓ | Offset L2 ↓ |
|---|---:|---:|---:|---:|
| CoView | 0.341293 | 10.415773 | 0.166712 | 10.305183 |
| Spatial | 0.431676 | 9.705963 | 0.140724 | 10.181911 |
| Random | 0.081768 | 12.713907 | 0.206078 | 10.580433 |

CoView neighbours are more attribute-similar than random neighbours but consistently less similar than Euclidean neighbours. This indicates that much of the apparent CoView signal can be explained by spatial proximity.

### Spatial-distance-controlled high/low CoView experiment

The controlled pool contains 1,280,000 pairs: 20,000 deterministically sampled anchors and their 64 Euclidean nearest neighbours. High/low CoView sets are formed separately within every source anchor, each high pair is matched to the nearest-distance low pair from that same source, and matches farther apart than one codec voxel (0.005) are rejected. This retains 267,208 matched pairs. They are then divided into 10 spatial-distance quantile bins. Mean absolute distance mismatch is 0.0015–0.0019 across bins; high/low mean-distance residuals are below 0.0004.

| Attribute statistic | Bins in expected direction | Mean high − matched-low |
|---|---:|---:|
| Feature cosine | 0 / 10 | -0.005085 |
| Feature L2 | 1 / 10 | +0.062870 |
| Scaling L2 | 10 / 10 | -0.002090 |
| Offset L2 | 2 / 10 | +0.040817 |

After controlling both source identity and spatial distance, feature results consistently contradict the proposed Phase 2 feature-context hypothesis. Scaling retains a small, consistent signal. Offset is inconsistent and trends in the wrong mean direction. Results from looser matching changed materially because high-CoView neighbours were systematically closer; those diagnostics were rejected rather than reported as independent CoView evidence.

## Decision

**Do not enter Phase 2 yet with the current Jaccard Camera–Anchor CoView definition.**

The Phase 1 descriptive gate required both:

1. mean Spatial/CoView overlap below 0.80; and
2. expected feature cosine or feature-L2 direction in at least 60% of valid distance bins.

Condition 1 passes (0.23 at K=8), but condition 2 fails decisively (0% and 10%). No p-value or hard correlation threshold was used.

A second Mip-NeRF360 scene would still be useful as a falsification/robustness check if the research question warrants it, but the agreed first-scene result does not justify implementing `mlp_coview` or changing `mlp_grid`. A narrower follow-up hypothesis for scaling-only context would be a new research decision, not part of this Phase 1 implementation.

## Verification

Server test command:

```bash
/home/fansonglin/miniconda3/bin/conda run -n HAC_5090_a100 \
  python -m pytest -q tests/test_coview_context.py
```

Result: `5 passed`.

Tests cover:

- camera permutation invariance;
- valid/original/codec attribute index alignment;
- codec-equivalent quantized coordinate consistency;
- deterministic CoView and Euclidean Top-K tie-breaks;
- bounded signature score blocks and an explicit diagnostic that no dense `N×N` anchor matrix is created.

## Remaining risks and technical debt

- This is one Deep Blending scene, not cross-scene evidence.
- Frustum-only observation gives many almost-identical camera signatures: mean selected Top-8 Jaccard is 0.993394. Jaccard Top-K therefore has limited score dynamic range and many deterministic ties.
- The saved baseline PLY is post-codec, so the original 175,883-to-108,549 pruning map is not recoverable from this artifact alone. The Phase 1 analysis correctly targets the final codec anchors, but auditing a pre-codec mapping would require preserving the pre-round-trip state in a future baseline run.
- HAC++'s standalone-decoder problem recorded in `HACPP_CODE_AUDIT.md` remains unchanged. It must be addressed before any Phase 2 codec modification or paper claim about a standalone bitstream.

Phase 1 stops here. No Phase 2 code has been added.
