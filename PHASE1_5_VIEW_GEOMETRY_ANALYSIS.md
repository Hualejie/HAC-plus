# Phase 1.5: View-Geometry Context Analysis

## Status and decision

Phase 1.5 is complete on Deep Blending `playroom` and Mip-NeRF360 `room`.
The second scene used a newly trained, unchanged HAC++ baseline. No entropy
model, renderer, training loss, model structure, rate-estimation path, or codec
code was modified.

The cross-scene result supports a narrower hypothesis:

```text
Scaling <- camera-distance / camera-space-depth geometry
```

It does **not** support a generic equal-weight geometric context shared by all
attributes. Binary Jaccard does not reproduce across scenes, viewing direction
and image-plane proximity have the wrong Scaling trend, and the equal-weight
composite fails on both scenes. Feature and Offset remain observations only;
Phase 1.5 does not authorize a Phase 2 entropy-model implementation.

## Implementation scope

Phase 1.5 adds or extends only analysis code:

- `scene/coview_context.py`: sparse geometric observation descriptors and
  pairwise view-geometry scores;
- `analysis/analyze_geometric_view.py`: the same-source,
  spatial-distance-matched controlled experiment;
- `analysis/plot_geometric_view_analysis.py`: per-scene diagnostic plots;
- `analysis/summarize_geometric_view_scenes.py`: cross-scene aggregation;
- `tests/test_coview_context.py`: geometry invariance, determinism, alignment,
  and bounded-memory tests;
- `analysis_output_phase15/`: raw CSV/JSON results, summaries, and plots.

The experiment samples 20,000 codec-aligned anchors and 64 Euclidean
neighbours per anchor (1,280,000 candidate pairs), forms high/low groups per
source anchor, matches each high pair to a same-source low pair with the nearest
spatial distance, rejects matches beyond one codec voxel, and reports results
in 10 spatial-distance bins. Because the matching caliper differs with codec
voxel size, matched counts are reported for every score and scene.

## Decoder-reconstructable contract

The analysis uses final valid anchors at codec-equivalent positions and in the
current codec canonical order:

```python
valid_original_idx = nonzero(get_mask_anchor)
anchor_int = round(anchor[valid_original_idx] / voxel_size)
codec_order = calculate_morton_order(anchor_int)
codec_xyz = anchor_int[codec_order] * voxel_size
codec_original_idx = valid_original_idx[codec_order]
```

The existing mixed-radix `calculate_morton_order()` is retained unchanged; it
is not replaced with true Morton/Z-order. Attributes use the same index map.

For each observed camera-anchor edge, the descriptor contains only:

- Euclidean camera-anchor distance;
- world-space unit viewing direction;
- normalized projected image/NDC coordinate;
- camera-space depth.

Inputs are codec-equivalent anchor xyz plus train-camera geometry. Test cameras,
RGB, feature, scaling, offset, opacity, rotation, rasterizer visibility, and
occlusion are not used. Camera sorting makes the result invariant to input
camera order. Thus the descriptor is reconstructable on encoder and decoder
given identical decoded anchors and camera metadata. The existing HAC++
standalone-decoder limitation documented in `HACPP_CODE_AUDIT.md` remains a
separate prerequisite before any codec integration.

Scores are averaged over cameras shared by a pair. The analysis evaluates
binary Jaccard and four geometry components: relative distance difference,
view-direction cosine, an RBF kernel on normalized image coordinates, and
relative camera-space depth difference. `geometric_composite` is their
equal-weight mean; it is an analysis baseline, not a proposed final model.

## Mip-NeRF360 `room` baseline

The official HAC++ algorithm was trained for 30,000 iterations with:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  -s /home/fansonglin/xieliang/Chenzhenxin/dataset/360_v2/room \
  --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 \
  --iterations 30000 \
  -m /home/fansonglin/data_space/Chenzhenxin/HAC-plus/outputs/mipnerf360/room/0.004 \
  --lmbda 0.004 --mask_lr_final 0.0015
```

| Measurement | Result |
|---|---:|
| Training time | 1,613.14 s |
| Training-state anchors | 306,017 |
| Final codec-valid anchors | 157,573 |
| Encoded total | 3.4102 MB |
| Feature stream | 1.3395 MB |
| Scaling stream | 0.6814 MB |
| Offset stream | 0.5472 MB |
| Encoding time | 3.1012 s |
| Decoding time | 6.3674 s |
| PSNR | 31.7615814 dB |
| SSIM | 0.9201781 |
| LPIPS | 0.2156915 |

The training process exited normally, and the official encode -> decode ->
render/test flow completed before Phase 1.5 analysis.

## Cross-scene controlled Scaling result

Lower Scaling L2 is the expected direction, so a negative high-score minus
matched-low delta supports the context hypothesis.

| Score | `playroom` | `room` | Cross-scene result |
|---|---:|---:|---|
| Binary Jaccard | 10/10, -0.002090, n=267,208 | 6/10, +0.000539, n=104,920 | Not reproduced |
| Geometric distance | 9/10, -0.004756, n=49,039 | 9/10, -0.001736, n=37,165 | Consistent |
| Geometric direction | 0/10, +0.011226, n=6,598 | 1/10, +0.005444, n=6,106 | Rejected |
| Geometric image | 1/10, +0.009425, n=7,791 | 2/10, +0.005132, n=8,228 | Rejected |
| Geometric depth | 8/10, -0.004410, n=30,884 | 8/10, -0.001096, n=31,502 | Consistent |
| Equal-weight composite | 3/10, +0.014420, n=1,358 | 2/10, +0.012224, n=1,062 | Rejected |

The geometric components have more dynamic range than binary signatures, but
dynamic range alone does not imply usefulness. Distance and depth are the only
Scaling signals with the expected direction in both scenes. The composite is
especially weak and retains relatively few caliper-matched pairs, so it must
not be used to justify a generic `mlp_coview`.

## Feature and Offset observations

Feature is not a Phase 1.5 gate. Some component-level signals occur—for
example, direction Feature cosine is positive in 8/10 `playroom` bins and 9/10
`room` bins, while composite Feature cosine is 10/10 and 6/10—but they are not
uniform across scores or scenes and do not reverse the Phase 1 decision against
Binary-CoView Feature context.

Offset also lacks a stable cross-scene conclusion. Distance and depth are
positive on `room` but not on `playroom`; the composite changes from 8/10 on
`playroom` to 4/10 on `room`. Neither attribute should be added to a codec based
on this analysis.

## Recommendation

Stop Phase 1.5 here. Do not implement a generic geometric composite and do not
attach one context to Feature, Scaling, and Offset together.

If a later phase is explicitly approved, the smallest evidence-backed next
experiment is a **Scaling-only** context using camera-distance and/or
camera-space-depth components, with ablations for each component and their
combination. Before changing `conduct_encoding()` or `conduct_decoding()`, the
standalone-decoder contract must be fixed or explicitly defined so the same
decoded hash/context state is available on both sides.

Detailed scene-level results are in:

- `analysis_output_phase15/playroom/summary.md`;
- `analysis_output_phase15/room/summary.md`;
- `analysis_output_phase15/cross_scene/summary.md`.

Phase 1.5 ends without implementing Phase 2.
