# Phase 1.5 Geometric View Analysis: playroom

## Scope

- Existing HAC++ checkpoint: `/home/fansonglin/data_space/Chenzhenxin/HAC-plus/outputs/blending/playroom/0.004` at iteration 30000
- 108,549 codec-aligned anchors and 196 train cameras; test cameras were not used
- Geometry-only descriptors: distance, unit viewing direction, normalized image coordinate, camera-space depth
- Controlled pool: 1,280,000 spatial-neighbour pairs; same-source matching with distance caliper 0.005
- No training, renderer, entropy model, model structure, loss, or codec modification was performed by this analysis

## Dynamic range

- Binary Jaccard: mean 0.948630, std 0.067932, unique values 427
- Geometric composite: mean 0.979022, std 0.033284, unique values 574,902

## Controlled high/low results

| Score | Attribute statistic | Expected bins | Mean high − matched-low |
|---|---|---:|---:|
| binary_jaccard | feature_cosine | 0/10 | -0.005085 |
| binary_jaccard | feature_l2 | 1/10 | 0.062870 |
| binary_jaccard | scaling_l2 | 10/10 | -0.002090 |
| binary_jaccard | offset_l2 | 2/10 | 0.040817 |
| geometric_distance | feature_cosine | 4/10 | -0.003335 |
| geometric_distance | feature_l2 | 1/10 | 0.111719 |
| geometric_distance | scaling_l2 | 9/10 | -0.004756 |
| geometric_distance | offset_l2 | 0/10 | 0.114970 |
| geometric_direction | feature_cosine | 8/10 | 0.021488 |
| geometric_direction | feature_l2 | 8/10 | -0.201849 |
| geometric_direction | scaling_l2 | 0/10 | 0.011226 |
| geometric_direction | offset_l2 | 3/10 | 0.018922 |
| geometric_image | feature_cosine | 9/10 | 0.035342 |
| geometric_image | feature_l2 | 10/10 | -0.325463 |
| geometric_image | scaling_l2 | 1/10 | 0.009425 |
| geometric_image | offset_l2 | 6/10 | 0.011455 |
| geometric_depth | feature_cosine | 5/10 | 0.003069 |
| geometric_depth | feature_l2 | 3/10 | 0.060083 |
| geometric_depth | scaling_l2 | 8/10 | -0.004410 |
| geometric_depth | offset_l2 | 2/10 | 0.108104 |
| geometric_composite | feature_cosine | 10/10 | 0.055976 |
| geometric_composite | feature_l2 | 9/10 | -0.512570 |
| geometric_composite | scaling_l2 | 3/10 | 0.014420 |
| geometric_composite | offset_l2 | 8/10 | -0.477688 |

## Scaling decision

**Geometric composite does not improve the controlled Scaling result over binary Jaccard.** Binary Scaling: 10/10 bins, mean delta -0.002090. Geometric composite Scaling: 3/10 bins, mean delta 0.014420.

Feature remains an observation-only metric. Phase 1.5 does not authorize or implement an entropy-model change.

Runtime: 29.9 seconds.
