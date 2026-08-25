# Phase 1.5 Geometric View Analysis: room

## Scope

- Existing HAC++ checkpoint: `/home/fansonglin/data_space/Chenzhenxin/HAC-plus/outputs/mipnerf360/room/0.004` at iteration 30000
- 157,573 codec-aligned anchors and 272 train cameras; test cameras were not used
- Geometry-only descriptors: distance, unit viewing direction, normalized image coordinate, camera-space depth
- Controlled pool: 1,280,000 spatial-neighbour pairs; same-source matching with distance caliper 0.001
- No training, renderer, entropy model, model structure, loss, or codec modification was performed by this analysis

## Dynamic range

- Binary Jaccard: mean 0.958295, std 0.065676, unique values 2,725
- Geometric composite: mean 0.987924, std 0.027022, unique values 415,786

## Controlled high/low results

| Score | Attribute statistic | Expected bins | Mean high − matched-low |
|---|---|---:|---:|
| binary_jaccard | feature_cosine | 4/10 | 0.000100 |
| binary_jaccard | feature_l2 | 4/10 | 0.000177 |
| binary_jaccard | scaling_l2 | 6/10 | 0.000539 |
| binary_jaccard | offset_l2 | 5/10 | 0.004114 |
| geometric_distance | feature_cosine | 9/10 | 0.011162 |
| geometric_distance | feature_l2 | 7/10 | -0.084893 |
| geometric_distance | scaling_l2 | 9/10 | -0.001736 |
| geometric_distance | offset_l2 | 7/10 | -0.021945 |
| geometric_direction | feature_cosine | 9/10 | 0.015386 |
| geometric_direction | feature_l2 | 6/10 | -0.105149 |
| geometric_direction | scaling_l2 | 1/10 | 0.005444 |
| geometric_direction | offset_l2 | 5/10 | 0.028535 |
| geometric_image | feature_cosine | 6/10 | 0.007123 |
| geometric_image | feature_l2 | 8/10 | -0.106794 |
| geometric_image | scaling_l2 | 2/10 | 0.005132 |
| geometric_image | offset_l2 | 6/10 | 0.006259 |
| geometric_depth | feature_cosine | 9/10 | 0.006050 |
| geometric_depth | feature_l2 | 6/10 | -0.041095 |
| geometric_depth | scaling_l2 | 8/10 | -0.001096 |
| geometric_depth | offset_l2 | 7/10 | -0.030655 |
| geometric_composite | feature_cosine | 6/10 | 0.017042 |
| geometric_composite | feature_l2 | 6/10 | -0.093129 |
| geometric_composite | scaling_l2 | 2/10 | 0.012224 |
| geometric_composite | offset_l2 | 4/10 | -0.009163 |

## Scaling decision

**Geometric composite does not improve the controlled Scaling result over binary Jaccard.** Binary Scaling: 6/10 bins, mean delta 0.000539. Geometric composite Scaling: 2/10 bins, mean delta 0.012224.

Feature remains an observation-only metric. Phase 1.5 does not authorize or implement an entropy-model change.

Runtime: 82.4 seconds.
