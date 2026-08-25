# Phase 1.5 Cross-Scene Summary

Scenes: playroom, room.

| Score | Per-scene Scaling result | Mean support fraction | Mean high-low delta | Negative delta in every scene |
|---|---|---:|---:|---:|
| binary_jaccard | playroom: 10/10, delta=-0.002090, n=267,208; room: 6/10, delta=0.000539, n=104,920 | 0.800 | -0.000776 | no |
| geometric_distance | playroom: 9/10, delta=-0.004756, n=49,039; room: 9/10, delta=-0.001736, n=37,165 | 0.900 | -0.003246 | yes |
| geometric_direction | playroom: 0/10, delta=0.011226, n=6,598; room: 1/10, delta=0.005444, n=6,106 | 0.050 | 0.008335 | no |
| geometric_image | playroom: 1/10, delta=0.009425, n=7,791; room: 2/10, delta=0.005132, n=8,228 | 0.150 | 0.007278 | no |
| geometric_depth | playroom: 8/10, delta=-0.004410, n=30,884; room: 8/10, delta=-0.001096, n=31,502 | 0.800 | -0.002753 | yes |
| geometric_composite | playroom: 3/10, delta=0.014420, n=1,358; room: 2/10, delta=0.012224, n=1,062 | 0.250 | 0.013322 | no |

The table is descriptive. Scaling uses lower L2 as the expected direction. Feature remains observation-only, and this report does not authorize an entropy-model implementation.
