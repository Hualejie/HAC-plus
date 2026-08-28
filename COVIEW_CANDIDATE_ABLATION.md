# CoView Candidate-Ceiling Ablation

## Scope

This experiment tests whether weak Feature gains are caused by the original
16-nearest-anchor Euclidean candidate ceiling. It keeps the represented
Feature symbols, quantization steps, HAC++ predictor, 15-D geometry descriptor,
CoView entropy head, training schedule, and arithmetic coder fixed. Only the
candidate topology changes.

The experiment uses the frozen `playroom` Phase 2D Scaling checkpoint at
lambda 0.0005. It contains 154,413 codec-valid anchors and 196 train-camera
geometry records. No renderer, reconstruction parameter, or training loss is
changed.

## Topology benchmark

| Candidate topology | Candidate pairs | Mean candidates | Build time (s) | Top-8 outside spatial-32 |
|---|---:|---:|---:|---:|
| Spatial 16 | 2,470,608 | 16.00 | 24.90 | 0% |
| Spatial 32 | 4,941,216 | 32.00 | 33.58 | 0% |
| Spatial 64 | 9,882,432 | 64.00 | 52.30 | 0% |
| Spatial 128 | 19,764,864 | 128.00 | 90.97 | 0% |
| Hybrid spatial-32 + view-32 | 7,793,122 | 50.47 | 55.87 | 0.6156% |

The hybrid path creates neither a dense anchor-pair matrix nor quadratic pair
enumeration. Its deterministic MinHash/LSH stage retained 23.13 view candidates
per anchor on average, but only 0.6156% of final Top-8 edges fell outside the
spatial-32 pool.

## Frozen Feature A1

All rows encode the same Feature symbol-index tensor. The Base stream is
2,451,588 bytes. The serialized FP16 CoView entropy module is 1,854 bytes.

| Candidate topology | CoView stream (B) | Conditional delta vs Base (B) | Stream + module (B) | Net delta vs Base (B) |
|---|---:|---:|---:|---:|
| Spatial 16 | 2,451,236 | -352 | 2,453,090 | +1,502 |
| Spatial 32 | 2,451,232 | -356 | 2,453,086 | +1,498 |
| Spatial 64 | 2,451,232 | -356 | 2,453,086 | +1,498 |
| Spatial 128 | 2,451,237 | -351 | 2,453,091 | +1,503 |
| Hybrid spatial-32 + view-32 | 2,451,272 | -316 | 2,453,126 | +1,538 |

Every package passed a fresh-process decode with exact Feature symbol-index
checksum equality. Direct float-tensor byte checksums differ after arithmetic
decode, while the quantized symbol indices are identical; codec correctness is
therefore evaluated at the coded-symbol level.

## Conclusion

Increasing Euclidean candidates beyond 32 is saturated: Spatial 32 and 64
produce the same stream size, and Spatial 128 is five bytes worse. The scalable
hybrid candidate generator successfully introduces non-local edges, but those
edges do not improve Feature conditional rate under the current geometry-only
15-D descriptor and residual entropy head. It saves 40 fewer bytes than
Spatial 32 and remains net-negative after module serialization.

The 16-neighbour ceiling is therefore not the main cause of weak Feature
results. Candidate expansion should not be promoted as the CoView innovation
and should remain an optional topology ablation. The next model change should
target the information carried by the context: view-selected, causally decoded
neighbour attributes with attribute-specific aggregation and multi-prior
fusion. Scaling is the safest first codec-integrated stream; Feature should
retain its 5x10 channel order, and Offset should be normalized by decoded
Scaling and modeled per offset index.

Server artifacts are under:

```text
/mnt/003/experiments/coview_candidate_ablation/playroom/
```
