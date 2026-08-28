# Causal CoView Feature: Formal HAC++ Integration

## Status

The 15-parameter affine causal Feature prior is integrated into the HAC++
training-rate path and the real arithmetic codec. It is disabled by default, so
the original HAC++ model, RNG trajectory, rate estimator and two-component
Feature codec remain unchanged unless `--use_causal_coview_feature` is set.

The integrated codec has been verified on `playroom` and `drjohnson` in a fresh
Python process. Both decoders reproduced the encoder's exact quantized Feature
symbol indices.

## Coding contract

- Valid anchors are quantized and ordered with the existing
  `calculate_morton_order()` canonical sort. That function is intentionally not
  changed to a true Morton order.
- Canonical anchor `i` belongs to coding group `i % G`, with `G=4` by default.
- An anchor may only use CoView neighbors in strictly earlier groups.
- HAC++'s five sequential 10-channel `Channel_CTX_fea` chunks are preserved.
- HAC++'s two Gaussian Feature priors are preserved. CoView is a third Gaussian
  component, mixed at probability level.
- The prior has five mean-blend, five scale-correction and five gate parameters.
- Scaling and Offset keep their original coding order and stream format.
- The formal package records and verifies both causal-graph and Feature
  symbol-index checksums.

The spatial candidate pools are deliberately separate:

- `view_topology_candidates=16` for the existing Scaling-CoView branch;
- `causal_coview_candidates=32` for causal Feature.

This prevents the Feature experiment from silently changing Scaling's topology
or bitstream.

## Frozen formal-codec result

The comparison below adds causal Feature to the already trained FP16
Scaling-CoView package. Anchor, representation, renderer, Scaling predictor and
Offset predictor are fixed.

| Scene | Scaling-only Feature | Causal Feature | Feature delta | Model blob delta | Full directory delta |
|---|---:|---:|---:|---:|---:|
| playroom | 2,449,289 B | 2,446,742 B | **-2,547 B** | +155 B | **-1,304 B** |
| drjohnson | 2,779,958 B | 2,777,686 B | **-2,272 B** | +155 B | **-1,093 B** |

`Full directory delta` includes the larger standalone `entropy_context.pth`
(+1,088 B and +1,024 B respectively), not just entropy payload and prior
parameters. Scaling and Offset stream sizes are exactly unchanged:

| Scene | Scaling before/after | Offset before/after |
|---|---:|---:|
| playroom | 880,397 / 880,397 B | 1,526,267 / 1,526,267 B |
| drjohnson | 1,117,227 / 1,117,227 B | 1,739,477 / 1,739,477 B |

This is a conditional-coding result. Reconstruction quality is unchanged by
construction. It proves a positive net standalone-package reduction on both
tested frozen representations; it does not by itself prove an RD improvement.

## Training integration

After densification stops, training builds the graph over all valid anchors in
codec-canonical order and explicitly stores both original-to-canonical and
canonical-to-original mappings. A random rate sample uses decoded-equivalent
quantized Features from earlier groups, never only the visible render subset.

Earlier-group values are treated as stop-gradient decoder state. This avoids
unstable cross-anchor gradients through zero-variance neighbor statistics while
retaining gradients through the current Feature symbol, the HAC++ predictor and
the causal-prior parameters.

Branching from a causal-disabled shared 15k checkpoint and resuming a causal
checkpoint were both tested through a real optimizer step. All Feature and prior
parameters remained finite.

## Commands and artifacts

Server artifacts are under:

- `/mnt/003/experiments/causal_coview_formal/playroom/bitstreams_fp16`
- `/mnt/003/experiments/causal_coview_formal/drjohnson/bitstreams_fp16`
- `/mnt/003/experiments/causal_coview_joint_smoke/playroom_stopgrad`
- `/mnt/003/experiments/causal_coview_joint_smoke/playroom_stopgrad_resume`

The multi-scene joint-RD runs use GPUs 4--6 and are written under
`/mnt/003/experiments/causal_coview_joint/`.

## Joint-training result

Three branches resumed matched 15k checkpoints and jointly optimized through
30k. The table compares the resulting Scaling+Causal-Feature package with the
corresponding Scaling-only branch. Both sides were re-encoded after packaging
the renderer networks, so `package delta` is an exact standalone-directory
comparison rather than a resident-model estimate.

| Scene | Lambda | Package delta | PSNR delta | SSIM delta | LPIPS delta |
|---|---:|---:|---:|---:|---:|
| playroom | 0.0005 | **-626 B** | +0.19116 | +0.000079 | +0.010764 |
| drjohnson | 0.001 | **-25,671 B** | +0.02375 | +0.000916 | +0.007630 |
| drjohnson | 0.0005 | +505 B | +0.09400 | +0.000915 | +0.007054 |

The causal expert saturates near its configured 0.25 maximum mixture weight in
all five Feature chunks, so the small/variable rate result is not caused by the
optimizer simply disabling the module. It is representation- and lambda-
dependent: the `drjohnson` 0.001 branch saves about 25.7 kB, while the lower-rate
branch is essentially neutral. PSNR and SSIM improve at all three tested points,
but LPIPS regresses at all three. This is therefore not evidence of a universal
RD gain.

## Standalone rendering contract fixes

The original entropy package could recover coded attributes but still depended
on resident renderer state. Formal evaluation exposed and fixed two omissions:

- fixed identity anchor rotations and constant opacities are reconstructed at
  the decoded anchor count;
- `mlp_opacity`, `mlp_cov`, `mlp_color`, and the optional feature-bank network
  are now packaged and loaded by the decoder.

These base networks were already included in HAC++'s reported `base_MLPs` size;
the change makes the on-disk package match that accounting. Fresh-process
decode, render, and metric evaluation now succeeds without a training model.

## Current conclusion

The decoder-safe causal CoView mechanism is now a real HAC++ entropy module, not
only an offline ablation. Frozen conditional coding remains positive on both
tested scenes, but joint-training gains are too small and inconsistent to claim
general RD improvement. Candidate expansion has already saturated; the next
model iteration should change the information path by adding a decoder-causal
Scaling expert, then condition normalized Offset coding on decoded Scaling,
before spending compute on a broad scene sweep.
