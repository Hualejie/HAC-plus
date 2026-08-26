# Phase 2B: Generalized CoView Context Results

## Decision

The shared-checkpoint experiment is complete for Deep Blending `playroom`.

The result is **Go only for the Scaling-specific branch, and No-Go for the
generalized Feature/Offset/All claim**:

- Scaling-only saves 2,945 total bytes (0.115%) against the shared Control while
  improving PSNR by 0.0653 dB, SSIM by 0.000788, and LPIPS by 0.000969;
- Feature-only, Offset-only, and All are respectively 12,497, 11,262, and 12,190
  bytes larger than Control after counting the active CoView parameters;
- All saves 12,847 raw Feature/Scaling/Offset bytes, but its 24,764-byte active
  CoView network is larger than that saving and its fidelity is slightly worse;
- same-checkpoint ON/OFF diagnostics show that every active branch is genuinely
  used by its trained model. This is conditional entropy value, not evidence of
  an end-to-end improvement over Control.

Do not enter Attribute-Similarity or decoded-neighbour context on the strength
of this one-scene result. The next controlled step should replicate Scaling-only
on another scene and/or seed, then reduce or regularize the shared/head cost.

## Implemented method

Phase 2A's deterministic distance/depth View-Geometry topology is unchanged. A
single 15-D topology descriptor is processed by a shared trunk, followed by
independent entropy heads:

```text
15-D View-Geometry topology
          |
          v
shared trunk: 15 -> 32 -> ReLU
          |
          +-- Feature head: 32 -> 100
          +-- Scaling head: 32 -> 12
          +-- Offset head:  32 -> 60
```

Each head predicts a mean residual and log-scale residual. Feature is adjusted
before the existing `Channel_CTX_fea`; the 5x10 chunk order, mixture
probabilities, masks, coding order, and all quantization steps remain unchanged.

`coview_target=none|feature|scaling|offset|all` selects active heads. All heads
exist in every training model and optimizer so the shared checkpoint has an
identical parameter-group structure. Final layers are zero-initialized and the
three gates start at 1.0. This combination is exactly baseline at M15k while
giving the active head a gradient on its first step; a zero gate plus a zero head
would leave both without a first-step gradient.

Training, estimation, real encoding, and real decoding all call the same
`apply_coview_entropy_parameters()` interface.

## Strict shared-checkpoint contract

The original HAC++ tuple checkpoint was not usable for a controlled fork: it
omitted `_anchor_feat`, hash-grid and MLP values, CoView state, most training
buffers, RNG state, and the remaining camera stack, and its `capture()` and
`restore()` tuple layouts disagreed.

The versioned M15k checkpoint now contains:

- all Gaussian parameters and relevant densification/training buffers;
- hash-grid, base entropy/render MLPs, shared trunk, all three heads, and gates;
- optimizer state and parameter-group order;
- bounds and architecture metadata;
- Python, NumPy, Torch CPU, and Torch CUDA RNG states;
- the exact remaining training-camera stack using stable camera keys.

The checkpoint is saved after iteration 15,000's optimizer step. Its SHA-256 is:

```text
e95f4fc368187a94840c996af7fb7e4419f172f22f7cf0637b36c911ed0d1924
```

It contains 176,120 anchors, 92 cameras remaining in the current sampling
stack, one CUDA RNG state, all fourteen optimizer groups, zero-valued final
layers for all three heads, and unit gates. A real `all` resume to iteration
15,001 rebuilt the topology, completed an optimizer step, and wrote a new
versioned checkpoint successfully.

## End-to-end results

All five runs start from the exact M15k file. The table uses exact stream and
float32 parameter bytes; Total also includes the 24 bytes used by the two xyz
bounds. HAC++'s log labels use powers of 1024 despite saying MB.

| Target | Valid anchors | Feature B | Scaling B | Offset B | Active CoView B | Total B | Net saved vs Control | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none / Control | 106,382 | 856,108 | 519,906 | 524,099 | 0 | 2,556,384 | 0 | 30.645916 | 0.908795 | 0.268275 |
| feature | 106,450 | 854,124 | 516,928 | 526,251 | 15,252 | 2,568,881 | -12,497 | 30.651796 | 0.908000 | 0.269284 |
| scaling | 106,574 | 853,926 | 514,134 | 525,083 | 3,636 | 2,553,439 | **+2,945** | **30.711266** | **0.909583** | **0.267306** |
| offset | 106,527 | 858,686 | 516,656 | 525,750 | 9,972 | 2,567,646 | -11,262 | 30.647057 | 0.908581 | 0.268556 |
| all | 106,542 | 853,064 | 511,524 | 522,678 | 24,764 | 2,568,574 | -12,190 | 30.627415 | 0.907170 | 0.270717 |

Relative to Control, raw attribute-stream savings are:

| Target | Feature saved B | Scaling saved B | Offset saved B | Raw F+L+O saved B | Net total saved B |
|---|---:|---:|---:|---:|---:|
| feature | 1,984 | 2,978 | -2,152 | 2,810 | -12,497 |
| scaling | 2,182 | 5,772 | -984 | 6,970 | **2,945** |
| offset | -2,578 | 3,250 | -1,651 | -979 | -11,262 |
| all | 3,044 | 8,382 | 1,421 | 12,847 | -12,190 |

Anchor, hash, and mask bytes also vary slightly because the active entropy loss
changes the second-half optimization trajectory. Net Total, rather than an
isolated target stream, is therefore the decision metric.

## Same-checkpoint ON/OFF diagnostic

Each trained floating model was preserved before `conduct_decoding()` replaced
its attributes. Re-encoding that exact model with CoView disabled gives:

| Trained target | Feature OFF-ON B | Scaling OFF-ON B | Offset OFF-ON B |
|---|---:|---:|---:|
| feature | 304,397 | 0 | 0 |
| scaling | 0 | 121,609 | 0 |
| offset | 0 | 0 | 5,478 |
| all | 212,345 | 123,059 | 11,097 |

This confirms target isolation and proves that the trained Feature and Scaling
models strongly rely on their CoView distributions. It does not make Feature or
All better than the shared Control: their learned non-CoView fallback is poor,
while their actual ON stream plus parameter cost is still larger than Control.

## Gates and residuals

Single-target final gates are 1.18237 (Feature), 1.50058 (Scaling), and 1.22152
(Offset). In All they are 0.99818, 1.75143, and 0.97389 respectively. The All
residual mean-absolute values are 0.33362, 0.80726, and 0.31984.

The largest gate is Scaling, consistent with Scaling being the only end-to-end
winner. Gate magnitude is not a sufficient bitrate-gain estimator, however:
Feature remains near a unit gate and has a 212,345-byte same-checkpoint ON/OFF
effect in All, but does not improve Total against Control.

## Runtime and codec validation

The shared 0-to-15k stage took 372.17 s. The matched 15k-to-30k branches took
509.66-521.11 s, so full training time was about 14.7-14.9 minutes per branch
when the shared first half is included. Encoding took 2.05-2.07 s.

Every branch executed real estimate, arithmetic encoding, decoding, decoded-model
save/reload, rendering, and evaluation. A separate fresh process, with no Scene
or training checkpoint, decoded all five entropy packages successfully:

| Target | Fresh decoded anchors | Fresh decode s |
|---|---:|---:|
| none | 106,382 | 4.0428 |
| feature | 106,450 | 8.4479 |
| scaling | 106,574 | 8.4596 |
| offset | 106,527 | 8.4578 |
| all | 106,542 | 8.0732 |

The Control was re-encoded after fixing the baseline standalone entropy contract;
all 257 pre-existing stream files were byte-identical, with only the previously
missing `entropy_context.pth` added. The package now supplies `mlp_grid` and
`mlp_deform` for every target, plus camera/topology/head state and checksum when
CoView is active.

This validates standalone entropy decoding. The complete renderer package still
uses the saved rendering MLPs, opacity, and rotation from the model checkpoint;
it is not claimed to be a bitstream-only renderer deployment.

## Answers to the Phase 2B questions

1. **Feature:** contains conditional entropy information, but no end-to-end gain.
2. **Scaling:** effective; it is the only branch with a positive net Total result.
3. **Offset:** weakly used and not effective end to end.
4. **All:** not better than Scaling-only or Control.
5. **Total after new MLP:** decreases only for Scaling-only, by 2,945 bytes.
6. **Fidelity:** maintained for the single branches; Scaling improves all three
   reported fidelity metrics. All is slightly worse.
7. **Main source:** Scaling.
8. **Gate consistency:** qualitatively identifies Scaling, but gate magnitude
   does not predict net gain for Feature/Offset.
9. **Shared-control validity:** yes; all branches restore the same complete M15k,
   optimizer, RNG, and remaining camera stack.
10. **Next stage:** do not add Attribute-Similarity yet. First replicate the small
    Scaling-only net gain on a second scene/seed and test a cheaper residual head.

Raw data are in `analysis_output_phase2b/summary.json`,
`analysis_output_phase2b/attribute_rate_ablation.csv`, and
`analysis_output_phase2b/gate_statistics.csv`.
