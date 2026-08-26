# Phase 2A: View-Geometry Topology Context Results

## Status and decision

Phase 2A is complete for the Deep Blending `playroom` scene at 30k iterations. The implementation, training rate path, final estimate, real encoder, real decoder, rendering test, and a fresh-process entropy decode were all exercised.

The result is **No-Go for Phase 2B at this point**. The topology context demonstrably predicts Scaling better on a fixed trained checkpoint, but the improvement did not survive an independent same-code training control. No Feature/Offset relational context or decoded-neighbour causal model should be added on the strength of this one scene.

## Implemented scope

The experiment is Scaling-only and topology-only:

- each anchor starts from 16 deterministic Euclidean candidates and selects Top-8 using geometry-only distance/depth support from train cameras;
- the fixed 15-D descriptor contains edge-score statistics, common-camera support, and weighted relative xyz/distance;
- no neighbour feature, scaling, offset, opacity, rotation, rasterizer visibility, or occlusion is read;
- `mlp_view_scaling` is `15 -> 32 -> 12` and adjusts only the six Scaling means and six log-scales;
- training, `estimate_final_bits()`, `conduct_encoding()`, and `conduct_decoding()` use the same context application;
- Feature and Offset coding are unchanged;
- no autoregressive dependency is introduced.

The new MLP has 908 parameters (3,632 float32 tensor bytes). Its output layer was zero-initialized, then trained to an output-weight L2 norm of 7.7075 with 312 non-zero weights. Initialization is wrapped in `torch.random.fork_rng(devices=[])` so it does not perturb the baseline CPU or CUDA RNG trajectory.

## Validation

### Unit and scale tests

- `python -m pytest -q tests/test_coview_context.py`: **11 passed**;
- full official-checkpoint topology smoke: 108,549 anchors, 196 train cameras, `[108549, 15]` features, 2,874,437 sparse observation edges, 3.86 s;
- no persistent dense anchor-pair matrix was created;
- CPU and CUDA post-construction RNG samples match the baseline model exactly after the RNG-isolation fix.

The delayed MLP learning-rate schedule initially exposed a divide-by-zero before iteration 1. The fix makes `max_steps` the absolute schedule endpoint, consistent with `get_expon_lr_func(step_sub=...)`; a regression test covers the 15k-to-30k schedule.

### Training topology

Topology was first refreshed after densification ended. The paired run built features for 135,436 current anchors and 196 train cameras, with 3,509,977 sparse observation edges, full valid-neighbour support, and no dense anchor-pair matrix.

Training completed in about 904 s versus 882 s for the same-code control. The approximately 22 s (2.5%) overhead includes the topology refresh and subsequent Scaling-context evaluation.

### Codec parity and fresh-process decode

The real encoder packaged the entropy MLP state, topology configuration, train-camera geometry, and encoder feature checksum in `bitstreams/entropy_context.pth`. A completely fresh Python process, with no Scene, checkpoint, or resident training model, decoded only from the packaged bitstreams and configuration.

Fresh decode passed with:

- 103,128 decoded anchors;
- 2,737,409 sparse observation edges;
- feature checksum `b679a838a665cc1fdb2c5c5dd6d282652e35836c3734701f31859eed51064e0e` on both encoder and decoder;
- 7.7467 s decode time;
- no dense anchor-pair matrix.

This validates the entropy-decoder contract. It does not claim that the complete renderer is a standalone deployment package.

## End-to-end results

HAC++ labels its binary-size values as MB, although the implementation divides by powers of 1024. The table preserves the logged values.

| Run | Valid codec anchors | Scaling | Feature | Offset | MLPs | Total | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing official baseline | 108,549 | 0.5038 | 0.8038 | 0.5091 | 0.3305 | 2.4405 | 30.8020 | 0.908829 | 0.274567 |
| Same-code control reproduction | 103,218 | 0.4737 | 0.7778 | 0.5049 | 0.3305 | 2.3803 | 30.7861 | 0.909029 | 0.274425 |
| View-topology | 103,128 | 0.4830 | 0.7804 | 0.5100 | 0.3340 | 2.3921 | 30.8808 | 0.910482 | 0.273509 |

Against the older official baseline, view topology appears favourable: Scaling drops by about 0.0208 MiB and total reported size drops by about 0.0484 MiB, with no quality loss. This comparison is not causal because the topology run already had lower Scaling rate before topology was enabled.

Against the contemporaneous same-code control, view topology is not smaller:

- Scaling is 9,693 bytes larger (0.4830 versus 0.4737 MiB);
- reported total size is about 0.0118 MiB larger;
- PSNR is 0.0947 dB higher, SSIM is 0.00145 higher, and LPIPS is 0.00092 lower;
- encode time is 1.8378 versus 1.7995 s;
- decode time is 7.5314 versus 3.7214 s, with nearly all extra time attributable to reconstructing topology.

The quality/size shift may be an ordinary rate-distortion trajectory difference. It does not meet the predeclared condition that Scaling savings must exceed the added MLP and that total size must decrease.

## Fixed-checkpoint real-codec ablation

Independent training remained measurably non-deterministic even after preserving the added module's RNG consumption. A stricter codec ablation therefore fixed all final anchors, attributes, masks, hash state, and entropy MLPs, then encoded twice from the same view-topology checkpoint:

| Context during encoding | Scaling bytes | Feature bytes | Offset bytes |
|---|---:|---:|---:|
| View topology enabled | 506,453 | 818,328 | 534,765 |
| Topology residual disabled | 592,004 | 818,328 | 534,765 |

Topology saves **85,551 Scaling bytes (14.45%)** on that fixed checkpoint. Feature and Offset sizes are exactly unchanged. After subtracting the 3,632-byte new MLP, the net conditional-coding gain is 81,919 bytes.

This is strong evidence that the view-geometry graph contains usable Scaling information and that all four entropy paths apply it consistently. It is not evidence that the current joint-training recipe improves the final rate-distortion point relative to a separately trained control.

## Package-size caveat

The view run writes a 458,375-byte `entropy_context.pth`. It packages existing `mlp_grid`, `mlp_deform`, train-camera geometry, and the new Scaling MLP so a fresh entropy decoder can reconstruct the same probabilities. The old baseline writes no equivalent context and is not independently entropy-decodable.

Consequently raw directory byte totals are not directly comparable: including `entropy_context.pth` penalizes only the corrected standalone contract, while excluding all learned entropy state makes neither package self-contained. The paper-style totals above count model parameters by their intended coding precision and remain the fair HAC++ metric; physical deployment packaging needs a separate common serialization contract.

## Recommendation

Stop after Phase 2A as requested. Do not enter decoded-neighbour causal Phase 2B yet.

A follow-up should first establish reproducible end-to-end gain with at least one of:

1. a shared iteration-15k checkpoint followed by matched topology-on/off fine-tuning;
2. several controlled seeds and confidence intervals;
3. a second scene, preferably Mip-NeRF360 `room`;
4. a smaller or regularized topology residual if the context is overfitting the topology-trained Scaling distribution.

The fixed-checkpoint result justifies retaining the Phase 2A implementation for further study, but the current same-code control fails the stated Go condition.

Raw result data are in `analysis_output_phase2a/summary.json` and `analysis_output_phase2a/scaling_rate_curve.csv`.
