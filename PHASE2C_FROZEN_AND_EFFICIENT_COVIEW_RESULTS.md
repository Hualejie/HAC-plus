# Phase 2C: Frozen and Parameter-Efficient CoView Results

## Decision

Phase 2C is complete for Deep Blending `playroom`, with a second-scene Scaling
replication on `drjohnson`.

The result is a **narrow Feature Go under the Frozen-A1 test, but no automatic
Go to Attribute-Similarity, and a No-Go for the current Scaling generalization
claim**:

- for identical frozen Feature symbols and a frozen HAC++ base predictor,
  CoView lowers the real Feature stream by 12,496 bytes with the `32->100` head
  and by 4,090 bytes with the parameter-efficient `32->10` head;
- the `32->10` head is net-positive after actual model serialization: +550
  bytes with FP32, +2,237 bytes with FP16, and +3,049 bytes with INT8;
- when the base entropy predictor is allowed the same matched entropy-only
  optimization, CoView saves only 41 raw bytes and loses 3,499 bytes after its
  FP32 model is counted. The base predictor therefore absorbs essentially all
  of the incremental gain in this experiment;
- Scaling FP16 improves the `playroom` net saving to 4,649 bytes, but INT8 makes
  the arithmetic stream 140,149 bytes larger than FP16 and is decisively worse;
- Scaling does not replicate on `drjohnson`: the FP32 CoView result is 13,490
  total bytes larger than Control, with slightly worse SSIM and LPIPS.

These results establish a pure conditional-rate gain in the controlled A1
setting and show that a cheap Feature head can retain it. They do not establish
that the gain remains useful once HAC++'s existing predictor is equally adapted,
nor that Scaling has a reproducible cross-scene end-to-end gain. Phase 2C stops
here as required.

## Three distinct claims

The experiments support three deliberately separate conclusions.

### 1. Model reliance

Phase 2B's same-checkpoint ON/OFF diagnostic measured the behavior of models
trained with CoView. Feature and Scaling streams increased by 304,397 and
121,609 bytes respectively when their trained CoView branch was disabled.

This proves that those trained models rely on View-Geometry Context. It does
not compare the same representation under independently controlled predictors,
and is not a pure conditional-rate or net-coding result.

### 2. Pure conditional-rate gain

Phase 2C A1 fixes the representation, Feature symbols, hash, `mlp_grid`, and
`Channel_CTX_fea` / `mlp_deform`. Only the CoView residual is optimized. On the
same 106,382 anchors and exactly the same integer Feature symbol indices:

| Feature head | Base stream B | CoView stream B | Raw saving B |
|---|---:|---:|---:|
| `32->100`, FP32 | 880,608 | 868,112 | **12,496** |
| `32->10`, FP32 | 880,608 | 876,518 | **4,090** |
| `32->10`, FP16 | 880,608 | 876,517 | **4,091** |
| `32->10`, INT8 | 880,608 | 876,528 | **4,080** |

Therefore:

```text
R_feature(base + CoView) < R_feature(base)
```

is true for an identical frozen representation. The compact head retains about
one third of the full head's raw saving.

### 3. Net coding gain

The real serialized CoView blob must be added to the arithmetic stream:

| Feature head | Storage | Stream B | Model B | Stream + model B | Net saved B |
|---|---|---:|---:|---:|---:|
| `32->100` | FP32 | 868,112 | 15,420 | 883,532 | -2,924 |
| `32->10` | FP32 | 876,518 | 3,540 | 880,058 | **+550** |
| `32->10` | FP16 | 876,517 | 1,854 | 878,371 | **+2,237** |
| `32->10` | INT8 | 876,528 | 1,031 | 877,559 | **+3,049** |

The full head does not break even. All three compact serializations do break
even in A1, with INT8 giving the largest Feature net saving. This is measured
with actual blobs and real 5x10 mixture arithmetic coding, not a parameter-count
estimate.

## Frozen Feature experiment contract

Both A1 and A2 use the Control 30k representation. The following remain fixed:

- canonical quantized anchor positions and valid-anchor order;
- Feature, Scaling, Offset, masks, and hash representation;
- renderer state and all non-entropy scene representation;
- Feature quantization and integer symbol indices;
- HAC++'s five chunks of ten channels, mixture probabilities, intra-feature
  context semantics, and arithmetic coding order.

`valid_original_idx` and `codec_original_idx` are preserved while capturing the
frozen representation. Every reported branch has the same Feature
symbol-index SHA-256:

```text
5a61a8215d2c4b89deeb190c4aa4ecae2733dfb664b5161f2a34d0e1086fb757
```

The parameter-efficient Feature head emits five mean residuals and five
log-scale residuals. Each residual pair is shared by the ten channels in the
corresponding HAC++ Feature chunk. The shared View-Geometry trunk remains
`15->32`; only the Feature output changes from 100 to 10.

## A2 matched entropy-only optimization

A2 starts Frozen-Control and Frozen-CoView from identical common entropy state.
It gives both branches the same sampled-anchor schedule, RNG seed, batch size,
number of steps, and base-predictor optimization. Frozen-CoView alone also
updates its CoView residual branch.

| Branch | Feature stream B | CoView model B | Feature + model B |
|---|---:|---:|---:|
| Frozen-Control | 848,430 | 0 | 848,430 |
| Frozen-CoView `32->10` FP32 | 848,389 | 3,540 | 851,929 |

The raw difference is only **41 bytes**. After signaling the residual model,
Frozen-CoView is **3,499 bytes worse**.

This does not contradict A1. A1 shows that CoView can improve a fixed base
distribution for the same symbols. A2 shows that, under this optimization
budget, HAC++'s existing `mlp_grid` and `Channel_CTX_fea` predictor can absorb
nearly all of that available improvement without an extra signaled model.

## Scaling serialization on `playroom`

The trained Phase 2B Scaling representation is encoded with decoder-equivalent
deserialized parameters for every storage format.

| Storage | Scaling stream B | Model B | Total package B | Saved vs Control B |
|---|---:|---:|---:|---:|
| Control | 519,906 | 0 | 2,556,384 | 0 |
| FP32 | 514,134 | 3,804 | 2,553,607 | **+2,777** |
| FP16 | 514,080 | 1,986 | 2,551,735 | **+4,649** |
| INT8 | 654,229 | 1,097 | 2,690,995 | -134,611 |

FP16 is the best Scaling serialization tested. It reduces both the blob and the
Scaling stream relative to FP32. INT8 makes the model another 889 bytes smaller
than FP16, but perturbs the probability parameters enough to increase the
Scaling arithmetic stream by 140,149 bytes. Compressing an entropy network must
therefore be evaluated by `stream + model`, not model bytes alone.

All three Scaling formats fresh-decode to the same representation checksum:

```text
d00373b393cf4752f98e440cea6b23dcfbc51f0bf8990b538d1fbb739623c658
```

The fidelity metrics are those of the same trained Scaling representation:
PSNR 30.711266, SSIM 0.909583, and LPIPS 0.267306.

## Scaling replication on `drjohnson`

| Branch | Total B | Saved vs Control B | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Control | 3,510,054 | 0 | 29.641968 | 0.904935 | 0.263852 |
| Scaling FP32 | 3,523,544 | -13,490 | 29.665388 | 0.904815 | 0.264290 |

Scaling FP32 changes fidelity by +0.023420 dB PSNR, -0.000120 SSIM, and
+0.000438 LPIPS. It increases total size by 13,490 bytes. The required
cross-scene condition—smaller total size without fidelity degradation—is not
met.

## Serialization and decoder validation

The CoView format records named tensors, shape/dtype metadata, serialized byte
count, and SHA-256. FP32, FP16, and symmetric per-tensor INT8 are implemented.
The encoder deserializes the just-written blob before producing probabilities,
so its parameter values match the fresh decoder rather than an untransmitted
training state.

Fresh-process arithmetic decode passed for the tested Feature FP32/FP16/INT8
packages and for all Scaling serializations and both `drjohnson` packages.
Feature validation uses the integer codec symbol:

```text
round(feature / Q_feat)
```

All decoded integer symbol checksums match. A float-tensor byte checksum may
differ only through `-0.0` versus `+0.0`; this is not an arithmetic-symbol or
representation mismatch.

## Answers to the Phase 2C questions

1. **Same frozen Feature symbols:** yes, CoView lowers conditional rate in A1.
2. **`32->10` head:** yes, it retains a smaller but real 4,080-4,091-byte raw
   saving depending on serialization.
3. **Feature net gain after real model bytes:** yes in A1 for compact FP32,
   FP16, and INT8; no for the full head, and no in A2.
4. **Matched entropy-only result:** the raw gain collapses to 41 bytes and the
   net result is -3,499 bytes. It is not practically useful.
5. **Scaling cross-scene replication:** no; `drjohnson` is 13,490 bytes larger.
6. **Scaling FP16/INT8:** FP16 improves the `playroom` net saving to 4,649
   bytes. INT8 severely worsens the arithmetic stream and total package.
7. **Enter Attribute-Similarity:** not automatically. A1 proves that a frozen
   base predictor can exploit CoView, but A2 shows negligible incremental gain
   after matched base adaptation. A new stage needs an explicit hypothesis and
   authorization rather than following from Phase 2C alone.

## Final recommendation

Do not claim a general cross-scene Scaling win, and do not add
Attribute-Similarity merely because A1 is positive. The strongest supported
claim is:

```text
On a fixed Feature representation and frozen HAC++ base predictor,
View-Geometry Context lowers the real conditional coding rate, and a compact
serialized head can retain a small positive net gain. That incremental gain is
almost entirely absorbed when the base entropy predictor is equally adapted.
```

If work continues after review, the next experiment should first determine why
A1 and A2 separate—using matched capacity/optimization diagnostics or a second
scene Frozen Feature replication—before introducing a new
Attribute-Similarity relation.

Raw results are in `analysis_output_phase2c/frozen_feature_rate.csv`,
`analysis_output_phase2c/feature_head_ablation.csv`,
`analysis_output_phase2c/scaling_replication.csv`,
`analysis_output_phase2c/model_serialization.csv`, and
`analysis_output_phase2c/summary.json`.
