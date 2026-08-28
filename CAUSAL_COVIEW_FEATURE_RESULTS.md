# View-Selected Causal CoView Feature Results

## Result

The candidate-ceiling ablation showed that increasing Euclidean candidates or
adding MinHash/LSH candidates did not explain the weak Feature result. The
follow-up therefore changes the information source rather than the candidate
range: selected neighbours contribute only attributes decoded in earlier
coding groups.

The final Frozen prototype obtains a positive conditional and net coding result
on both tested scenes:

| Scene | Valid anchors | Base Feature stream (B) | Causal Feature stream (B) | Conditional delta (B) | FP16 prior (B) | Net delta (B) |
|---|---:|---:|---:|---:|---:|---:|
| playroom | 154,413 | 2,451,588 | 2,448,872 | -2,716 | 123 | **-2,593** |
| drjohnson | 194,968 | 2,785,379 | 2,783,015 | -2,364 | 123 | **-2,241** |

Net Feature size falls by about 0.106% on `playroom` and 0.080% on
`drjohnson`. These are small but genuine net reductions: the serialized causal
model is included. Both packages pass fresh-process arithmetic decoding with
exact quantized Feature symbol-index checksums.

Because the experiment is Frozen, Feature symbols, anchors, hash parameters,
renderer inputs, and reconstruction are identical. It evaluates conditional
and net coding only; it is not yet a rate-distortion result.

## Final causal contract

- Anchors retain the current HAC++ canonical order.
- Anchor `i` belongs to deterministic group `i mod 4`.
- A target may read only graph neighbours in strictly lower groups.
- Every group is parallel; no future symbol is visible.
- The graph remains geometry-only and train-camera-only.
- Feature retains the HAC++ 5x10 `Channel_CTX_fea` order.
- The causal expert is a third Gaussian component. The original two HAC++
  components are unchanged and their probability mass is gated, rather than
  having their mean/log-scale directly overwritten.

The selected graph has no dense NxN matrix. On `playroom`, 73.49% of anchors
have at least one causal neighbour and the mean is 3.09 causal neighbours per
anchor.

## Why the parameter-efficient prior wins

The initial 21-16-21 MLP confirms that decoded neighbour attributes contain
more usable information than geometry-only context, but its 1,559-byte FP16
cost exceeds the conditional saving. The size/benefit ablation is:

| Prior | Conditional delta (B) | Model (B) | Net delta (B) |
|---|---:|---:|---:|
| MLP hidden 1 | +40 | 269 | +309 |
| MLP hidden 2 | -47 | 355 | +308 |
| MLP hidden 4 | -118 | 527 | +409 |
| MLP hidden 16 | -423 | 1,559 | +1,136 |

The final affine prior uses only 15 learned values: for each 10-channel Feature
group, one neighbour-mean interpolation value, one relative-scale value, and
one mixture gate. Its FP16 package is 123 bytes. Increasing the initialized
fusion strength improves the real stream monotonically in the tested range;
the selected gate `+4` gives the best `playroom` net result.

Eight coding groups improve estimated likelihood because more anchors have
decoded context, but extra arithmetic-coder group boundaries reduce the real
byte gain. Four groups are retained.

## Next integration stages

1. **Feature training and full codec**
   - Refresh the deterministic causal graph after densification stops.
   - Use quantized lower-group Feature values in the rate loss from the
     post-densification stage onward.
   - Encode and decode Features group-by-group with the tested three-component
     mixture path.
   - Serialize the 15 parameters and graph contract in `entropy_context.pth`.

2. **Scaling causal prior**
   - Keep the existing geometry-only Scaling expert.
   - Add decoded lower-group Scaling as a separate affine Gaussian expert.
   - Use a three-way gate between HAC++, geometry CoView, and causal CoView.
   - Scaling is encoded before Offset and is the next safest full-codec target.

3. **Offset causal prior**
   - Normalize target and neighbour Offset by decoded local Scaling.
   - Aggregate and predict each of the ten offset indices separately.
   - Condition on decoded Scaling and, only where codec order permits, coarse
     decoded Feature groups.

4. **Rate-distortion validation**
   - Run Control and causal CoView at multiple lambdas on multiple scenes.
   - Report total package size, PSNR/SSIM/LPIPS, encode/decode time, RD curves,
     and BD-Rate.

The current evidence supports the narrower claim that a parameter-efficient,
view-selected causal Feature prior provides reproducible net coding value. It
does not yet support a universal RD improvement claim for all attributes.

Server artifacts:

```text
/mnt/003/experiments/causal_coview_feature/playroom/affine_g4_gate_p4/
/mnt/003/experiments/causal_coview_feature/drjohnson/affine_g4_gate_p4/
```
