# CoView Context V2 Design

## Decision

The current 15-D geometry-only residual remains a useful cheap prior, but it is
not expressive enough to serve Feature, Scaling, and Offset uniformly.
Scaling is directly coupled to geometry, whereas Feature and Offset need
attribute-dependent context. The next model should therefore be a
**view-selected causal graph entropy model**, not a larger topology MLP.

This change introduces the first prerequisite: a deterministic `hybrid`
candidate mode. It unions Euclidean candidates with deterministic MinHash/LSH
sparse-observation candidates, evaluates exact Jaccard only inside that bounded
pool, and then applies the existing distance/depth ranking. The legacy `spatial`
mode remains the default so all previous baselines are reproducible.

## Why the current residual saturates

1. The nearest-16 prefilter prevents a strong non-local view relation from ever
   reaching the Top-8 graph.
2. Mean/std/max pooling removes neighbour identity and most edge structure.
3. The context contains no decoded Feature, Scaling, or Offset symbols. It can
   predict geometry-correlated Scaling, but it has little direct evidence for
   appearance Feature or local Gaussian displacement.
4. Adding residuals to the HAC++ mean and log-scale is easy for the base
   predictor to absorb. Frozen A2 already showed that most Feature gain
   disappears when the base predictor is re-optimized.
5. One attribute-agnostic descriptor forces three statistically different
   streams to use the same information bottleneck.

## Stage A: candidate-ceiling ablation

Keep the 15-D descriptor and entropy heads unchanged. Compare:

```text
candidate_mode = spatial, spatial_k = 16 / 32 / 64 / 128
candidate_mode = hybrid,  spatial_k = 16 / 32 / 64,
                          view_k = 16 / 32 / 64
final Top-K = 8
```

Report conditional bytes, net bytes, topology time, decode time,
`selected_outside_spatial_fraction`, and final RD metrics. This isolates the
candidate ceiling from a change in the entropy network.

Recommended first command-line configuration:

```text
--view_topology_candidate_mode hybrid
--view_topology_candidates 32
--view_topology_view_candidates 32
--view_topology_k 8
```

## Stage B: view-selected causal attribute context

Use the hybrid graph only to select relations. Split canonical anchors into
deterministic coding groups. Group 0 uses the baseline HAC++ prior; group `g`
may aggregate only already decoded groups `< g`.

```text
decoded neighbour symbols + edge geometry
                  |
             attribute encoder
                  |
        permutation-invariant weighted pool
                  |
HAC++ prior ------+------ CoView causal prior
                  |
          learned mixture/gating
                  |
        arithmetic-coding distribution
```

This preserves a standalone decoder contract and permits group-parallel rather
than fully serial decoding.

### Feature

- Retain the existing 5x10 `Channel_CTX_fea` order.
- For each channel group, aggregate the same already decoded channel groups
  from causal CoView neighbours.
- Fuse hash-grid, intra-anchor channel context, and cross-anchor CoView context
  as separate experts or tokens.
- Start with a low-rank 16-D neighbour projection and one gated mixture head;
  do not add a large full-channel MLP.

This follows the space-channel principle of ELIC and the complementary
hierarchical/autoregressive priors of Minnen et al., adapted from a 2-D raster
to the view-selected anchor graph.

### Scaling

- Keep the geometry-only prior as the cheap base path.
- Add decoded-neighbour Scaling only in later coding groups.
- Let the gate choose between HAC++, geometry-only CoView, and causal CoView.

Scaling is the safest first end-to-end implementation because the existing
experiments already establish conditional value.

### Offset

- Encode/predict Offset after Scaling.
- Normalize neighbour and target offsets by decoded local Scaling before graph
  aggregation; raw world-coordinate offsets mix different anchor scales.
- Aggregate per-offset-index context rather than flattening all 30 values into
  one attribute-agnostic head.
- Condition the Offset distribution on decoded Scaling and coarse Feature
  groups, because Offset controls local Gaussian placement rather than anchor
  size alone.

The present `32 -> 60` residual head has none of these structures, which is a
more plausible cause of weak results than insufficient head width.

## Stage C: multi-prior fusion instead of additive residuals

Do not force CoView to perturb the HAC++ Gaussian directly. Predict multiple
valid priors and combine their probabilities:

```text
p(x) = alpha_hash * p_hash(x)
     + alpha_view * p_view(x)
     + alpha_causal * p_causal(x)
```

with non-negative normalized gates. This is easier to diagnose than a mean and
log-scale residual: per-stream gate utilization and cross-entropy contribution
show whether CoView adds information or merely duplicates HAC++.

A cheaper alternative is FiLM modulation of the existing `mlp_grid` hidden
feature. It should be tested after mixture fusion, because direct concatenation
can let the larger base network silently absorb the new signal again.

## Lessons from related codecs

- [ContextGS](https://arxiv.org/abs/2405.20721) uses anchor-level
  autoregressive levels and a hyperprior. The relevant lesson is to condition
  on already coded anchors, not only on geometry.
- [HEMGS](https://arxiv.org/abs/2411.18473) combines a hyperprior with adaptive
  autoregressive context selection. The relevant lesson is flexible receptive
  fields plus complementary priors.
- [FCGS](https://arxiv.org/abs/2410.08017) combines inter-Gaussian spatial,
  intra-Gaussian channel, and hyperprior paths. Its implementation also shows
  that deterministic accumulation is a codec requirement, not just a testing
  detail.
- [PCGS](https://arxiv.org/abs/2503.08511) uses previously decoded progressive
  levels to refine later probability prediction.
- [ELIC](https://arxiv.org/abs/2203.10886) uses uneven channel groups and
  space-channel context for efficient parallel coding.
- [Joint Autoregressive and Hierarchical Priors](https://arxiv.org/abs/1809.02736)
  establishes that side/hierarchical and causal contexts are complementary.
- [Deep Contextual Video Compression](https://arxiv.org/abs/2109.15047) replaces
  simple residual prediction with feature-domain conditional coding. The
  analogue here is to use view-aligned decoded anchor features as a condition,
  rather than treating CoView as a scalar correction.

## Required evaluation order

1. Frozen same-symbol conditional rate.
2. Add serialized model and all camera/topology metadata for net rate.
3. Fresh-process encode/decode checksum and reconstruction equality.
4. Joint multi-lambda RD curves and BD-Rate on multiple scenes.

Feature, Scaling, and Offset should be reported separately. A universal
context claim requires consistent net or RD benefit across the three streams;
one positive Scaling point is not sufficient.
