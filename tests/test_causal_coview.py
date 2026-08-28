import numpy as np
import torch

from scene.coview_causal_context import (
    AffineCausalFeaturePrior,
    AffineCausalScalingPrior,
    CausalFeaturePrior,
    build_causal_anchor_graph,
    causal_neighbor_statistics,
)
from scene.coview_context import ViewTopologyContext


def _topology():
    neighbors = np.asarray([
        [1, 2], [0, 2], [1, 3], [1, 2],
        [5, 6], [4, 6], [5, 7], [5, 6],
    ], dtype=np.int64)
    scores = np.ones_like(neighbors, dtype=np.float32)
    return ViewTopologyContext(
        features=np.zeros((8, 15), dtype=np.float32),
        neighbors=neighbors,
        distance_scores=scores,
        depth_scores=scores,
        diagnostics={},
    )


def test_causal_graph_is_deterministic_and_has_no_future_edges():
    first = build_causal_anchor_graph(_topology(), num_groups=4)
    second = build_causal_anchor_graph(_topology(), num_groups=4)

    torch.testing.assert_close(first.groups, second.groups)
    torch.testing.assert_close(first.neighbors, second.neighbors)
    torch.testing.assert_close(first.weights, second.weights)
    assert first.diagnostics["future_edge_violations"] == 0
    assert first.diagnostics["dense_anchor_pair_matrix_created"] is False
    valid = first.neighbors >= 0
    target_groups = first.groups[:, None].expand_as(first.neighbors)
    source_groups = first.groups[torch.clamp(first.neighbors, min=0)]
    assert torch.all(source_groups[valid] < target_groups[valid])


def test_group_zero_has_no_context_and_weights_are_normalized():
    graph = build_causal_anchor_graph(_topology(), num_groups=4)
    assert torch.all(graph.support[graph.groups == 0] == 0)
    weight_sum = graph.weights.sum(dim=1)
    has_context = graph.support[:, 0] > 0
    torch.testing.assert_close(weight_sum[has_context], torch.ones_like(weight_sum[has_context]))
    torch.testing.assert_close(weight_sum[~has_context], torch.zeros_like(weight_sum[~has_context]))


def test_neighbor_statistics_cannot_read_a_future_group():
    graph = build_causal_anchor_graph(_topology(), num_groups=4)
    values = torch.arange(8, dtype=torch.float32)[:, None]
    changed = values.clone()
    changed[3] = 10_000.0
    indices = torch.tensor([2])

    first = causal_neighbor_statistics(
        values, graph.neighbors, graph.weights, indices
    )
    second = causal_neighbor_statistics(
        changed, graph.neighbors, graph.weights, indices
    )
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    torch.testing.assert_close(first[0], torch.tensor([[1.0]]))


def test_causal_feature_prior_preserves_5x10_shapes_and_zero_support():
    prior = CausalFeaturePrior(hidden_dim=8)
    base_mean = torch.zeros(2, 50)
    base_scale = torch.ones(2, 50)
    q_feature = torch.ones(2, 50)
    neighbor_mean = torch.full((2, 50), 0.25)
    neighbor_std = torch.full((2, 50), 0.5)
    support = torch.tensor([[0.0], [0.5]])

    mean, scale, weight = prior(
        base_mean,
        base_scale,
        q_feature,
        neighbor_mean,
        neighbor_std,
        support,
    )
    assert mean.shape == scale.shape == weight.shape == (2, 50)
    assert torch.all(weight[0] == 0)
    assert torch.all(weight[1] > 0)
    assert torch.all(weight <= 0.25)
    for chunk in range(5):
        chunk_weight = weight[:, chunk * 10:(chunk + 1) * 10]
        torch.testing.assert_close(
            chunk_weight, chunk_weight[:, :1].expand_as(chunk_weight)
        )

    selected = prior.forward_selected(
        base_mean[:, :10],
        base_scale[:, :10],
        q_feature[:, :10],
        neighbor_mean[:, :10],
        neighbor_std[:, :10],
        support,
    )
    assert all(value.shape == (2, 10) for value in selected)


def test_affine_prior_uses_only_fifteen_parameters_and_chunk_identity():
    prior = AffineCausalFeaturePrior()
    assert sum(parameter.numel() for parameter in prior.parameters()) == 15
    values = torch.zeros(2, 10)
    support = torch.tensor([[0.0], [1.0]])
    output = prior.forward_selected(
        values,
        torch.ones_like(values),
        torch.ones_like(values),
        torch.full_like(values, 0.5),
        torch.ones_like(values),
        support,
        chunk_indices=(3,),
    )
    assert all(value.shape == (2, 10) for value in output)
    assert torch.all(output[2][0] == 0)


def test_affine_scaling_prior_is_small_and_support_gated():
    prior = AffineCausalScalingPrior()
    assert sum(parameter.numel() for parameter in prior.parameters()) == 18
    base_mean = torch.zeros(2, 6)
    base_scale = torch.ones(2, 6)
    q_scaling = torch.full((2, 6), 0.001)
    neighbor_mean = torch.full((2, 6), 0.25)
    neighbor_std = torch.full((2, 6), 0.5)
    support = torch.tensor([[0.0], [0.75]])

    mean, scale, weight = prior(
        base_mean,
        base_scale,
        q_scaling,
        neighbor_mean,
        neighbor_std,
        support,
    )
    assert mean.shape == scale.shape == weight.shape == (2, 6)
    assert torch.all(weight[0] == 0)
    assert torch.all(weight[1] > 0)
    assert torch.all(weight <= 0.25)
