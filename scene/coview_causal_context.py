"""Decoder-safe causal attribute context on a deterministic CoView graph."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from scene.coview_context import ViewTopologyContext


FEATURE_CHUNKS = 5
FEATURE_CHUNK_DIM = 10


@dataclass(frozen=True)
class CausalAnchorGraph:
    """Fixed-width graph whose edges only point to earlier coding groups."""

    groups: torch.Tensor
    neighbors: torch.Tensor
    weights: torch.Tensor
    support: torch.Tensor
    diagnostics: Dict[str, object]


def build_causal_anchor_graph(
    topology: ViewTopologyContext,
    num_groups: int = 4,
    epsilon: float = 1e-12,
) -> CausalAnchorGraph:
    """Orient selected CoView edges by a reproducible group-parallel schedule.

    Canonical anchor ``i`` belongs to ``i % num_groups``. An anchor may read
    only neighbours in a strictly lower group, so every group is parallel and
    a fresh decoder can reproduce the context without future-symbol leakage.
    """
    neighbors = np.asarray(topology.neighbors, dtype=np.int64)
    if neighbors.ndim != 2:
        raise ValueError("topology.neighbors must have shape [N, K]")
    if num_groups < 2:
        raise ValueError("num_groups must be at least two")
    num_anchors, width = neighbors.shape
    if np.any(neighbors < 0) or np.any(neighbors >= num_anchors):
        raise ValueError("topology contains an out-of-range neighbour")

    groups = np.arange(num_anchors, dtype=np.int64) % num_groups
    valid = groups[neighbors] < groups[:, None]
    joint_score = 0.5 * (
        np.asarray(topology.distance_scores, dtype=np.float64)
        + np.asarray(topology.depth_scores, dtype=np.float64)
    )
    raw_weights = np.where(valid, np.maximum(joint_score, 0.0), 0.0)
    weight_sum = raw_weights.sum(axis=1, keepdims=True)
    valid_count = valid.sum(axis=1, keepdims=True)
    uniform = np.divide(
        valid.astype(np.float64),
        np.maximum(valid_count, 1),
    )
    weights = np.where(
        weight_sum > epsilon,
        raw_weights / np.maximum(weight_sum, epsilon),
        uniform,
    ).astype(np.float32)
    causal_neighbors = np.where(valid, neighbors, -1)
    support = (valid_count[:, 0] / max(width, 1)).astype(np.float32)
    violations = valid & (groups[neighbors] >= groups[:, None])
    diagnostics = {
        "num_anchors": int(num_anchors),
        "num_groups": int(num_groups),
        "neighbors_per_anchor": int(width),
        "mean_causal_neighbor_count": float(valid_count.mean()),
        "mean_support_fraction": float(support.mean()),
        "anchors_with_context_fraction": float((valid_count[:, 0] > 0).mean()),
        "future_edge_violations": int(violations.sum()),
        "group_assignment": "canonical_index_modulo",
        "dense_anchor_pair_matrix_created": False,
    }
    return CausalAnchorGraph(
        groups=torch.from_numpy(groups),
        neighbors=torch.from_numpy(causal_neighbors),
        weights=torch.from_numpy(weights),
        support=torch.from_numpy(support[:, None]),
        diagnostics=diagnostics,
    )


def causal_neighbor_statistics(
    decoded_values: torch.Tensor,
    neighbors: torch.Tensor,
    weights: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
):
    """Return weighted mean/std using only the supplied decoded-value table."""
    if decoded_values.ndim != 2:
        raise ValueError("decoded_values must have shape [N, C]")
    if indices is not None:
        neighbors = neighbors[indices]
        weights = weights[indices]
    safe_neighbors = torch.clamp(neighbors, min=0)
    valid = neighbors >= 0
    effective_weights = weights * valid.to(weights.dtype)
    gathered = decoded_values[safe_neighbors]
    mean = (gathered * effective_weights[..., None]).sum(dim=1)
    variance = (
        (gathered - mean[:, None, :]).square()
        * effective_weights[..., None]
    ).sum(dim=1)
    return mean, torch.sqrt(torch.clamp(variance, min=0.0))


class CausalFeaturePrior(nn.Module):
    """Small shared 5x10 Feature expert fused as a third Gaussian prior."""

    def __init__(self, hidden_dim: int = 16, max_mixture_weight: float = 0.25):
        super().__init__()
        if not 0.0 < max_mixture_weight <= 1.0:
            raise ValueError("max_mixture_weight must be in (0, 1]")
        self.max_mixture_weight = float(max_mixture_weight)
        input_dim = FEATURE_CHUNK_DIM * 2 + 1
        output_dim = FEATURE_CHUNK_DIM * 2 + 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
        )
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        final.bias.data[-1] = -8.0

    def forward(
        self,
        base_mean: torch.Tensor,
        base_scale: torch.Tensor,
        q_feature: torch.Tensor,
        neighbor_mean: torch.Tensor,
        neighbor_std: torch.Tensor,
        support: torch.Tensor,
    ):
        return self.forward_selected(
            base_mean,
            base_scale,
            q_feature,
            neighbor_mean,
            neighbor_std,
            support,
        )

    def forward_selected(
        self,
        base_mean: torch.Tensor,
        base_scale: torch.Tensor,
        q_feature: torch.Tensor,
        neighbor_mean: torch.Tensor,
        neighbor_std: torch.Tensor,
        support: torch.Tensor,
    ):
        """Evaluate one or more complete 10-channel Feature chunks."""
        if base_mean.shape[-1] % FEATURE_CHUNK_DIM:
            raise ValueError("selected Feature channels must contain full chunks")
        batch = base_mean.shape[0]
        num_chunks = base_mean.shape[-1] // FEATURE_CHUNK_DIM
        base_mean_chunks = base_mean.view(batch, num_chunks, FEATURE_CHUNK_DIM)
        base_scale_chunks = torch.clamp(
            base_scale.view(batch, num_chunks, FEATURE_CHUNK_DIM),
            min=1e-9,
        )
        q_chunks = q_feature.view(batch, num_chunks, FEATURE_CHUNK_DIM)
        neighbor_mean_chunks = neighbor_mean.view(
            batch, num_chunks, FEATURE_CHUNK_DIM
        )
        neighbor_std_chunks = neighbor_std.view(
            batch, num_chunks, FEATURE_CHUNK_DIM
        )
        normalized_delta = (
            neighbor_mean_chunks - base_mean_chunks
        ) / base_scale_chunks
        normalized_std = torch.log(torch.clamp(
            (neighbor_std_chunks + 0.5 * q_chunks) / base_scale_chunks,
            min=1e-6,
            max=1e6,
        ))
        chunk_support = support[:, None, :].expand(-1, num_chunks, -1)
        network_input = torch.cat(
            (normalized_delta, normalized_std, chunk_support), dim=-1
        )
        output = self.network(network_input)
        mean_residual = output[..., :FEATURE_CHUNK_DIM]
        log_scale = torch.clamp(
            output[..., FEATURE_CHUNK_DIM:2 * FEATURE_CHUNK_DIM],
            min=-5.0,
            max=5.0,
        )
        gate_logit = output[..., -1:]
        # This is a separate Gaussian expert, but its zero-output state starts
        # from the stable HAC++ first prior. CoView statistics enter through
        # ``network_input`` and learn a conditional departure from that prior.
        causal_mean = base_mean_chunks + mean_residual * base_scale_chunks
        causal_scale = base_scale_chunks * torch.exp(log_scale)
        mixture_weight = (
            self.max_mixture_weight * torch.sigmoid(gate_logit) * chunk_support
        )
        return (
            causal_mean.reshape_as(base_mean),
            causal_scale.reshape_as(base_scale),
            mixture_weight.repeat_interleave(
                FEATURE_CHUNK_DIM, dim=1
            ).reshape_as(base_mean),
        )
