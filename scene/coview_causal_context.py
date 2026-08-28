"""Decoder-safe causal attribute context on a deterministic CoView graph."""

from dataclasses import dataclass
from pathlib import Path
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
        chunk_indices=None,
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


class AffineCausalFeaturePrior(nn.Module):
    """Fifteen-parameter causal prior shared with HAC++ mixture moments."""

    def __init__(self, max_mixture_weight: float = 0.25):
        super().__init__()
        if not 0.0 < max_mixture_weight <= 1.0:
            raise ValueError("max_mixture_weight must be in (0, 1]")
        self.max_mixture_weight = float(max_mixture_weight)
        self.mean_blend = nn.Parameter(torch.zeros(FEATURE_CHUNKS))
        self.log_scale_blend = nn.Parameter(torch.zeros(FEATURE_CHUNKS))
        self.gate_logit = nn.Parameter(torch.full((FEATURE_CHUNKS,), -8.0))

    def forward(
        self,
        base_mean,
        base_scale,
        q_feature,
        neighbor_mean,
        neighbor_std,
        support,
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
        base_mean,
        base_scale,
        q_feature,
        neighbor_mean,
        neighbor_std,
        support,
        chunk_indices=None,
    ):
        if base_mean.shape[-1] % FEATURE_CHUNK_DIM:
            raise ValueError("selected Feature channels must contain full chunks")
        batch = base_mean.shape[0]
        num_chunks = base_mean.shape[-1] // FEATURE_CHUNK_DIM
        if chunk_indices is None:
            if num_chunks != FEATURE_CHUNKS:
                raise ValueError("chunk_indices are required for a Feature subset")
            chunk_indices = tuple(range(FEATURE_CHUNKS))
        if len(chunk_indices) != num_chunks:
            raise ValueError("chunk_indices length does not match selected Feature chunks")
        parameter_indices = torch.as_tensor(
            chunk_indices, dtype=torch.long, device=base_mean.device
        )
        base_mean_chunks = base_mean.view(batch, num_chunks, FEATURE_CHUNK_DIM)
        base_scale_chunks = torch.clamp(
            base_scale.view(batch, num_chunks, FEATURE_CHUNK_DIM), min=1e-9
        )
        q_chunks = q_feature.view(batch, num_chunks, FEATURE_CHUNK_DIM)
        neighbor_mean_chunks = neighbor_mean.view(
            batch, num_chunks, FEATURE_CHUNK_DIM
        )
        neighbor_std_chunks = neighbor_std.view(
            batch, num_chunks, FEATURE_CHUNK_DIM
        )
        beta = torch.tanh(self.mean_blend[parameter_indices])[None, :, None]
        gamma = self.log_scale_blend[parameter_indices][None, :, None]
        causal_mean = base_mean_chunks + beta * (
            neighbor_mean_chunks - base_mean_chunks
        )
        relative_std = torch.log(torch.clamp(
            (neighbor_std_chunks + 0.5 * q_chunks) / base_scale_chunks,
            min=1e-6,
            max=1e6,
        ))
        causal_scale = base_scale_chunks * torch.exp(torch.clamp(
            gamma * relative_std, min=-5.0, max=5.0
        ))
        gate = self.max_mixture_weight * torch.sigmoid(
            self.gate_logit[parameter_indices]
        )[None, :, None]
        mixture_weight = gate * support[:, None, :]
        return (
            causal_mean.reshape_as(base_mean),
            causal_scale.reshape_as(base_scale),
            mixture_weight.repeat_interleave(
                FEATURE_CHUNK_DIM, dim=1
            ).reshape_as(base_mean),
        )


class AffineCausalScalingPrior(nn.Module):
    """Per-channel affine expert for six decoded Scaling values."""

    CHANNELS = 6

    def __init__(self, max_mixture_weight: float = 0.25):
        super().__init__()
        if not 0.0 < max_mixture_weight <= 1.0:
            raise ValueError("max_mixture_weight must be in (0, 1]")
        self.max_mixture_weight = float(max_mixture_weight)
        self.mean_blend = nn.Parameter(torch.zeros(self.CHANNELS))
        self.log_scale_blend = nn.Parameter(torch.zeros(self.CHANNELS))
        self.gate_logit = nn.Parameter(torch.full((self.CHANNELS,), -8.0))

    def forward(
        self,
        base_mean,
        base_scale,
        q_scaling,
        neighbor_mean,
        neighbor_std,
        support,
    ):
        expected = (base_mean.shape[0], self.CHANNELS)
        for name, value in (
            ("base_mean", base_mean),
            ("base_scale", base_scale),
            ("q_scaling", q_scaling),
            ("neighbor_mean", neighbor_mean),
            ("neighbor_std", neighbor_std),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if support.shape != (base_mean.shape[0], 1):
            raise ValueError("support must have shape [N, 1]")
        beta = torch.tanh(self.mean_blend)[None, :]
        causal_mean = base_mean + beta * (neighbor_mean - base_mean)
        relative_std = torch.log(torch.clamp(
            (neighbor_std + 0.5 * q_scaling)
            / torch.clamp(base_scale, min=1e-9),
            min=1e-6,
            max=1e6,
        ))
        causal_scale = torch.clamp(base_scale, min=1e-9) * torch.exp(
            torch.clamp(
                self.log_scale_blend[None, :] * relative_std,
                min=-5.0,
                max=5.0,
            )
        )
        gate = self.max_mixture_weight * torch.sigmoid(
            self.gate_logit
        )[None, :]
        return causal_mean, causal_scale, gate * support


def mixture_moments(means, scales, probabilities):
    """Moment-match a Gaussian mixture for the causal expert reference."""
    mixture_mean = sum(
        probability * mean
        for probability, mean in zip(probabilities, means)
    )
    second_moment = sum(
        probability * (scale.square() + mean.square())
        for probability, mean, scale in zip(probabilities, means, scales)
    )
    mixture_scale = torch.sqrt(torch.clamp(
        second_moment - mixture_mean.square(), min=1e-9
    ))
    return mixture_mean, mixture_scale


def causal_feature_components(
    model,
    causal_prior,
    symbol_state,
    q_feature,
    anchors,
    neighbor_mean,
    neighbor_std,
    support,
    chunk=None,
    topology_features=None,
):
    """Build the two HAC++ Feature priors plus one causal CoView prior."""
    context = model.calc_interp_feat(anchors)
    outputs = torch.split(
        model.get_grid_mlp(context),
        [
            model.feat_dim, model.feat_dim, model.feat_dim,
            6, 6, 3 * model.n_offsets, 3 * model.n_offsets,
            1, 1, 1,
        ],
        dim=-1,
    )
    mean, scale, probability = outputs[:3]
    if model.coview_enabled:
        mean, scale = model.apply_coview_entropy_context(
            mean, scale, topology_features, "feature"
        )
    mean_scale = torch.cat((mean, scale, probability), dim=-1)
    if chunk is None:
        mean_adjusted, scale_adjusted, probability_adjusted = (
            model.get_deform_mlp.forward(symbol_state, mean_scale)
        )
        feature_slice = slice(None)
    else:
        mean_adjusted, scale_adjusted, probability_adjusted = (
            model.get_deform_mlp.forward(symbol_state, mean_scale, to_dec=chunk)
        )
        feature_slice = slice(
            chunk * FEATURE_CHUNK_DIM,
            (chunk + 1) * FEATURE_CHUNK_DIM,
        )
    base_probabilities = torch.softmax(
        torch.stack(
            (probability[:, feature_slice], probability_adjusted), dim=-1
        ),
        dim=-1,
    )
    selected_means = (mean[:, feature_slice], mean_adjusted)
    selected_scales = (
        torch.clamp(scale[:, feature_slice], min=1e-9),
        torch.clamp(scale_adjusted, min=1e-9),
    )
    mixture_mean, mixture_scale = mixture_moments(
        selected_means,
        selected_scales,
        (base_probabilities[..., 0], base_probabilities[..., 1]),
    )
    causal_mean, causal_scale, causal_weight = causal_prior.forward_selected(
        mixture_mean,
        mixture_scale,
        q_feature[:, feature_slice],
        neighbor_mean[:, feature_slice],
        neighbor_std[:, feature_slice],
        support,
        chunk_indices=None if chunk is None else (chunk,),
    )
    causal_weight = torch.clamp(causal_weight, min=0.0, max=1.0 - 1e-6)
    base_mass = 1.0 - causal_weight
    return (
        (mean[:, feature_slice], mean_adjusted, causal_mean),
        (
            torch.clamp(scale[:, feature_slice], min=1e-9),
            torch.clamp(scale_adjusted, min=1e-9),
            torch.clamp(causal_scale, min=1e-9),
        ),
        (
            base_probabilities[..., 0] * base_mass,
            base_probabilities[..., 1] * base_mass,
            causal_weight,
        ),
    )


def causal_scaling_components(
    model,
    causal_prior,
    q_scaling,
    anchors,
    neighbor_mean,
    neighbor_std,
    support,
    topology_features=None,
):
    """Build the HAC++ Scaling prior plus one causal decoded-neighbour prior."""
    outputs = torch.split(
        model.get_grid_mlp(model.calc_interp_feat(anchors)),
        [
            model.feat_dim, model.feat_dim, model.feat_dim,
            6, 6, 3 * model.n_offsets, 3 * model.n_offsets,
            1, 1, 1,
        ],
        dim=-1,
    )
    mean_scaling, scale_scaling = outputs[3], outputs[4]
    if model.coview_enabled:
        mean_scaling, scale_scaling = model.apply_coview_entropy_context(
            mean_scaling, scale_scaling, topology_features, "scaling"
        )
    scale_scaling = torch.clamp(scale_scaling, min=1e-9)
    causal_mean, causal_scale, causal_weight = causal_prior(
        mean_scaling,
        scale_scaling,
        q_scaling,
        neighbor_mean,
        neighbor_std,
        support,
    )
    causal_weight = torch.clamp(causal_weight, min=0.0, max=1.0 - 1e-6)
    return (
        (mean_scaling, causal_mean),
        (scale_scaling, torch.clamp(causal_scale, min=1e-9)),
        (1.0 - causal_weight, causal_weight),
        q_scaling,
    )


@torch.no_grad()
def encode_causal_feature_symbols(
    model,
    causal_prior,
    symbols,
    q_feature,
    anchors,
    graph,
    output,
    batch_size=3000,
    topology_features=None,
):
    """Encode canonical Feature symbols in decoder-safe causal groups."""
    from utils.encodings_cuda import encoder_gaussian_mixed_chunk

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    neighbors = graph.neighbors.to(symbols.device)
    weights = graph.weights.to(symbols.device)
    support = graph.support.to(symbols.device)
    groups = graph.groups.to(symbols.device)
    bit_count = 0
    for group in range(graph.diagnostics["num_groups"]):
        group_indices = torch.nonzero(groups == group, as_tuple=False)[:, 0]
        for batch_index, start in enumerate(range(0, group_indices.numel(), batch_size)):
            indices = group_indices[start:start + batch_size]
            neighbor_mean, neighbor_std = causal_neighbor_statistics(
                symbols, neighbors, weights, indices
            )
            batch_symbols = symbols[indices]
            batch_q = q_feature[indices]
            batch_topology = (
                None if topology_features is None else topology_features[indices]
            )
            for chunk in range(FEATURE_CHUNKS):
                means, scales, probabilities = causal_feature_components(
                    model,
                    causal_prior,
                    batch_symbols,
                    batch_q,
                    anchors[indices],
                    neighbor_mean,
                    neighbor_std,
                    support[indices],
                    chunk=chunk,
                    topology_features=batch_topology,
                )
                feature_slice = slice(
                    chunk * FEATURE_CHUNK_DIM,
                    (chunk + 1) * FEATURE_CHUNK_DIM,
                )
                file_name = output / (
                    f"feat_group{group}_batch{batch_index}_chunk{chunk}.b"
                )
                bit_count += encoder_gaussian_mixed_chunk(
                    batch_symbols[:, feature_slice].contiguous().view(-1),
                    [value.contiguous().view(-1) for value in means],
                    [value.contiguous().view(-1) for value in scales],
                    [value.contiguous().view(-1) for value in probabilities],
                    batch_q[:, feature_slice].contiguous().view(-1),
                    file_name=str(file_name),
                    chunk_size=500_000,
                )
    stream_bytes = sum(
        path.stat().st_size for path in output.glob("feat_group*_batch*_chunk*.b")
    )
    return {"coder_bits": bit_count, "stream_bytes": stream_bytes}


@torch.no_grad()
def decode_causal_feature_symbols(
    model,
    causal_prior,
    q_feature,
    anchors,
    graph,
    stream,
    batch_size=3000,
    topology_features=None,
):
    """Decode Feature groups without reading any same/future-group symbol."""
    from utils.encodings_cuda import decoder_gaussian_mixed_chunk

    stream = Path(stream)
    decoded = torch.zeros(
        anchors.shape[0], model.feat_dim,
        dtype=torch.float32, device=anchors.device,
    )
    neighbors = graph.neighbors.to(anchors.device)
    weights = graph.weights.to(anchors.device)
    support = graph.support.to(anchors.device)
    groups = graph.groups.to(anchors.device)
    for group in range(graph.diagnostics["num_groups"]):
        group_indices = torch.nonzero(groups == group, as_tuple=False)[:, 0]
        for batch_index, start in enumerate(range(0, group_indices.numel(), batch_size)):
            indices = group_indices[start:start + batch_size]
            neighbor_mean, neighbor_std = causal_neighbor_statistics(
                decoded, neighbors, weights, indices
            )
            batch_decoded = torch.zeros(
                indices.numel(), model.feat_dim,
                dtype=torch.float32, device=anchors.device,
            )
            batch_topology = (
                None if topology_features is None else topology_features[indices]
            )
            for chunk in range(FEATURE_CHUNKS):
                means, scales, probabilities = causal_feature_components(
                    model,
                    causal_prior,
                    batch_decoded,
                    q_feature[indices],
                    anchors[indices],
                    neighbor_mean,
                    neighbor_std,
                    support[indices],
                    chunk=chunk,
                    topology_features=batch_topology,
                )
                feature_slice = slice(
                    chunk * FEATURE_CHUNK_DIM,
                    (chunk + 1) * FEATURE_CHUNK_DIM,
                )
                file_name = stream / (
                    f"feat_group{group}_batch{batch_index}_chunk{chunk}.b"
                )
                values = decoder_gaussian_mixed_chunk(
                    [value.contiguous().view(-1) for value in means],
                    [value.contiguous().view(-1) for value in scales],
                    [value.contiguous().view(-1) for value in probabilities],
                    q_feature[indices, feature_slice].contiguous().view(-1),
                    file_name=str(file_name),
                    chunk_size=500_000,
                )
                batch_decoded[:, feature_slice] = values.view(
                    indices.numel(), FEATURE_CHUNK_DIM
                )
            decoded[indices] = batch_decoded
    return decoded


@torch.no_grad()
def encode_causal_scaling_symbols(
    model,
    causal_prior,
    symbols,
    q_scaling,
    anchors,
    graph,
    output,
    batch_size=3000,
    topology_features=None,
):
    """Encode canonical Scaling symbols using only earlier coding groups."""
    from utils.encodings_cuda import encoder_gaussian_mixed_chunk

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    neighbors = graph.neighbors.to(symbols.device)
    weights = graph.weights.to(symbols.device)
    support = graph.support.to(symbols.device)
    groups = graph.groups.to(symbols.device)
    bit_count = 0
    quantized = torch.zeros_like(symbols)
    for group in range(graph.diagnostics["num_groups"]):
        group_indices = torch.nonzero(groups == group, as_tuple=False)[:, 0]
        for batch_index, start in enumerate(
            range(0, group_indices.numel(), batch_size)
        ):
            indices = group_indices[start:start + batch_size]
            neighbor_mean, neighbor_std = causal_neighbor_statistics(
                quantized, neighbors, weights, indices
            )
            batch_topology = (
                None if topology_features is None else topology_features[indices]
            )
            means, scales, probabilities, batch_q = causal_scaling_components(
                model,
                causal_prior,
                q_scaling[indices],
                anchors[indices],
                neighbor_mean,
                neighbor_std,
                support[indices],
                topology_features=batch_topology,
            )
            batch_symbols = symbols[indices]
            file_name = output / (
                f"scaling_group{group}_batch{batch_index}.b"
            )
            bit_count += encoder_gaussian_mixed_chunk(
                batch_symbols.contiguous().view(-1),
                [value.contiguous().view(-1) for value in means],
                [value.contiguous().view(-1) for value in scales],
                [value.contiguous().view(-1) for value in probabilities],
                batch_q.contiguous().view(-1),
                file_name=str(file_name),
                chunk_size=100_000,
            )
            quantized[indices] = batch_symbols
    stream_bytes = sum(
        path.stat().st_size
        for path in output.glob("scaling_group*_batch*.b")
    )
    return {
        "coder_bits": bit_count,
        "stream_bytes": stream_bytes,
        "quantized": quantized,
    }


@torch.no_grad()
def decode_causal_scaling_symbols(
    model,
    causal_prior,
    q_scaling,
    anchors,
    graph,
    stream,
    batch_size=3000,
    topology_features=None,
):
    """Decode Scaling without reading same- or future-group symbols."""
    from utils.encodings_cuda import decoder_gaussian_mixed_chunk

    stream = Path(stream)
    decoded = torch.zeros(
        anchors.shape[0], 6, dtype=torch.float32, device=anchors.device
    )
    neighbors = graph.neighbors.to(anchors.device)
    weights = graph.weights.to(anchors.device)
    support = graph.support.to(anchors.device)
    groups = graph.groups.to(anchors.device)
    for group in range(graph.diagnostics["num_groups"]):
        group_indices = torch.nonzero(groups == group, as_tuple=False)[:, 0]
        for batch_index, start in enumerate(
            range(0, group_indices.numel(), batch_size)
        ):
            indices = group_indices[start:start + batch_size]
            neighbor_mean, neighbor_std = causal_neighbor_statistics(
                decoded, neighbors, weights, indices
            )
            batch_topology = (
                None if topology_features is None else topology_features[indices]
            )
            means, scales, probabilities, batch_q = causal_scaling_components(
                model,
                causal_prior,
                q_scaling[indices],
                anchors[indices],
                neighbor_mean,
                neighbor_std,
                support[indices],
                topology_features=batch_topology,
            )
            file_name = stream / (
                f"scaling_group{group}_batch{batch_index}.b"
            )
            values = decoder_gaussian_mixed_chunk(
                [value.contiguous().view(-1) for value in means],
                [value.contiguous().view(-1) for value in scales],
                [value.contiguous().view(-1) for value in probabilities],
                batch_q.contiguous().view(-1),
                file_name=str(file_name),
                chunk_size=100_000,
            )
            decoded[indices] = values.view(indices.numel(), 6)
    return decoded
