"""Frozen Feature rate and codec helpers for causal CoView experiments."""

from pathlib import Path

import torch

from analysis.frozen_feature_codec import (
    CODEC_BATCH_SIZE,
    FEATURE_CHUNK_DIM,
    FEATURE_CHUNKS,
    feature_parameters,
)
from scene.coview_causal_context import causal_neighbor_statistics
from utils.encodings_cuda import (
    decoder_gaussian_mixed_chunk,
    encoder_gaussian_mixed_chunk,
)
from utils.entropy_models import Entropy_gaussian_mix_prob_3
from utils.entropy_models import Entropy_gaussian


def mixture_moments(means, scales, probabilities):
    """Moment-match a Gaussian mixture for stable causal-expert initialization."""
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
):
    """Build the two HAC++ components plus one causal-neighbour component."""
    mean, scale, probability, _ = feature_parameters(model, anchors)
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
        torch.stack((probability[:, feature_slice], probability_adjusted), dim=-1),
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
    causal_weight = torch.clamp(
        causal_weight, min=0.0, max=1.0 - 1e-6
    )
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


def causal_feature_rate_bits(
    model,
    causal_prior,
    symbols,
    q_feature,
    anchors,
    neighbor_mean,
    neighbor_std,
    support,
):
    means, scales, probabilities = causal_feature_components(
        model,
        causal_prior,
        symbols,
        q_feature,
        anchors,
        neighbor_mean,
        neighbor_std,
        support,
    )
    entropy_model = Entropy_gaussian_mix_prob_3()
    return entropy_model.forward(
        symbols,
        means[0], means[1], means[2],
        scales[0], scales[1], scales[2],
        probabilities[0], probabilities[1], probabilities[2],
        Q=q_feature,
    ).sum()


def causal_expert_rate_bits(
    model,
    causal_prior,
    symbols,
    q_feature,
    anchors,
    neighbor_mean,
    neighbor_std,
    support,
):
    """Standalone causal-expert objective used before mixture fine-tuning."""
    mean, scale, probability, _ = feature_parameters(model, anchors)
    mean_adjusted, scale_adjusted, probability_adjusted = (
        model.get_deform_mlp.forward(
            symbols, torch.cat((mean, scale, probability), dim=-1)
        )
    )
    base_probabilities = torch.softmax(
        torch.stack((probability, probability_adjusted), dim=-1), dim=-1
    )
    mixture_mean, mixture_scale = mixture_moments(
        (mean, mean_adjusted),
        (torch.clamp(scale, min=1e-9), torch.clamp(scale_adjusted, min=1e-9)),
        (base_probabilities[..., 0], base_probabilities[..., 1]),
    )
    causal_mean, causal_scale, _ = causal_prior(
        mixture_mean,
        mixture_scale,
        q_feature,
        neighbor_mean,
        neighbor_std,
        support,
    )
    bits = Entropy_gaussian().forward(
        symbols,
        causal_mean,
        causal_scale,
        Q=q_feature,
    )
    valid = (support > 0).to(bits.dtype).expand_as(bits)
    return (bits * valid).sum(), valid.sum()


@torch.no_grad()
def encode_causal_feature_symbols(
    model,
    causal_prior,
    symbols,
    q_feature,
    anchors,
    graph,
    output,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    neighbors = graph.neighbors.to(symbols.device)
    weights = graph.weights.to(symbols.device)
    support = graph.support.to(symbols.device)
    bit_count = 0
    for group in range(graph.diagnostics["num_groups"]):
        group_indices = torch.nonzero(
            graph.groups == group, as_tuple=False
        )[:, 0].to(symbols.device)
        for batch_index, start in enumerate(
            range(0, group_indices.numel(), CODEC_BATCH_SIZE)
        ):
            indices = group_indices[start:start + CODEC_BATCH_SIZE]
            neighbor_mean, neighbor_std = causal_neighbor_statistics(
                symbols, neighbors, weights, indices
            )
            batch_symbols = symbols[indices]
            batch_q = q_feature[indices]
            batch_anchors = anchors[indices]
            for chunk in range(FEATURE_CHUNKS):
                means, scales, probabilities = causal_feature_components(
                    model,
                    causal_prior,
                    batch_symbols,
                    batch_q,
                    batch_anchors,
                    neighbor_mean,
                    neighbor_std,
                    support[indices],
                    chunk=chunk,
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
    real_bytes = sum(path.stat().st_size for path in output.glob("feat_*.b"))
    return {"coder_bits": bit_count, "stream_bytes": real_bytes}


@torch.no_grad()
def decode_causal_feature_symbols(
    model,
    causal_prior,
    q_feature,
    anchors,
    graph,
    stream,
):
    stream = Path(stream)
    decoded = torch.zeros(
        anchors.shape[0], model.feat_dim, dtype=torch.float32, device=anchors.device
    )
    neighbors = graph.neighbors.to(anchors.device)
    weights = graph.weights.to(anchors.device)
    support = graph.support.to(anchors.device)
    for group in range(graph.diagnostics["num_groups"]):
        group_indices = torch.nonzero(
            graph.groups == group, as_tuple=False
        )[:, 0].to(anchors.device)
        for batch_index, start in enumerate(
            range(0, group_indices.numel(), CODEC_BATCH_SIZE)
        ):
            indices = group_indices[start:start + CODEC_BATCH_SIZE]
            neighbor_mean, neighbor_std = causal_neighbor_statistics(
                decoded, neighbors, weights, indices
            )
            batch_decoded = torch.zeros(
                indices.numel(), model.feat_dim,
                dtype=torch.float32, device=anchors.device,
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
