"""Shared fixed-symbol Feature rate and arithmetic-coding helpers."""

from pathlib import Path
import hashlib

import torch

from utils.encodings_cuda import (
    decoder_gaussian_mixed_chunk,
    encoder_gaussian_mixed_chunk,
)


FEATURE_CHUNKS = 5
FEATURE_CHUNK_DIM = 10
CODEC_BATCH_SIZE = 3000


def tensor_checksum(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def feature_parameters(model, anchors, topology_features=None):
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
    return mean, torch.clamp(scale, min=1e-9), probability, outputs[7]


def feature_rate_bits(model, symbols, q_feature, anchors, topology_features=None):
    mean, scale, probability, _ = feature_parameters(
        model, anchors, topology_features
    )
    mean_adjusted, scale_adjusted, probability_adjusted = (
        model.get_deform_mlp.forward(
            symbols, torch.cat([mean, scale, probability], dim=-1)
        )
    )
    probabilities = torch.softmax(
        torch.stack([probability, probability_adjusted], dim=-1), dim=-1
    )
    bits = model.EG_mix_prob_2.forward(
        symbols,
        mean, mean_adjusted,
        scale, scale_adjusted,
        probabilities[..., 0], probabilities[..., 1],
        Q=q_feature,
    )
    return bits.sum()


@torch.no_grad()
def encode_feature_symbols(model, symbols, q_feature, anchors, output, topology=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bit_count = 0
    for batch_index, start in enumerate(range(0, anchors.shape[0], CODEC_BATCH_SIZE)):
        end = min(start + CODEC_BATCH_SIZE, anchors.shape[0])
        batch_topology = None if topology is None else topology[start:end]
        mean, scale, probability, _ = feature_parameters(
            model, anchors[start:end], batch_topology
        )
        batch_symbols = symbols[start:end]
        batch_q = q_feature[start:end]
        mean_scale = torch.cat([mean, scale, probability], dim=-1)
        for chunk in range(FEATURE_CHUNKS):
            mean_adjusted, scale_adjusted, probability_adjusted = (
                model.get_deform_mlp.forward(
                    batch_symbols, mean_scale, to_dec=chunk
                )
            )
            chunk_slice = slice(
                chunk * FEATURE_CHUNK_DIM, (chunk + 1) * FEATURE_CHUNK_DIM
            )
            probabilities = torch.softmax(
                torch.stack([
                    probability[:, chunk_slice], probability_adjusted,
                ], dim=-1),
                dim=-1,
            )
            base_name = output / f"feat_{batch_index}_{chunk}.b"
            bit_count += encoder_gaussian_mixed_chunk(
                batch_symbols[:, chunk_slice].contiguous().view(-1),
                [
                    mean[:, chunk_slice].contiguous().view(-1),
                    mean_adjusted.contiguous().view(-1),
                ],
                [
                    scale[:, chunk_slice].contiguous().view(-1),
                    scale_adjusted.contiguous().view(-1),
                ],
                [
                    probabilities[..., 0].contiguous().view(-1),
                    probabilities[..., 1].contiguous().view(-1),
                ],
                batch_q[:, chunk_slice].contiguous().view(-1),
                file_name=str(base_name),
                chunk_size=500_000,
            )
    real_bytes = sum(path.stat().st_size for path in output.glob("feat_*.b"))
    return {"coder_bits": bit_count, "stream_bytes": real_bytes}


@torch.no_grad()
def decode_feature_symbols(model, q_feature, anchors, stream, topology=None):
    decoded_batches = []
    stream = Path(stream)
    for batch_index, start in enumerate(range(0, anchors.shape[0], CODEC_BATCH_SIZE)):
        end = min(start + CODEC_BATCH_SIZE, anchors.shape[0])
        count = end - start
        batch_topology = None if topology is None else topology[start:end]
        mean, scale, probability, _ = feature_parameters(
            model, anchors[start:end], batch_topology
        )
        decoded = torch.zeros(
            count, model.feat_dim, dtype=torch.float32, device=anchors.device
        )
        mean_scale = torch.cat([mean, scale, probability], dim=-1)
        for chunk in range(FEATURE_CHUNKS):
            mean_adjusted, scale_adjusted, probability_adjusted = (
                model.get_deform_mlp.forward(decoded, mean_scale, to_dec=chunk)
            )
            chunk_slice = slice(
                chunk * FEATURE_CHUNK_DIM, (chunk + 1) * FEATURE_CHUNK_DIM
            )
            probabilities = torch.softmax(
                torch.stack([
                    probability[:, chunk_slice], probability_adjusted,
                ], dim=-1),
                dim=-1,
            )
            base_name = stream / f"feat_{batch_index}_{chunk}.b"
            values = decoder_gaussian_mixed_chunk(
                [
                    mean[:, chunk_slice].contiguous().view(-1),
                    mean_adjusted.contiguous().view(-1),
                ],
                [
                    scale[:, chunk_slice].contiguous().view(-1),
                    scale_adjusted.contiguous().view(-1),
                ],
                [
                    probabilities[..., 0].contiguous().view(-1),
                    probabilities[..., 1].contiguous().view(-1),
                ],
                q_feature[start:end, chunk_slice].contiguous().view(-1),
                file_name=str(base_name),
                chunk_size=500_000,
            )
            decoded[:, chunk_slice] = values.view(count, FEATURE_CHUNK_DIM)
        decoded_batches.append(decoded)
    return torch.cat(decoded_batches, dim=0)
