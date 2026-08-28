"""Optimize the decoder-causal Scaling prior on a fixed HAC++ representation."""

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from analysis.frozen_feature_codec import CODEC_BATCH_SIZE, tensor_checksum
from analysis.train_frozen_feature_entropy import build_model
from scene.coview_causal_context import (
    AffineCausalScalingPrior,
    build_causal_anchor_graph,
    causal_neighbor_statistics,
    causal_scaling_components,
)
from scene.coview_context import (
    build_view_topology_context,
    camera_geometry_from_state,
)
from utils.coview_serialization import (
    deserialize_named_tensors,
    serialize_named_tensors,
)
from utils.encodings import STE_multistep
from utils.gpcc_utils import calculate_morton_order


@torch.no_grad()
def capture_fixed_scaling(model):
    valid = model.get_mask_anchor.to(torch.bool)[:, 0]
    valid_original_idx = torch.nonzero(valid, as_tuple=False)[:, 0]
    anchor_int = torch.round(model.get_anchor[valid] / model.voxel_size)
    order = calculate_morton_order(anchor_int)
    anchors = anchor_int[order] * model.voxel_size
    raw_scaling = model.get_scaling[valid][order]
    q_scaling = torch.cat([
        model.scaling_quantization_steps(anchors[start:start + CODEC_BATCH_SIZE])
        for start in range(0, anchors.shape[0], CODEC_BATCH_SIZE)
    ])
    symbols = STE_multistep.apply(
        raw_scaling, q_scaling, model.get_scaling.mean()
    ).detach()
    return {
        "anchors": anchors.detach(),
        "symbols": symbols,
        "q_scaling": q_scaling.detach(),
        "valid_original_idx": valid_original_idx.detach(),
        "codec_original_idx": valid_original_idx[order].detach(),
        "anchor_checksum": tensor_checksum(anchors),
        "symbol_checksum": tensor_checksum(symbols),
        "symbol_index_checksum": tensor_checksum(
            torch.round(symbols / q_scaling).to(torch.int32)
        ),
        "q_checksum": tensor_checksum(q_scaling),
    }


def _base_parameters(model, anchors):
    outputs = torch.split(
        model.get_grid_mlp(model.calc_interp_feat(anchors)),
        [
            model.feat_dim, model.feat_dim, model.feat_dim,
            6, 6, 3 * model.n_offsets, 3 * model.n_offsets,
            1, 1, 1,
        ],
        dim=-1,
    )
    return outputs[3], torch.clamp(outputs[4], min=1e-9)


def base_rate_bits(model, fixed, indices):
    mean, scale = _base_parameters(model, fixed["anchors"][indices])
    return model.entropy_gaussian.forward(
        fixed["symbols"][indices],
        mean,
        scale,
        fixed["q_scaling"][indices],
        model.get_scaling.mean(),
    ).sum()


def causal_rate_bits(model, prior, fixed, graph_tensors, indices):
    neighbors, weights, support = graph_tensors
    neighbor_mean, neighbor_std = causal_neighbor_statistics(
        fixed["symbols"], neighbors, weights, indices
    )
    means, scales, probabilities, q_scaling = causal_scaling_components(
        model,
        prior,
        fixed["q_scaling"][indices],
        fixed["anchors"][indices],
        neighbor_mean,
        neighbor_std,
        support[indices],
    )
    return model.EG_mix_prob_2.forward(
        fixed["symbols"][indices],
        means[0], means[1],
        scales[0], scales[1],
        probabilities[0], probabilities[1],
        Q=q_scaling,
        x_mean=model.get_scaling.mean(),
    ).sum()


def expert_rate_bits(model, prior, fixed, graph_tensors, indices):
    neighbors, weights, support = graph_tensors
    neighbor_mean, neighbor_std = causal_neighbor_statistics(
        fixed["symbols"], neighbors, weights, indices
    )
    mean, scale = _base_parameters(model, fixed["anchors"][indices])
    causal_mean, causal_scale, _ = prior(
        mean,
        scale,
        fixed["q_scaling"][indices],
        neighbor_mean,
        neighbor_std,
        support[indices],
    )
    bits = model.entropy_gaussian.forward(
        fixed["symbols"][indices],
        causal_mean,
        causal_scale,
        fixed["q_scaling"][indices],
        model.get_scaling.mean(),
    )
    supported = (support[indices] > 0).to(bits.dtype)
    return (bits * supported).sum(), supported.sum() * bits.shape[1]


def _batched_bps(rate_function, indices, batch_size):
    total_bits = 0.0
    total_symbols = 0
    with torch.no_grad():
        for start in range(0, indices.numel(), batch_size):
            batch = indices[start:start + batch_size]
            value = rate_function(batch)
            if isinstance(value, tuple):
                bits, count = value
                total_symbols += int(count.item())
            else:
                bits = value
                total_symbols += batch.numel() * 6
            total_bits += float(bits.detach().cpu())
    return total_bits / max(total_symbols, 1)


def optimize_prior(model, prior, fixed, graph, args):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = fixed["anchors"].device
    graph_tensors = (
        graph.neighbors.to(device),
        graph.weights.to(device),
        graph.support.to(device),
    )
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(fixed["anchors"].shape[0]).astype(np.int64)
    validation_size = min(args.validation_size, permutation.size // 4)
    validation = torch.as_tensor(permutation[:validation_size], device=device)
    train_pool = permutation[validation_size:]
    supported_pool = train_pool[
        graph.support.numpy()[train_pool, 0] > 0
    ]

    def expert_validation(batch):
        return expert_rate_bits(model, prior, fixed, graph_tensors, batch)

    def mixture_validation(batch):
        return causal_rate_bits(model, prior, fixed, graph_tensors, batch)

    started = time.time()
    pretrain_optimizer = torch.optim.Adam(
        (prior.mean_blend, prior.log_scale_blend),
        lr=args.pretrain_lr,
        eps=1e-15,
    )
    best_expert_bps = _batched_bps(
        expert_validation, validation, args.batch_size
    )
    best_pretrain_step = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in prior.state_dict().items()
    }
    prior.train()
    for step in range(args.pretrain_steps):
        indices = torch.as_tensor(
            rng.choice(supported_pool, args.batch_size, replace=True),
            device=device,
        )
        bits, count = expert_rate_bits(
            model, prior, fixed, graph_tensors, indices
        )
        loss = bits / torch.clamp(count, min=1.0)
        pretrain_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        pretrain_optimizer.step()
        if step == 0 or (step + 1) % args.log_interval == 0:
            prior.eval()
            current = _batched_bps(
                expert_validation, validation, args.batch_size
            )
            if current < best_expert_bps:
                best_expert_bps = current
                best_pretrain_step = step + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in prior.state_dict().items()
                }
            print(json.dumps({
                "stage": "expert_pretrain",
                "step": step + 1,
                "loss_bps": float(loss.detach().cpu()),
                "validation_bps": current,
                "best_validation_bps": best_expert_bps,
            }), flush=True)
            prior.train()

    prior.load_state_dict({
        name: value.to(device) for name, value in best_state.items()
    })
    with torch.no_grad():
        prior.gate_logit.fill_(args.fusion_gate_init)

    optimizer = torch.optim.Adam(prior.parameters(), lr=args.lr, eps=1e-15)
    prior.eval()
    initial_mixture_bps = _batched_bps(
        mixture_validation, validation, args.batch_size
    )
    best_mixture_bps = initial_mixture_bps
    best_step = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in prior.state_dict().items()
    }
    prior.train()
    for step in range(args.steps):
        indices = torch.as_tensor(
            rng.choice(train_pool, args.batch_size, replace=True),
            device=device,
        )
        loss = causal_rate_bits(
            model, prior, fixed, graph_tensors, indices
        ) / indices.numel() / 6
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % args.log_interval == 0:
            prior.eval()
            current = _batched_bps(
                mixture_validation, validation, args.batch_size
            )
            if current < best_mixture_bps:
                best_mixture_bps = current
                best_step = step + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in prior.state_dict().items()
                }
            print(json.dumps({
                "stage": "mixture_finetune",
                "step": step + 1,
                "loss_bps": float(loss.detach().cpu()),
                "validation_bps": current,
                "best_validation_bps": best_mixture_bps,
            }), flush=True)
            prior.train()

    prior.load_state_dict({
        name: value.to(device) for name, value in best_state.items()
    })
    prior.eval()
    all_indices = torch.arange(fixed["anchors"].shape[0], device=device)
    base_bps = _batched_bps(
        lambda batch: base_rate_bits(model, fixed, batch),
        all_indices,
        args.batch_size,
    )
    causal_bps = _batched_bps(
        mixture_validation, all_indices, args.batch_size
    )
    return {
        "training_seconds": time.time() - started,
        "best_pretrain_step": best_pretrain_step,
        "best_expert_validation_bps": best_expert_bps,
        "initial_mixture_validation_bps": initial_mixture_bps,
        "best_mixture_step": best_step,
        "best_mixture_validation_bps": best_mixture_bps,
        "base_full_bps": base_bps,
        "causal_full_bps": causal_bps,
        "estimated_delta_bytes": (
            (causal_bps - base_bps) * fixed["anchors"].shape[0] * 6 / 8
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--float_model", required=True)
    parser.add_argument("--camera_context", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--serialization", choices=("fp32", "fp16", "int8"), default="fp16")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--pretrain_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--pretrain_lr", type=float, default=0.003)
    parser.add_argument("--fusion_gate_init", type=float, default=-2.0)
    parser.add_argument("--validation_size", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--causal_groups", type=int, default=4)
    parser.add_argument("--causal_max_mixture_weight", type=float, default=0.25)
    parser.add_argument("--feat_dim", type=int, default=50)
    parser.add_argument("--n_offsets", type=int, default=10)
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--view_topology_k", type=int, default=8)
    parser.add_argument("--view_topology_candidates", type=int, default=32)
    parser.add_argument("--view_topology_candidate_mode", choices=("spatial", "hybrid"), default="spatial")
    parser.add_argument("--view_topology_view_candidates", type=int, default=32)
    parser.add_argument("--feature_mode", choices=("full", "chunk"), default="full")
    parser.add_argument("--n_features", type=int, default=4)
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    parser.add_argument("--allow_topology_override", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model = build_model(args, "none")
    model.eval()
    fixed = capture_fixed_scaling(model)
    camera_context = torch.load(args.camera_context, map_location="cpu")
    cameras = camera_geometry_from_state(camera_context["camera_geometry"])
    topology = build_view_topology_context(
        fixed["anchors"],
        cameras,
        candidate_k=args.view_topology_candidates,
        topk=args.view_topology_k,
        candidate_mode=args.view_topology_candidate_mode,
        view_candidate_k=args.view_topology_view_candidates,
    )
    graph = build_causal_anchor_graph(topology, num_groups=args.causal_groups)
    prior = AffineCausalScalingPrior(
        max_mixture_weight=args.causal_max_mixture_weight
    ).cuda()
    metrics = optimize_prior(model, prior, fixed, graph, args)

    model_path = output / "causal_scaling_model.bin"
    metadata = serialize_named_tensors(
        prior.state_dict(), model_path, args.serialization
    )
    decoded, decoded_metadata = deserialize_named_tensors(model_path)
    if decoded_metadata != metadata:
        raise RuntimeError("causal Scaling model serialization metadata mismatch")
    prior.load_state_dict({name: value.cuda() for name, value in decoded.items()})
    result = {
        "num_anchors": int(fixed["anchors"].shape[0]),
        "serialization": args.serialization,
        "model_bytes": metadata["bytes"],
        "anchor_checksum": fixed["anchor_checksum"],
        "symbol_checksum": fixed["symbol_checksum"],
        "symbol_index_checksum": fixed["symbol_index_checksum"],
        "q_checksum": fixed["q_checksum"],
        "graph_diagnostics": graph.diagnostics,
        "topology_diagnostics": topology.diagnostics,
        **metrics,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
