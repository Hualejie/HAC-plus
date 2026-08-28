"""Train and code a fixed Feature tensor with causal CoView neighbours."""

import argparse
import hashlib
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

from analysis.frozen_causal_feature_codec import (
    causal_expert_rate_bits,
    causal_feature_rate_bits,
    encode_causal_feature_symbols,
)
from analysis.frozen_feature_codec import (
    CODEC_BATCH_SIZE,
    encode_feature_symbols,
    feature_rate_bits,
    tensor_checksum,
)
from analysis.train_frozen_feature_entropy import (
    build_model,
    capture_fixed_representation,
)
from scene.coview_causal_context import (
    AffineCausalFeaturePrior,
    CausalFeaturePrior,
    build_causal_anchor_graph,
    causal_neighbor_statistics,
)
from scene.coview_context import (
    build_view_topology_context,
    camera_geometry_from_state,
)
from utils.coview_serialization import (
    deserialize_named_tensors,
    serialize_named_tensors,
)


def graph_checksum(graph):
    digest = hashlib.sha256()
    for tensor in (graph.groups, graph.neighbors, graph.weights, graph.support):
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def estimated_base_bits(model, fixed):
    total = 0.0
    for start in range(0, fixed["anchors"].shape[0], CODEC_BATCH_SIZE):
        end = min(start + CODEC_BATCH_SIZE, fixed["anchors"].shape[0])
        total += float(feature_rate_bits(
            model,
            fixed["symbols"][start:end],
            fixed["q_feature"][start:end],
            fixed["anchors"][start:end],
        ).cpu())
    return total


@torch.no_grad()
def estimated_causal_bits(model, prior, fixed, graph, batch_size=CODEC_BATCH_SIZE):
    neighbors = graph.neighbors.to(fixed["symbols"].device)
    weights = graph.weights.to(fixed["symbols"].device)
    support = graph.support.to(fixed["symbols"].device)
    total = 0.0
    for start in range(0, fixed["anchors"].shape[0], batch_size):
        end = min(start + batch_size, fixed["anchors"].shape[0])
        indices = torch.arange(start, end, device=fixed["anchors"].device)
        neighbor_mean, neighbor_std = causal_neighbor_statistics(
            fixed["symbols"], neighbors, weights, indices
        )
        total += float(causal_feature_rate_bits(
            model,
            prior,
            fixed["symbols"][indices],
            fixed["q_feature"][indices],
            fixed["anchors"][indices],
            neighbor_mean,
            neighbor_std,
            support[indices],
        ).cpu())
    return total


def optimize_prior(model, prior, fixed, graph, pretrain_schedule, schedule, args):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    neighbors = graph.neighbors.to(fixed["symbols"].device)
    weights = graph.weights.to(fixed["symbols"].device)
    support = graph.support.to(fixed["symbols"].device)
    losses = []
    started = time.time()
    validation_indices = torch.as_tensor(
        args.validation_indices, device="cuda", dtype=torch.long
    )

    @torch.no_grad()
    def validation_bps():
        values = []
        for start in range(0, validation_indices.numel(), args.batch_size):
            indices = validation_indices[start:start + args.batch_size]
            neighbor_mean, neighbor_std = causal_neighbor_statistics(
                fixed["symbols"], neighbors, weights, indices
            )
            bits = causal_feature_rate_bits(
                model,
                prior,
                fixed["symbols"][indices],
                fixed["q_feature"][indices],
                fixed["anchors"][indices],
                neighbor_mean,
                neighbor_std,
                support[indices],
            )
            values.append(float(bits.cpu()))
        return sum(values) / validation_indices.numel() / model.feat_dim

    @torch.no_grad()
    def expert_validation_bps():
        total_bits = 0.0
        total_symbols = 0.0
        for start in range(0, validation_indices.numel(), args.batch_size):
            indices = validation_indices[start:start + args.batch_size]
            neighbor_mean, neighbor_std = causal_neighbor_statistics(
                fixed["symbols"], neighbors, weights, indices
            )
            bits, symbol_count = causal_expert_rate_bits(
                model,
                prior,
                fixed["symbols"][indices],
                fixed["q_feature"][indices],
                fixed["anchors"][indices],
                neighbor_mean,
                neighbor_std,
                support[indices],
            )
            total_bits += float(bits.cpu())
            total_symbols += float(symbol_count.cpu())
        return total_bits / max(total_symbols, 1.0)

    prior.train()
    pretrain_optimizer = torch.optim.Adam(
        prior.parameters(), lr=args.pretrain_lr, eps=1e-15
    )
    pretrain_losses = []
    prior.eval()
    initial_expert_validation_bps = expert_validation_bps()
    best_expert_validation_bps = initial_expert_validation_bps
    best_pretrain_step = 0
    best_expert_state = {
        name: value.detach().cpu().clone()
        for name, value in prior.state_dict().items()
    }
    prior.train()
    for step, cpu_indices in enumerate(pretrain_schedule):
        indices = torch.as_tensor(cpu_indices, device="cuda", dtype=torch.long)
        with torch.no_grad():
            neighbor_mean, neighbor_std = causal_neighbor_statistics(
                fixed["symbols"], neighbors, weights, indices
            )
        bits, symbol_count = causal_expert_rate_bits(
            model,
            prior,
            fixed["symbols"][indices],
            fixed["q_feature"][indices],
            fixed["anchors"][indices],
            neighbor_mean,
            neighbor_std,
            support[indices],
        )
        loss = bits / torch.clamp(symbol_count, min=1.0)
        pretrain_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), max_norm=1.0)
        pretrain_optimizer.step()
        pretrain_losses.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % args.log_interval == 0:
            prior.eval()
            current_expert_validation_bps = expert_validation_bps()
            if current_expert_validation_bps < best_expert_validation_bps:
                best_expert_validation_bps = current_expert_validation_bps
                best_pretrain_step = step + 1
                best_expert_state = {
                    name: value.detach().cpu().clone()
                    for name, value in prior.state_dict().items()
                }
            print(json.dumps({
                "stage": "causal_expert_pretrain",
                "step": step + 1,
                "steps": len(pretrain_schedule),
                "bits_per_supported_symbol": pretrain_losses[-1],
                "validation_bits_per_supported_symbol": current_expert_validation_bps,
                "best_validation_bits_per_supported_symbol": best_expert_validation_bps,
            }), flush=True)
            prior.train()

    prior.load_state_dict({
        name: value.to(next(prior.parameters()).device)
        for name, value in best_expert_state.items()
    })
    with torch.no_grad():
        if isinstance(prior, AffineCausalFeaturePrior):
            prior.gate_logit.fill_(args.fusion_gate_init)
        else:
            prior.network[-1].weight[-1].zero_()
            prior.network[-1].bias[-1].fill_(args.fusion_gate_init)

    optimizer = torch.optim.Adam(prior.parameters(), lr=args.lr, eps=1e-15)
    prior.eval()
    initial_validation_bps = validation_bps()
    best_validation_bps = initial_validation_bps
    best_step = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in prior.state_dict().items()
    }
    prior.train()
    for step, cpu_indices in enumerate(schedule):
        indices = torch.as_tensor(cpu_indices, device="cuda", dtype=torch.long)
        with torch.no_grad():
            neighbor_mean, neighbor_std = causal_neighbor_statistics(
                fixed["symbols"], neighbors, weights, indices
            )
        bits = causal_feature_rate_bits(
            model,
            prior,
            fixed["symbols"][indices],
            fixed["q_feature"][indices],
            fixed["anchors"][indices],
            neighbor_mean,
            neighbor_std,
            support[indices],
        )
        loss = bits / indices.numel() / model.feat_dim
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % args.log_interval == 0:
            prior.eval()
            current_validation_bps = validation_bps()
            if current_validation_bps < best_validation_bps:
                best_validation_bps = current_validation_bps
                best_step = step + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in prior.state_dict().items()
                }
            print(json.dumps({
                "stage": "mixture_finetune",
                "step": step + 1,
                "steps": len(schedule),
                "bits_per_symbol": losses[-1],
                "validation_bits_per_symbol": current_validation_bps,
                "best_validation_bits_per_symbol": best_validation_bps,
            }), flush=True)
            prior.train()
    prior.load_state_dict({
        name: value.to(next(prior.parameters()).device)
        for name, value in best_state.items()
    })
    prior.eval()
    return {
        "training_seconds": time.time() - started,
        "initial_pretrain_bps": pretrain_losses[0],
        "final_pretrain_bps": pretrain_losses[-1],
        "initial_expert_validation_bps": initial_expert_validation_bps,
        "best_expert_validation_bps": best_expert_validation_bps,
        "best_pretrain_step": best_pretrain_step,
        "initial_sample_bps": losses[0],
        "final_sample_bps": losses[-1],
        "initial_validation_bps": initial_validation_bps,
        "best_validation_bps": best_validation_bps,
        "best_step": best_step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--float_model", required=True)
    parser.add_argument("--camera_context", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--serialization", choices=("fp32", "fp16", "int8"), default="fp16")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrain_lr", type=float, default=1e-4)
    parser.add_argument("--pretrain_steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--causal_groups", type=int, default=4)
    parser.add_argument("--causal_hidden_dim", type=int, default=16)
    parser.add_argument(
        "--causal_prior_type", choices=("mlp", "affine"), default="mlp"
    )
    parser.add_argument("--causal_max_mixture_weight", type=float, default=0.25)
    parser.add_argument("--fusion_gate_init", type=float, default=-2.0)
    parser.add_argument("--validation_size", type=int, default=12000)
    parser.add_argument("--feat_dim", type=int, default=50)
    parser.add_argument("--n_offsets", type=int, default=10)
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--view_topology_k", type=int, default=8)
    parser.add_argument("--view_topology_candidates", type=int, default=32)
    parser.add_argument(
        "--view_topology_candidate_mode", choices=("spatial", "hybrid"),
        default="spatial",
    )
    parser.add_argument("--view_topology_view_candidates", type=int, default=32)
    parser.add_argument("--feature_mode", choices=("full", "chunk"), default="chunk")
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
    fixed = capture_fixed_representation(model)
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

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(fixed["anchors"].shape[0]).astype(np.int32)
    validation_size = min(args.validation_size, permutation.size // 4)
    args.validation_indices = permutation[:validation_size]
    train_pool = permutation[validation_size:]
    pretrain_schedule = train_pool[rng.integers(
        0, train_pool.size,
        size=(args.pretrain_steps, args.batch_size), dtype=np.int32,
    )]
    schedule = train_pool[rng.integers(
        0, train_pool.size,
        size=(args.steps, args.batch_size), dtype=np.int32,
    )]
    if args.causal_prior_type == "mlp":
        prior = CausalFeaturePrior(
            args.causal_hidden_dim,
            max_mixture_weight=args.causal_max_mixture_weight,
        ).cuda()
    else:
        prior = AffineCausalFeaturePrior(
            max_mixture_weight=args.causal_max_mixture_weight,
        ).cuda()
    training_metrics = optimize_prior(
        model, prior, fixed, graph, pretrain_schedule, schedule, args
    )

    model_path = output / "causal_feature_model.bin"
    model_metadata = serialize_named_tensors(
        prior.state_dict(), model_path, args.serialization
    )
    decoded_state, decoded_metadata = deserialize_named_tensors(model_path)
    if decoded_metadata != model_metadata:
        raise RuntimeError("causal model serialization metadata mismatch")
    prior.load_state_dict({name: value.cuda() for name, value in decoded_state.items()})
    prior.eval()

    base_estimate = estimated_base_bits(model, fixed)
    causal_estimate = estimated_causal_bits(model, prior, fixed, graph)
    base_codec = encode_feature_symbols(
        model,
        fixed["symbols"],
        fixed["q_feature"],
        fixed["anchors"],
        output / "base_stream",
    )
    causal_codec = encode_causal_feature_symbols(
        model,
        prior,
        fixed["symbols"],
        fixed["q_feature"],
        fixed["anchors"],
        graph,
        output / "causal_stream",
    )

    package = {
        "version": 1,
        "config": {
            "feat_dim": args.feat_dim,
            "n_offsets": args.n_offsets,
            "voxel_size": args.voxel_size,
            "view_topology_k": args.view_topology_k,
            "view_topology_candidates": args.view_topology_candidates,
            "view_topology_candidate_mode": args.view_topology_candidate_mode,
            "view_topology_view_candidates": args.view_topology_view_candidates,
            "causal_groups": args.causal_groups,
            "causal_hidden_dim": args.causal_hidden_dim,
            "causal_prior_type": args.causal_prior_type,
            "causal_max_mixture_weight": args.causal_max_mixture_weight,
            "fusion_gate_init": args.fusion_gate_init,
            "n_features": args.n_features,
            "log2": args.log2,
            "log2_2D": args.log2_2D,
        },
        "anchors": fixed["anchors"].detach().cpu(),
        "q_feature": fixed["q_feature"].detach().cpu(),
        "encoding_xyz": model.encoding_xyz.state_dict(),
        "grid_mlp": model.mlp_grid.state_dict(),
        "deform_mlp": model.mlp_deform.state_dict(),
        "x_bound_min": model.x_bound_min.detach().cpu(),
        "x_bound_max": model.x_bound_max.detach().cpu(),
        "camera_geometry": camera_context["camera_geometry"],
        "graph_checksum": graph_checksum(graph),
        "symbol_checksum": fixed["symbol_checksum"],
        "symbol_index_checksum": fixed["symbol_index_checksum"],
        "q_checksum": fixed["q_checksum"],
        "causal_model_file": model_path.name,
        "causal_model_metadata": model_metadata,
    }
    torch.save(package, output / "frozen_causal_feature_context.pth")
    result = {
        "num_anchors": int(fixed["anchors"].shape[0]),
        "serialization": args.serialization,
        "base_estimated_bits": base_estimate,
        "causal_estimated_bits": causal_estimate,
        "base_stream_bytes": base_codec["stream_bytes"],
        "causal_stream_bytes": causal_codec["stream_bytes"],
        "causal_model_bytes": model_metadata["bytes"],
        "causal_plus_model_bytes": causal_codec["stream_bytes"] + model_metadata["bytes"],
        "conditional_delta_bytes": causal_codec["stream_bytes"] - base_codec["stream_bytes"],
        "net_delta_bytes": (
            causal_codec["stream_bytes"] + model_metadata["bytes"]
            - base_codec["stream_bytes"]
        ),
        "graph_diagnostics": graph.diagnostics,
        "topology_diagnostics": topology.diagnostics,
        "graph_checksum": graph_checksum(graph),
        **training_metrics,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
