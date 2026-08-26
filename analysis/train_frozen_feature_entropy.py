"""Phase 2C fixed-representation Feature entropy experiments.

The anchor representation, hash, Feature symbols, and Feature quantization step
are captured once from the Control 30k model and then reused by every branch.
"""

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

from analysis.frozen_feature_codec import (
    CODEC_BATCH_SIZE,
    encode_feature_symbols,
    feature_parameters,
    feature_rate_bits,
    tensor_checksum,
)
from scene.coview_context import camera_geometry_from_state
from scene.gaussian_model import GaussianModel
from utils.coview_serialization import (
    deserialize_named_tensors,
    serialize_named_tensors,
)
from utils.encodings import STE_multistep
from utils.gpcc_utils import calculate_morton_order


def build_model(args, target):
    model = GaussianModel(
        feat_dim=args.feat_dim,
        n_offsets=args.n_offsets,
        voxel_size=args.voxel_size,
        use_view_topology=True,
        view_topology_k=args.view_topology_k,
        view_topology_candidates=args.view_topology_candidates,
        coview_target=target,
        coview_feature_mode=args.feature_mode,
        n_features_per_level=args.n_features,
        log2_hashmap_size=args.log2,
        log2_hashmap_size_2D=args.log2_2D,
    )
    float_model = Path(args.float_model)
    model.load_ply_sparse_gaussian(str(float_model / "point_cloud.ply"))
    model.load_mlp_checkpoints(
        str(float_model / "checkpoint.pth"),
        load_coview_feature_head=args.feature_mode == "full",
    )
    model.x_bound_min = torch.load(float_model / "x_bound_min.pkl")
    model.x_bound_max = torch.load(float_model / "x_bound_max.pkl")
    camera_context = torch.load(args.camera_context)
    model._view_topology_cameras = camera_geometry_from_state(
        camera_context["camera_geometry"]
    )
    return model


def common_entropy_checksum(model):
    digest = hashlib.sha256()
    for module in (model.encoding_xyz, model.mlp_grid, model.mlp_deform):
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def capture_fixed_representation(model):
    valid_mask = model.get_mask_anchor.to(torch.bool)[:, 0]
    valid_original_idx = torch.nonzero(valid_mask, as_tuple=False)[:, 0]
    anchor = model.get_anchor[valid_mask]
    anchor_int = torch.round(anchor / model.voxel_size)
    codec_order = calculate_morton_order(anchor_int)
    codec_original_idx = valid_original_idx[codec_order]
    anchors = anchor_int[codec_order] * model.voxel_size
    raw_features = model._anchor_feat[valid_mask][codec_order]

    q_batches = []
    for start in range(0, anchors.shape[0], CODEC_BATCH_SIZE):
        _, _, _, q_adjustment = feature_parameters(
            model, anchors[start:start + CODEC_BATCH_SIZE]
        )
        q_batches.append(
            (1.0 + torch.tanh(q_adjustment)).repeat(1, model.feat_dim)
        )
    q_feature = torch.cat(q_batches, dim=0).detach()
    symbols = STE_multistep.apply(
        raw_features, q_feature, model._anchor_feat.mean()
    ).detach()
    return {
        "anchors": anchors.detach(),
        "symbols": symbols,
        "q_feature": q_feature,
        "valid_original_idx": valid_original_idx.detach(),
        "codec_original_idx": codec_original_idx.detach(),
        "hash_checksum": tensor_checksum(model.get_encoding_params()),
        "anchor_checksum": tensor_checksum(anchors),
        "symbol_checksum": tensor_checksum(symbols),
        "q_checksum": tensor_checksum(q_feature),
    }


@torch.no_grad()
def estimated_bits(model, fixed, topology):
    total = 0.0
    for start in range(0, fixed["anchors"].shape[0], CODEC_BATCH_SIZE):
        end = min(start + CODEC_BATCH_SIZE, fixed["anchors"].shape[0])
        batch_topology = None if topology is None else topology[start:end]
        total += float(feature_rate_bits(
            model,
            fixed["symbols"][start:end],
            fixed["q_feature"][start:end],
            fixed["anchors"][start:end],
            batch_topology,
        ).cpu())
    return total


def set_trainable(model, include_base, include_coview):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups = []
    if include_base:
        base_parameters = (
            list(model.mlp_grid.parameters()) + list(model.mlp_deform.parameters())
        )
        for parameter in base_parameters:
            parameter.requires_grad_(True)
        groups.append(("base", base_parameters))
    if include_coview:
        coview_parameters = (
            list(model.mlp_coview_shared.parameters())
            + list(model.mlp_coview_feature.parameters())
            + [model.coview_gates["feature"]]
        )
        for parameter in coview_parameters:
            parameter.requires_grad_(True)
        groups.append(("coview", coview_parameters))
    return groups


def optimize_entropy(model, fixed, topology, schedule, args, include_base, include_coview):
    parameter_groups = set_trainable(model, include_base, include_coview)
    optimizer_groups = []
    for name, parameters in parameter_groups:
        optimizer_groups.append({
            "params": parameters,
            "lr": args.base_lr if name == "base" else args.coview_lr,
        })
    optimizer = torch.optim.Adam(optimizer_groups, eps=1e-15)
    model.train()
    start_time = time.time()
    losses = []
    for step, cpu_indices in enumerate(schedule):
        indices = torch.as_tensor(cpu_indices, device="cuda", dtype=torch.long)
        batch_topology = None if topology is None else topology[indices]
        bits = feature_rate_bits(
            model,
            fixed["symbols"][indices],
            fixed["q_feature"][indices],
            fixed["anchors"][indices],
            batch_topology,
        )
        loss = bits / indices.numel() / model.feat_dim
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % args.log_interval == 0 or step == 0:
            print(json.dumps({
                "step": step + 1,
                "steps": len(schedule),
                "bits_per_symbol": losses[-1],
                "include_base": include_base,
                "include_coview": include_coview,
            }))
    model.eval()
    return {
        "training_seconds": time.time() - start_time,
        "initial_sample_bps": losses[0],
        "final_sample_bps": losses[-1],
    }


def save_and_encode(label, model, fixed, topology, args, training_metrics):
    run_path = Path(args.output) / label
    stream_path = run_path / "stream"
    run_path.mkdir(parents=True, exist_ok=True)
    coview_metadata = None
    if model.coview_enabled:
        blob_path = run_path / "coview_model.bin"
        coview_metadata = serialize_named_tensors(
            model.coview_serializable_state(), blob_path, args.serialization
        )
        decoded_state, decoded_metadata = deserialize_named_tensors(blob_path)
        if decoded_metadata != coview_metadata:
            raise RuntimeError("CoView blob round-trip metadata mismatch")
        model.install_coview_serializable_state(decoded_state)

    model.eval()
    estimate = estimated_bits(model, fixed, topology if model.coview_enabled else None)
    codec = encode_feature_symbols(
        model,
        fixed["symbols"],
        fixed["q_feature"],
        fixed["anchors"],
        stream_path,
        topology if model.coview_enabled else None,
    )
    camera_context = torch.load(args.camera_context, map_location="cpu")
    package = {
        "version": 1,
        "label": label,
        "config": {
            "feat_dim": args.feat_dim,
            "n_offsets": args.n_offsets,
            "voxel_size": args.voxel_size,
            "view_topology_k": args.view_topology_k,
            "view_topology_candidates": args.view_topology_candidates,
            "n_features": args.n_features,
            "log2": args.log2,
            "log2_2D": args.log2_2D,
            "coview_target": model.coview_target,
            "coview_feature_mode": args.feature_mode,
        },
        "anchors": fixed["anchors"].detach().cpu(),
        "q_feature": fixed["q_feature"].detach().cpu(),
        "encoding_xyz": model.encoding_xyz.state_dict(),
        "grid_mlp": model.mlp_grid.state_dict(),
        "deform_mlp": model.mlp_deform.state_dict(),
        "x_bound_min": model.x_bound_min.detach().cpu(),
        "x_bound_max": model.x_bound_max.detach().cpu(),
        "camera_geometry": camera_context["camera_geometry"],
        "topology_checksum": tensor_checksum(topology),
        "anchor_checksum": fixed["anchor_checksum"],
        "symbol_checksum": fixed["symbol_checksum"],
        "q_checksum": fixed["q_checksum"],
        "coview_model_file": "coview_model.bin" if model.coview_enabled else None,
        "coview_model_metadata": coview_metadata,
    }
    torch.save(package, run_path / "frozen_feature_context.pth")
    result = {
        "label": label,
        "feature_mode": args.feature_mode,
        "serialization": args.serialization if model.coview_enabled else None,
        "num_anchors": int(fixed["anchors"].shape[0]),
        "estimated_bits": estimate,
        "feature_stream_bytes": codec["stream_bytes"],
        "feature_coder_bits": codec["coder_bits"],
        "coview_model_bytes": 0 if coview_metadata is None else coview_metadata["bytes"],
        "feature_plus_model_bytes": codec["stream_bytes"] + (
            0 if coview_metadata is None else coview_metadata["bytes"]
        ),
        "symbol_checksum": fixed["symbol_checksum"],
        "q_checksum": fixed["q_checksum"],
        **training_metrics,
    }
    (run_path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--float_model", required=True)
    parser.add_argument("--camera_context", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiments", default="a1,a2")
    parser.add_argument("--feature_mode", choices=("full", "chunk"), default="chunk")
    parser.add_argument("--serialization", choices=("fp32", "fp16", "int8"), default="fp32")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=3000)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--coview_lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--feat_dim", type=int, default=50)
    parser.add_argument("--n_offsets", type=int, default=10)
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--view_topology_k", type=int, default=8)
    parser.add_argument("--view_topology_candidates", type=int, default=16)
    parser.add_argument("--n_features", type=int, default=4)
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    source = build_model(args, "none")
    source.eval()
    fixed = capture_fixed_representation(source)
    topology_model = build_model(args, "feature")
    topology, topology_diagnostics = topology_model.codec_view_topology(
        fixed["anchors"], force_rebuild=True
    )
    if topology_diagnostics["feature_checksum"] != tensor_checksum(topology):
        raise RuntimeError("topology checksum implementation mismatch")

    rng = np.random.default_rng(args.seed)
    schedule = rng.integers(
        0, fixed["anchors"].shape[0],
        size=(args.steps, args.batch_size), dtype=np.int32,
    )
    schedule_checksum = hashlib.sha256(schedule.tobytes()).hexdigest()
    experiments = {value.strip() for value in args.experiments.split(",") if value.strip()}
    if not experiments <= {"a1", "a2"}:
        raise ValueError(f"unknown experiments {sorted(experiments - {'a1', 'a2'})}")

    results = {}
    initial_entropy_checksums = {}
    if "a1" in experiments:
        base = build_model(args, "none")
        coview = build_model(args, "feature")
        initial_entropy_checksums["a1_base"] = common_entropy_checksum(base)
        initial_entropy_checksums["a1_coview"] = common_entropy_checksum(coview)
        if initial_entropy_checksums["a1_base"] != initial_entropy_checksums["a1_coview"]:
            raise RuntimeError("A1 branches did not start from the same base predictor")
        results["a1_base"] = save_and_encode(
            "a1_base", base, fixed, topology, args,
            {"training_seconds": 0.0},
        )
        metrics = optimize_entropy(
            coview, fixed, topology, schedule, args,
            include_base=False, include_coview=True,
        )
        results["a1_coview"] = save_and_encode(
            "a1_coview", coview, fixed, topology, args, metrics
        )

    if "a2" in experiments:
        control = build_model(args, "none")
        coview = build_model(args, "feature")
        initial_entropy_checksums["a2_control"] = common_entropy_checksum(control)
        initial_entropy_checksums["a2_coview"] = common_entropy_checksum(coview)
        if initial_entropy_checksums["a2_control"] != initial_entropy_checksums["a2_coview"]:
            raise RuntimeError("A2 branches did not start from the same entropy checkpoint")
        control_metrics = optimize_entropy(
            control, fixed, None, schedule, args,
            include_base=True, include_coview=False,
        )
        results["a2_control"] = save_and_encode(
            "a2_control", control, fixed, topology, args, control_metrics
        )
        coview_metrics = optimize_entropy(
            coview, fixed, topology, schedule, args,
            include_base=True, include_coview=True,
        )
        results["a2_coview"] = save_and_encode(
            "a2_coview", coview, fixed, topology, args, coview_metrics
        )

    manifest = {
        "feature_mode": args.feature_mode,
        "serialization": args.serialization,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "schedule_checksum": schedule_checksum,
        "initial_entropy_checksums": initial_entropy_checksums,
        "fixed_representation": {
            key: value for key, value in fixed.items()
            if key.endswith("checksum")
        },
        "valid_original_idx_checksum": tensor_checksum(fixed["valid_original_idx"]),
        "codec_original_idx_checksum": tensor_checksum(fixed["codec_original_idx"]),
        "topology_diagnostics": topology_diagnostics,
        "results": results,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
