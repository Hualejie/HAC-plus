"""Encode a preserved pre-decode HAC++ float model with a selected CoView target."""

import argparse
import json
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from scene.coview_context import camera_geometry_from_state
from scene.gaussian_model import GaussianModel
from utils.coview_serialization import deserialize_named_tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--float_model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--coview_target",
        choices=("none", "feature", "scaling", "offset", "all"),
        required=True,
    )
    parser.add_argument(
        "--camera_context",
        help="entropy_context.pth from the trained ON stream; required for active CoView",
    )
    parser.add_argument("--feat_dim", type=int, default=50)
    parser.add_argument(
        "--coview_feature_mode", choices=("full", "chunk"), default="full"
    )
    parser.add_argument("--n_offsets", type=int, default=10)
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--view_topology_k", type=int, default=8)
    parser.add_argument("--view_topology_candidates", type=int, default=16)
    parser.add_argument(
        "--view_topology_candidate_mode",
        choices=("spatial", "hybrid"),
        default="spatial",
    )
    parser.add_argument("--view_topology_view_candidates", type=int, default=16)
    parser.add_argument("--use_causal_coview_feature", action="store_true")
    parser.add_argument("--causal_coview_groups", type=int, default=4)
    parser.add_argument("--causal_coview_max_weight", type=float, default=0.25)
    parser.add_argument("--causal_coview_gate_init", type=float, default=4.0)
    parser.add_argument(
        "--causal_prior",
        help="optional frozen causal_feature_model.bin used to initialize the formal codec",
    )
    parser.add_argument("--n_features", type=int, default=4)
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    parser.add_argument(
        "--coview_serialization", choices=("fp32", "fp16", "int8"), default="fp32"
    )
    args = parser.parse_args()

    float_path = Path(args.float_model).resolve()
    output_path = Path(args.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    model = GaussianModel(
        feat_dim=args.feat_dim,
        n_offsets=args.n_offsets,
        voxel_size=args.voxel_size,
        use_view_topology=True,
        view_topology_k=args.view_topology_k,
        view_topology_candidates=args.view_topology_candidates,
        view_topology_candidate_mode=args.view_topology_candidate_mode,
        view_topology_view_candidates=args.view_topology_view_candidates,
        coview_target=args.coview_target,
        coview_feature_mode=args.coview_feature_mode,
        use_causal_coview_feature=args.use_causal_coview_feature,
        causal_coview_groups=args.causal_coview_groups,
        causal_coview_max_weight=args.causal_coview_max_weight,
        causal_coview_gate_init=args.causal_coview_gate_init,
        n_features_per_level=args.n_features,
        log2_hashmap_size=args.log2,
        log2_hashmap_size_2D=args.log2_2D,
    )
    model.load_ply_sparse_gaussian(str(float_path / "point_cloud.ply"))
    model.load_mlp_checkpoints(
        str(float_path / "checkpoint.pth"),
        load_causal_feature_prior=not bool(args.causal_prior),
    )
    if args.causal_prior:
        if not model.causal_coview_enabled:
            raise ValueError("--causal_prior requires --use_causal_coview_feature")
        causal_state, _ = deserialize_named_tensors(args.causal_prior)
        expected = set(model.causal_coview_feature_prior.state_dict())
        if set(causal_state) != expected:
            raise RuntimeError(
                "frozen causal prior state mismatch: "
                f"missing={sorted(expected - set(causal_state))}, "
                f"extra={sorted(set(causal_state) - expected)}"
            )
        device = next(model.causal_coview_feature_prior.parameters()).device
        model.causal_coview_feature_prior.load_state_dict({
            name: tensor.to(device) for name, tensor in causal_state.items()
        })
    model.x_bound_min = torch.load(float_path / "x_bound_min.pkl")
    model.x_bound_max = torch.load(float_path / "x_bound_max.pkl")
    if model.entropy_extension_enabled:
        if not args.camera_context:
            raise ValueError("--camera_context is required for an active CoView target")
        context = torch.load(args.camera_context)
        model._view_topology_cameras = camera_geometry_from_state(
            context["camera_geometry"]
        )
    model.eval()
    log = model.conduct_encoding(
        str(output_path), coview_serialization=args.coview_serialization
    )
    print(json.dumps({
        "coview_target": args.coview_target,
        "use_causal_coview_feature": args.use_causal_coview_feature,
        "coview_serialization": args.coview_serialization,
        "encode_log": log,
    }))


if __name__ == "__main__":
    main()
