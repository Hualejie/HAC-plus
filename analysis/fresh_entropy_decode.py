"""Decode a HAC++ attribute bitstream in a fresh process.

This intentionally constructs no Scene and loads no training checkpoint.  The
entropy package must therefore contain every network needed by attribute coding.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from scene.gaussian_model import GaussianModel


def _checksum(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitstream", required=True)
    parser.add_argument(
        "--coview_target",
        choices=("none", "feature", "scaling", "offset", "all"),
        required=True,
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
    parser.add_argument("--causal_coview_candidates", type=int, default=32)
    parser.add_argument("--causal_coview_max_weight", type=float, default=0.25)
    parser.add_argument("--n_features", type=int, default=4)
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    args = parser.parse_args()

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
        causal_coview_candidates=args.causal_coview_candidates,
        causal_coview_max_weight=args.causal_coview_max_weight,
        n_features_per_level=args.n_features,
        log2_hashmap_size=args.log2,
        log2_hashmap_size_2D=args.log2_2D,
        decoded_version=True,
    )
    model.eval()
    log = model.conduct_decoding(os.path.abspath(args.bitstream))
    feature_symbol_index_checksum = None
    if model.causal_coview_enabled:
        q_feature = torch.cat([
            model.feature_quantization_steps(
                model._anchor[start:start + 3000]
            )
            for start in range(0, model._anchor.shape[0], 3000)
        ], dim=0)
        feature_symbol_index_checksum = _checksum(
            torch.round(model._anchor_feat / q_feature).to(torch.int32)
        )
    result = {
        "coview_target": args.coview_target,
        "use_causal_coview_feature": args.use_causal_coview_feature,
        "num_anchors": int(model._anchor.shape[0]),
        "anchor_checksum": _checksum(model._anchor),
        "feature_checksum": _checksum(model._anchor_feat),
        "feature_symbol_index_checksum": feature_symbol_index_checksum,
        "scaling_checksum": _checksum(model._scaling),
        "offset_checksum": _checksum(model._offset),
        "mask_checksum": _checksum(model._mask),
        "decode_log": log,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
