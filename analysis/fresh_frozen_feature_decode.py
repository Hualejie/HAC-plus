"""Fresh-process decoder for Phase 2C fixed-symbol Feature streams."""

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from analysis.frozen_feature_codec import decode_feature_symbols, tensor_checksum
from scene.coview_context import camera_geometry_from_state
from scene.gaussian_model import GaussianModel
from utils.coview_serialization import deserialize_named_tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    args = parser.parse_args()
    package_path = Path(args.package).resolve()
    context = torch.load(package_path / "frozen_feature_context.pth", map_location="cpu")
    config = context["config"]
    model = GaussianModel(
        feat_dim=config["feat_dim"],
        n_offsets=config["n_offsets"],
        voxel_size=config["voxel_size"],
        use_view_topology=True,
        view_topology_k=config["view_topology_k"],
        view_topology_candidates=config["view_topology_candidates"],
        coview_target=config["coview_target"],
        coview_feature_mode=config["coview_feature_mode"],
        n_features_per_level=config["n_features"],
        log2_hashmap_size=config["log2"],
        log2_hashmap_size_2D=config["log2_2D"],
        decoded_version=True,
    )
    model.encoding_xyz.load_state_dict(context["encoding_xyz"])
    model.mlp_grid.load_state_dict(context["grid_mlp"])
    model.mlp_deform.load_state_dict(context["deform_mlp"])
    model.x_bound_min = context["x_bound_min"].cuda()
    model.x_bound_max = context["x_bound_max"].cuda()
    anchors = context["anchors"].cuda()
    q_feature = context["q_feature"].cuda()
    topology = None
    coview_metadata = None
    if model.coview_enabled:
        blob_path = package_path / context["coview_model_file"]
        coview_state, coview_metadata = deserialize_named_tensors(blob_path)
        if coview_metadata != context["coview_model_metadata"]:
            raise RuntimeError("CoView model blob metadata/checksum mismatch")
        model.install_coview_serializable_state(coview_state)
        model._view_topology_cameras = camera_geometry_from_state(
            context["camera_geometry"]
        )
        topology, diagnostics = model.codec_view_topology(
            anchors, force_rebuild=True
        )
        if tensor_checksum(topology) != context["topology_checksum"]:
            raise RuntimeError("fresh decoder topology checksum mismatch")
    model.eval()
    decoded = decode_feature_symbols(
        model, q_feature, anchors, package_path / "stream", topology
    )
    checksum = tensor_checksum(decoded)
    symbol_index_checksum = tensor_checksum(
        torch.round(decoded / q_feature).to(torch.int32)
    )
    if symbol_index_checksum != context["symbol_index_checksum"]:
        raise RuntimeError(
            "fresh Feature symbol-index checksum mismatch: "
            f"{symbol_index_checksum} != {context['symbol_index_checksum']}"
        )
    result = {
        "label": context["label"],
        "num_anchors": int(anchors.shape[0]),
        "symbol_checksum": checksum,
        "reference_float_symbol_checksum": context["symbol_checksum"],
        "float_symbol_checksum_match": checksum == context["symbol_checksum"],
        "symbol_index_checksum": symbol_index_checksum,
        "anchor_checksum": tensor_checksum(anchors),
        "q_checksum": tensor_checksum(q_feature),
        "coview_model_metadata": coview_metadata,
        "fresh_decode": True,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
