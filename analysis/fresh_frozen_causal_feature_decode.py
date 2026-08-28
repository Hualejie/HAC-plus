"""Fresh-process decoder for the causal Frozen Feature package."""

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from analysis.frozen_causal_feature_codec import decode_causal_feature_symbols
from analysis.frozen_feature_codec import tensor_checksum
from analysis.train_frozen_causal_feature_entropy import graph_checksum
from scene.coview_causal_context import CausalFeaturePrior, build_causal_anchor_graph
from scene.coview_context import (
    build_view_topology_context,
    camera_geometry_from_state,
)
from scene.gaussian_model import GaussianModel
from utils.coview_serialization import deserialize_named_tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    args = parser.parse_args()
    package_path = Path(args.package).resolve()
    context = torch.load(
        package_path / "frozen_causal_feature_context.pth", map_location="cpu"
    )
    config = context["config"]
    model = GaussianModel(
        feat_dim=config["feat_dim"],
        n_offsets=config["n_offsets"],
        voxel_size=config["voxel_size"],
        use_view_topology=False,
        coview_target="none",
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
    model.eval()

    anchors = context["anchors"].cuda()
    q_feature = context["q_feature"].cuda()
    cameras = camera_geometry_from_state(context["camera_geometry"])
    topology = build_view_topology_context(
        anchors,
        cameras,
        candidate_k=config["view_topology_candidates"],
        topk=config["view_topology_k"],
        candidate_mode=config["view_topology_candidate_mode"],
        view_candidate_k=config["view_topology_view_candidates"],
    )
    graph = build_causal_anchor_graph(
        topology, num_groups=config["causal_groups"]
    )
    if graph_checksum(graph) != context["graph_checksum"]:
        raise RuntimeError("fresh decoder causal graph checksum mismatch")

    prior = CausalFeaturePrior(config["causal_hidden_dim"]).cuda()
    prior_state, prior_metadata = deserialize_named_tensors(
        package_path / context["causal_model_file"]
    )
    if prior_metadata != context["causal_model_metadata"]:
        raise RuntimeError("fresh decoder causal model metadata mismatch")
    prior.load_state_dict({name: value.cuda() for name, value in prior_state.items()})
    prior.eval()

    decoded = decode_causal_feature_symbols(
        model,
        prior,
        q_feature,
        anchors,
        graph,
        package_path / "causal_stream",
    )
    symbol_index_checksum = tensor_checksum(
        torch.round(decoded / q_feature).to(torch.int32)
    )
    if symbol_index_checksum != context["symbol_index_checksum"]:
        raise RuntimeError("fresh causal Feature symbol-index checksum mismatch")
    result = {
        "fresh_decode": True,
        "num_anchors": int(anchors.shape[0]),
        "symbol_index_checksum": symbol_index_checksum,
        "float_symbol_checksum_match": (
            tensor_checksum(decoded) == context["symbol_checksum"]
        ),
        "graph_checksum": context["graph_checksum"],
        "causal_model_metadata": prior_metadata,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
