"""Benchmark view-topology candidate configurations on codec-aligned anchors."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from plyfile import PlyData
import torch

try:
    import resource
except ImportError:  # pragma: no cover - benchmark runs on Linux servers
    resource = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from scene.coview_context import (
    build_view_topology_context,
    camera_geometry_from_state,
    canonicalize_codec_anchors,
)


def load_codec_anchors(point_cloud, voxel_size):
    vertex = PlyData.read(point_cloud).elements[0]
    anchor = torch.from_numpy(np.stack((
        np.asarray(vertex["x"]),
        np.asarray(vertex["y"]),
        np.asarray(vertex["z"]),
    ), axis=1).astype(np.float32))
    mask_names = sorted(
        (prop.name for prop in vertex.properties if prop.name.startswith("f_mask")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if not mask_names:
        raise RuntimeError("point cloud contains no f_mask fields")
    raw_mask = np.stack(
        [np.asarray(vertex[name]) for name in mask_names], axis=1
    ).astype(np.float32)
    active_mask = torch.sigmoid(torch.from_numpy(raw_mask[:, :10])) > 0.01
    mask_anchor = active_mask.any(dim=1)
    return canonicalize_codec_anchors(anchor, mask_anchor, voxel_size)


def parse_configuration(value):
    parts = value.split(":")
    mode = parts[0]
    if mode == "spatial" and len(parts) == 2:
        return {"mode": mode, "spatial_k": int(parts[1]), "view_k": 0}
    if mode == "hybrid" and len(parts) == 3:
        return {
            "mode": mode,
            "spatial_k": int(parts[1]),
            "view_k": int(parts[2]),
        }
    raise argparse.ArgumentTypeError(
        "configuration must be spatial:K or hybrid:SPATIAL_K:VIEW_K"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point_cloud", required=True)
    parser.add_argument("--camera_context", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--config",
        type=parse_configuration,
        action="append",
        dest="configurations",
        help="Repeatable: spatial:K or hybrid:SPATIAL_K:VIEW_K",
    )
    args = parser.parse_args()
    configurations = args.configurations or [
        parse_configuration(value) for value in (
            "spatial:16", "spatial:32", "spatial:64", "spatial:128",
            "hybrid:32:32",
        )
    ]

    aligned = load_codec_anchors(args.point_cloud, args.voxel_size)
    context = torch.load(args.camera_context, map_location="cpu")
    cameras = camera_geometry_from_state(context["camera_geometry"])
    results = []
    for configuration in configurations:
        started = time.perf_counter()
        topology = build_view_topology_context(
            aligned.xyz,
            cameras,
            candidate_k=configuration["spatial_k"],
            topk=args.topk,
            candidate_mode=configuration["mode"],
            view_candidate_k=max(configuration["view_k"], 1),
        )
        result = {
            **configuration,
            "elapsed_seconds": time.perf_counter() - started,
            "max_rss_kib": (
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                if resource is not None else None
            ),
            **topology.diagnostics,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "point_cloud": str(Path(args.point_cloud).resolve()),
        "camera_context": str(Path(args.camera_context).resolve()),
        "voxel_size": args.voxel_size,
        "topk": args.topk,
        "num_valid_anchors": int(aligned.xyz.shape[0]),
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
