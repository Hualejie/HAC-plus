"""Render and evaluate a standalone HAC++ entropy package."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import lpips
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams
from gaussian_renderer import prefilter_voxel, render
from scene import GaussianModel, Scene
from utils.image_utils import psnr
from utils.loss_utils import ssim


def main():
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--bitstream", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_features", type=int, default=4)
    parser.add_argument("--log2", type=int, default=13)
    parser.add_argument("--log2_2D", type=int, default=15)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = model_params.extract(args)
    dataset.model_path = str(output)
    pipeline = pipeline_params.extract(args)
    is_synthetic_nerf = os.path.exists(
        os.path.join(dataset.source_path, "transforms_train.json")
    )
    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        use_view_topology=dataset.use_view_topology,
        view_topology_k=dataset.view_topology_k,
        view_topology_candidates=dataset.view_topology_candidates,
        view_topology_candidate_mode=dataset.view_topology_candidate_mode,
        view_topology_view_candidates=dataset.view_topology_view_candidates,
        coview_target=dataset.coview_target,
        coview_feature_mode=dataset.coview_feature_mode,
        use_causal_coview_feature=dataset.use_causal_coview_feature,
        use_causal_coview_scaling=dataset.use_causal_coview_scaling,
        causal_coview_groups=dataset.causal_coview_groups,
        causal_coview_candidates=dataset.causal_coview_candidates,
        causal_coview_max_weight=dataset.causal_coview_max_weight,
        causal_coview_gate_init=dataset.causal_coview_gate_init,
        causal_coview_camera_count=dataset.causal_coview_camera_count,
        n_features_per_level=args.n_features,
        log2_hashmap_size=args.log2,
        log2_hashmap_size_2D=args.log2_2D,
        decoded_version=True,
        is_synthetic_nerf=is_synthetic_nerf,
    )
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.eval()
    decode_log = gaussians.conduct_decoding(str(Path(args.bitstream).resolve()))

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    lpips_fn = lpips.LPIPS(net="vgg").to("cuda").eval()
    per_view = {}
    render_times = []
    with torch.no_grad():
        for viewpoint in tqdm(scene.getTestCameras(), desc="Evaluating bitstream"):
            torch.cuda.synchronize()
            start = time.time()
            visible = prefilter_voxel(viewpoint, gaussians, pipeline, background)
            render_output = render(
                viewpoint, gaussians, pipeline, background, visible_mask=visible
            )
            torch.cuda.synchronize()
            render_times.append(time.time() - start - render_output["time_sub"])
            image = torch.clamp(render_output["render"], 0.0, 1.0)
            ground_truth = torch.clamp(
                viewpoint.original_image[:3].to("cuda"), 0.0, 1.0
            )
            per_view[viewpoint.image_name] = {
                "PSNR": float(psnr(image, ground_truth).mean()),
                "SSIM": float(ssim(image, ground_truth).mean()),
                "LPIPS": float(
                    lpips_fn(image, ground_truth, normalize=False).mean()
                ),
            }

    aggregate = {
        metric: sum(values[metric] for values in per_view.values()) / len(per_view)
        for metric in ("PSNR", "SSIM", "LPIPS")
    }
    aggregate.update({
        "FPS": 1.0 / (sum(render_times) / len(render_times)),
        "num_test_views": len(per_view),
        "num_anchors": int(gaussians._anchor.shape[0]),
        "decode_log": decode_log,
    })
    (output / "results.json").write_text(
        json.dumps({"ours_30000": aggregate}, indent=2), encoding="utf-8"
    )
    (output / "per_view.json").write_text(
        json.dumps(per_view, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
