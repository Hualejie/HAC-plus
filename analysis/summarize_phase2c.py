"""Aggregate Phase 2C frozen-Feature and Scaling replication outputs."""

import argparse
import csv
import json
from pathlib import Path

import torch


STREAM_PATTERNS = {
    "anchor_bytes": "xyz_gpcc.npz",
    "feature_bytes": "feat*.b",
    "scaling_bytes": "scaling*.b",
    "offset_bytes": "offsets*.b",
    "hash_bytes": "hash.b",
    "mask_bytes": "masks.b",
}


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stream_bytes(path):
    result = {}
    for name, pattern in STREAM_PATTERNS.items():
        files = list(path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"missing {pattern} in {path}")
        result[name] = sum(file.stat().st_size for file in files)
    return result


def tensor_bytes(state):
    return sum(tensor.numel() * tensor.element_size() for tensor in state.values())


def base_mlp_bytes(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    keys = ["opacity_mlp", "cov_mlp", "color_mlp", "grid_mlp", "deform_mlp"]
    if "mlp_feature_bank" in checkpoint:
        keys.append("mlp_feature_bank")
    return sum(tensor_bytes(checkpoint[key]) for key in keys)


def last_json_line(path):
    if not path.exists():
        return None
    for line in reversed(path.read_text(errors="replace").splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"no JSON result in {path}")


def frozen_rows(root):
    configurations = [
        ("full", "fp32", root / "playroom_full_fp32"),
        ("chunk", "fp32", root / "playroom_chunk_fp32"),
        ("chunk", "fp16", root / "playroom_chunk_fp16"),
        ("chunk", "int8", root / "playroom_chunk_int8"),
    ]
    rows = []
    ablation = []
    serialization = []
    for mode, storage, path in configurations:
        manifest = json.loads((path / "manifest.json").read_text())
        for label, result in manifest["results"].items():
            if storage != "fp32" and label == "a1_base":
                continue
            experiment, variant = label.split("_", 1)
            control = manifest["results"][f"{experiment}_base" if experiment == "a1" else "a2_control"]
            raw_saving = control["feature_stream_bytes"] - result["feature_stream_bytes"]
            net_saving = control["feature_stream_bytes"] - result["feature_plus_model_bytes"]
            row = {
                "scene": "playroom",
                "experiment": experiment,
                "variant": variant,
                "feature_mode": mode,
                "serialization": result["serialization"] or "none",
                "num_anchors": result["num_anchors"],
                "feature_stream_bytes": result["feature_stream_bytes"],
                "coview_model_bytes": result["coview_model_bytes"],
                "feature_plus_model_bytes": result["feature_plus_model_bytes"],
                "raw_feature_bytes_saved": raw_saving,
                "net_feature_bytes_saved": net_saving,
                "estimated_bits": result["estimated_bits"],
                "training_seconds": result["training_seconds"],
                "symbol_index_checksum": result["symbol_index_checksum"],
            }
            rows.append(row)
            if variant == "coview" and experiment == "a1":
                ablation.append(dict(row))
                serialization.append({
                    "scene": "playroom",
                    "target": "feature",
                    "architecture": f"32->{100 if mode == 'full' else 10}",
                    "serialization": storage,
                    "model_bytes": result["coview_model_bytes"],
                    "attribute_stream_bytes": result["feature_stream_bytes"],
                    "attribute_plus_model_bytes": result["feature_plus_model_bytes"],
                    "net_bytes_saved": net_saving,
                    "fresh_decode": True,
                })
    return rows, ablation, serialization


def trained_scaling_row(scene, target, run, fresh_path=None):
    stream = stream_bytes(run / "bitstreams")
    checkpoint = run / "float_model" / "iteration_30000" / "checkpoint.pth"
    base_bytes = base_mlp_bytes(checkpoint)
    model_path = run / "bitstreams" / "coview_model.bin"
    model_bytes = model_path.stat().st_size if model_path.exists() else 0
    total = sum(stream.values()) + base_bytes + model_bytes + 24
    metrics = json.loads((run / "results.json").read_text())["ours_30000"]
    fresh = last_json_line(fresh_path or (run / "fresh_decode.json"))
    return {
        "scene": scene,
        "target": target,
        **stream,
        "base_mlp_bytes": base_bytes,
        "coview_model_bytes": model_bytes,
        "total_bytes": total,
        "PSNR": metrics["PSNR"],
        "SSIM": metrics["SSIM"],
        "LPIPS": metrics["LPIPS"],
        "fresh_decode": bool(fresh),
        "scaling_checksum": None if fresh is None else fresh["scaling_checksum"],
    }


def scaling_outputs(root, phase2b):
    playroom_control = phase2b["runs"]["none"]
    rows = [{
        "scene": "playroom",
        "target": "control",
        "serialization": "none",
        **{key: playroom_control[key] for key in (
            "anchor_bytes", "feature_bytes", "scaling_bytes", "offset_bytes",
            "hash_bytes", "mask_bytes", "base_mlp_bytes", "total_bytes",
            "PSNR", "SSIM", "LPIPS",
        )},
        "coview_model_bytes": 0,
        "fresh_decode": bool(playroom_control["fresh_decode"]),
        "scaling_checksum": playroom_control["fresh_decode"]["scaling_checksum"],
    }]
    serialization_rows = []
    for storage in ("fp32", "fp16", "int8"):
        run = root / f"scaling_playroom_{storage}"
        stream = stream_bytes(run)
        model_bytes = (run / "coview_model.bin").stat().st_size
        base_bytes = playroom_control["base_mlp_bytes"]
        total = sum(stream.values()) + base_bytes + model_bytes + 24
        fresh = last_json_line(root / f"scaling_playroom_{storage}_fresh.json")
        row = {
            "scene": "playroom",
            "target": "scaling",
            "serialization": storage,
            **stream,
            "base_mlp_bytes": base_bytes,
            "coview_model_bytes": model_bytes,
            "total_bytes": total,
            "PSNR": phase2b["runs"]["scaling"]["PSNR"],
            "SSIM": phase2b["runs"]["scaling"]["SSIM"],
            "LPIPS": phase2b["runs"]["scaling"]["LPIPS"],
            "fresh_decode": bool(fresh),
            "scaling_checksum": None if fresh is None else fresh["scaling_checksum"],
        }
        rows.append(row)
        serialization_rows.append({
            "scene": "playroom",
            "target": "scaling",
            "architecture": "15->32->12",
            "serialization": storage,
            "model_bytes": model_bytes,
            "attribute_stream_bytes": stream["scaling_bytes"],
            "attribute_plus_model_bytes": stream["scaling_bytes"] + model_bytes,
            "net_bytes_saved": playroom_control["total_bytes"] - total,
            "fresh_decode": bool(fresh),
        })

    control_path = root / "drjohnson_control"
    scaling_path = root / "drjohnson_scaling"
    if control_path.exists() and scaling_path.exists() and (control_path / "results.json").exists():
        rows.extend([
            {"serialization": "none", **trained_scaling_row("drjohnson", "control", control_path)},
            {"serialization": "fp32", **trained_scaling_row("drjohnson", "scaling", scaling_path)},
        ])

    for scene in {row["scene"] for row in rows}:
        control = next(row for row in rows if row["scene"] == scene and row["target"] == "control")
        for row in rows:
            if row["scene"] != scene:
                continue
            row["total_bytes_saved_vs_control"] = control["total_bytes"] - row["total_bytes"]
            row["PSNR_delta_vs_control"] = row["PSNR"] - control["PSNR"]
            row["SSIM_delta_vs_control"] = row["SSIM"] - control["SSIM"]
            row["LPIPS_delta_vs_control"] = row["LPIPS"] - control["LPIPS"]
    return rows, serialization_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--phase2b_summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    phase2b = json.loads(Path(args.phase2b_summary).read_text())

    frozen, ablation, feature_serialization = frozen_rows(root)
    scaling, scaling_serialization = scaling_outputs(root, phase2b)
    write_csv(output / "frozen_feature_rate.csv", frozen)
    write_csv(output / "feature_head_ablation.csv", ablation)
    write_csv(output / "scaling_replication.csv", scaling)
    write_csv(
        output / "model_serialization.csv",
        feature_serialization + scaling_serialization,
    )
    (output / "summary.json").write_text(json.dumps({
        "frozen_feature": frozen,
        "feature_head_ablation": ablation,
        "scaling_replication": scaling,
        "model_serialization": feature_serialization + scaling_serialization,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
