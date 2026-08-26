"""Aggregate Phase 2D multi-lambda RD and Frozen Feature experiments."""

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.rd_metrics import bd_rate


STREAM_PATTERNS = {
    "anchor_bytes": "xyz_gpcc.npz",
    "feature_bytes": "feat*.b",
    "scaling_bytes": "scaling*.b",
    "offset_bytes": "offsets*.b",
    "hash_bytes": "hash.b",
    "mask_bytes": "masks.b",
}
NUMERIC_FIELDS = {
    "anchor_bytes", "feature_bytes", "scaling_bytes", "offset_bytes",
    "hash_bytes", "mask_bytes", "base_mlp_bytes", "coview_model_bytes",
    "total_bytes", "PSNR", "SSIM", "LPIPS",
}


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def last_json_line(path):
    if path is None or not path.exists():
        return None
    for line in reversed(path.read_text(errors="replace").splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return None


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


def trained_row(scene, lmbda, target, run, bitstream, serialization, fresh_path):
    streams = stream_bytes(bitstream)
    checkpoint = run / "float_model" / "iteration_30000" / "checkpoint.pth"
    base_bytes = base_mlp_bytes(checkpoint)
    model_path = bitstream / "coview_model.bin"
    model_bytes = model_path.stat().st_size if model_path.exists() else 0
    metrics = json.loads((run / "results.json").read_text())["ours_30000"]
    fresh = last_json_line(fresh_path)
    return {
        "scene": scene,
        "lambda": lmbda,
        "target": target,
        "serialization": serialization,
        **streams,
        "base_mlp_bytes": base_bytes,
        "coview_model_bytes": model_bytes,
        "total_bytes": sum(streams.values()) + base_bytes + model_bytes + 24,
        "PSNR": metrics["PSNR"],
        "SSIM": metrics["SSIM"],
        "LPIPS": metrics["LPIPS"],
        "fresh_decode": bool(fresh and fresh.get("num_anchors")),
        "representation_checksum": None if fresh is None else fresh.get("scaling_checksum"),
    }


def phase2d_rd_rows(root):
    config = json.loads((root / "experiment_config.json").read_text())
    rows = []
    for scene in config["scenes"]:
        for lmbda in config["lambdas"]:
            point = root / scene / f"lambda_{format(lmbda, '.8f').rstrip('0').rstrip('.')}"
            control = point / "control"
            scaling = point / "scaling"
            rows.extend([
                trained_row(
                    scene, lmbda, "control", control, control / "bitstreams",
                    "none", control / "fresh_decode.json",
                ),
                trained_row(
                    scene, lmbda, "scaling", scaling,
                    scaling / "bitstreams_fp16", "fp16",
                    scaling / "fresh_decode_fp16.json",
                ),
            ])
    return rows


def legacy_rows(csv_path, dr_fp16, dr_fresh):
    source = list(csv.DictReader(csv_path.open()))
    selected = []
    for scene, target, storage in (
        ("playroom", "control", "none"),
        ("playroom", "scaling", "fp16"),
        ("drjohnson", "control", "none"),
    ):
        row = next(
            row for row in source
            if row["scene"] == scene
            and row["target"] == target
            and row["serialization"] == storage
        )
        converted = {key: value for key, value in row.items() if key in {
            "scene", "target", "serialization", *NUMERIC_FIELDS,
            "fresh_decode", "scaling_checksum",
        }}
        for key in NUMERIC_FIELDS:
            if key in converted:
                converted[key] = (
                    float(converted[key]) if key in {"PSNR", "SSIM", "LPIPS"}
                    else int(converted[key])
                )
        converted["lambda"] = 0.004
        converted["fresh_decode"] = converted["fresh_decode"].lower() == "true"
        converted["representation_checksum"] = converted.pop("scaling_checksum")
        selected.append(converted)

    dr_source = next(
        row for row in source
        if row["scene"] == "drjohnson" and row["target"] == "scaling"
    )
    streams = stream_bytes(dr_fp16)
    model_bytes = (dr_fp16 / "coview_model.bin").stat().st_size
    fresh = last_json_line(dr_fresh)
    selected.append({
        "scene": "drjohnson",
        "lambda": 0.004,
        "target": "scaling",
        "serialization": "fp16",
        **streams,
        "base_mlp_bytes": int(dr_source["base_mlp_bytes"]),
        "coview_model_bytes": model_bytes,
        "total_bytes": sum(streams.values()) + int(dr_source["base_mlp_bytes"])
        + model_bytes + 24,
        "PSNR": float(dr_source["PSNR"]),
        "SSIM": float(dr_source["SSIM"]),
        "LPIPS": float(dr_source["LPIPS"]),
        "fresh_decode": bool(fresh and fresh.get("num_anchors")),
        "representation_checksum": None if fresh is None else fresh.get("scaling_checksum"),
    })
    return selected


def feature_rows(legacy_csv, feature_root, inherited_quality):
    rows = []
    for row in csv.DictReader(legacy_csv.open()):
        copied = dict(row)
        for key in (
            "num_anchors", "feature_stream_bytes", "coview_model_bytes",
            "feature_plus_model_bytes", "raw_feature_bytes_saved",
            "net_feature_bytes_saved",
        ):
            copied[key] = int(copied[key])
        for key in ("estimated_bits", "training_seconds"):
            copied[key] = float(copied[key])
        copied["fresh_decode"] = True
        copied.update(inherited_quality["playroom"])
        rows.append(copied)

    for label, storage in (
        ("chunk_fp32", "fp32"),
        ("chunk_fp16", "fp16"),
        ("chunk_int8", "int8"),
    ):
        path = feature_root / label
        manifest = json.loads((path / "manifest.json").read_text())
        for result_label, result in manifest["results"].items():
            if storage != "fp32" and result_label == "a1_base":
                continue
            experiment, variant = result_label.split("_", 1)
            control_label = "a1_base" if experiment == "a1" else "a2_control"
            control = manifest["results"][control_label]
            fresh = last_json_line(path / result_label / "fresh_decode.json")
            row = {
                "scene": "drjohnson",
                "experiment": experiment,
                "variant": variant,
                "feature_mode": "chunk",
                "serialization": result["serialization"] or "none",
                "num_anchors": result["num_anchors"],
                "feature_stream_bytes": result["feature_stream_bytes"],
                "coview_model_bytes": result["coview_model_bytes"],
                "feature_plus_model_bytes": result["feature_plus_model_bytes"],
                "raw_feature_bytes_saved": control["feature_stream_bytes"]
                - result["feature_stream_bytes"],
                "net_feature_bytes_saved": control["feature_stream_bytes"]
                - result["feature_plus_model_bytes"],
                "estimated_bits": result["estimated_bits"],
                "training_seconds": result["training_seconds"],
                "symbol_index_checksum": result["symbol_index_checksum"],
                "fresh_decode": bool(
                    fresh and fresh.get("fresh_decode")
                    and fresh.get("symbol_index_checksum")
                    == result["symbol_index_checksum"]
                ),
            }
            row.update(inherited_quality["drjohnson"])
            rows.append(row)
    return rows


def paired_deltas(rows):
    results = []
    for scene in sorted({row["scene"] for row in rows}):
        for lmbda in sorted({row["lambda"] for row in rows if row["scene"] == scene}):
            control = next(row for row in rows if row["scene"] == scene and row["lambda"] == lmbda and row["target"] == "control")
            scaling = next(row for row in rows if row["scene"] == scene and row["lambda"] == lmbda and row["target"] == "scaling")
            results.append({
                "scene": scene,
                "lambda": lmbda,
                "total_bytes_saved": control["total_bytes"] - scaling["total_bytes"],
                "PSNR_delta": scaling["PSNR"] - control["PSNR"],
                "SSIM_delta": scaling["SSIM"] - control["SSIM"],
                "LPIPS_delta": scaling["LPIPS"] - control["LPIPS"],
            })
    return results


def bd_rate_rows(rows):
    results = []
    for scene in sorted({row["scene"] for row in rows}):
        control = [
            (row["total_bytes"], row["PSNR"]) for row in rows
            if row["scene"] == scene and row["target"] == "control"
        ]
        scaling = [
            (row["total_bytes"], row["PSNR"]) for row in rows
            if row["scene"] == scene and row["target"] == "scaling"
        ]
        result = {"scene": scene, "quality_metric": "PSNR"}
        result.update(bd_rate(control, scaling))
        results.append(result)
    return results


def plot_rd(rows, output):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths = []
    for scene in sorted({row["scene"] for row in rows}):
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for target, marker in (("control", "o"), ("scaling", "s")):
            selected = sorted(
                (row for row in rows if row["scene"] == scene and row["target"] == target),
                key=lambda row: row["total_bytes"],
            )
            rates = [row["total_bytes"] / (1024 * 1024) for row in selected]
            for axis, metric in zip(axes, ("PSNR", "SSIM", "LPIPS")):
                axis.plot(rates, [row[metric] for row in selected], marker=marker, label=target)
                axis.set_xlabel("complete package (MiB)")
                axis.set_ylabel(metric)
                axis.grid(alpha=0.25)
        axes[0].legend()
        figure.suptitle(f"Phase 2D Scaling RD: {scene}")
        figure.tight_layout()
        path = output / f"rd_{scene}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rd_root", required=True)
    parser.add_argument("--phase2c_scaling_csv", required=True)
    parser.add_argument("--dr_lambda004_fp16", required=True)
    parser.add_argument("--dr_lambda004_fresh", required=True)
    parser.add_argument("--phase2c_feature_csv", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    rd = phase2d_rd_rows(Path(args.rd_root).resolve())
    rd.extend(legacy_rows(
        Path(args.phase2c_scaling_csv).resolve(),
        Path(args.dr_lambda004_fp16).resolve(),
        Path(args.dr_lambda004_fresh).resolve(),
    ))
    rd.sort(key=lambda row: (row["scene"], row["lambda"], row["target"]))
    deltas = paired_deltas(rd)
    bd_rates = bd_rate_rows(rd)
    quality = {
        scene: {
            "PSNR": next(row for row in rd if row["scene"] == scene and row["lambda"] == 0.004 and row["target"] == "control")["PSNR"],
            "SSIM": next(row for row in rd if row["scene"] == scene and row["lambda"] == 0.004 and row["target"] == "control")["SSIM"],
            "LPIPS": next(row for row in rd if row["scene"] == scene and row["lambda"] == 0.004 and row["target"] == "control")["LPIPS"],
            "quality_relation": "identical_frozen_representation",
        }
        for scene in {row["scene"] for row in rd}
    }
    feature = feature_rows(
        Path(args.phase2c_feature_csv).resolve(),
        Path(args.feature_root).resolve(),
        quality,
    )
    write_csv(output / "scaling_rd.csv", rd)
    write_csv(output / "scaling_paired_deltas.csv", deltas)
    write_csv(output / "bd_rate.csv", bd_rates)
    write_csv(output / "feature_frozen_cross_scene.csv", feature)
    plots = plot_rd(rd, output)
    (output / "summary.json").write_text(json.dumps({
        "scaling_rd": rd,
        "scaling_paired_deltas": deltas,
        "bd_rate": bd_rates,
        "feature_frozen_cross_scene": feature,
        "plots": plots,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
