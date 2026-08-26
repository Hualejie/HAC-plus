"""Collect Phase 2B shared-checkpoint experiment outputs into JSON and CSV."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

import torch


TARGETS = ("none", "feature", "scaling", "offset", "all")
ATTRIBUTES = ("feature", "scaling", "offset")
SIZE_RE = re.compile(
    r"Encoded sizes in MB: anchor (?P<anchor>[0-9.]+), feat (?P<feature>[0-9.]+), "
    r"scaling (?P<scaling>[0-9.]+), offsets (?P<offset>[0-9.]+), "
    r"hash (?P<hash>[0-9.]+), masks (?P<mask>[0-9.]+), "
    r"base_MLPs (?P<base_mlp>[0-9.]+), active_CoView_MLPs (?P<coview_mlp>[0-9.]+), "
    r"MLPs (?P<mlp>[0-9.]+), Total (?P<total>[0-9.]+), EncTime (?P<encode_time>[0-9.]+)"
)


def _last_match(pattern, text):
    matches = list(pattern.finditer(text))
    if not matches:
        raise RuntimeError(f"missing log pattern {pattern.pattern!r}")
    return matches[-1]


def _last_literal(text, prefix):
    lines = [line.split(prefix, 1)[1] for line in text.splitlines() if prefix in line]
    return ast.literal_eval(lines[-1]) if lines else {}


def _sum_files(path, pattern):
    return sum(file.stat().st_size for file in path.glob(pattern))


def _tensor_bytes(state):
    return sum(tensor.numel() * tensor.element_size() for tensor in state.values())


def _mlp_bytes(float_checkpoint, target):
    checkpoint = torch.load(float_checkpoint, map_location="cpu")
    base_keys = ("opacity_mlp", "cov_mlp", "color_mlp", "grid_mlp", "deform_mlp")
    if "mlp_feature_bank" in checkpoint:
        base_keys += ("mlp_feature_bank",)
    base = sum(_tensor_bytes(checkpoint[key]) for key in base_keys)
    active = () if target == "none" else (ATTRIBUTES if target == "all" else (target,))
    coview = 0
    if active:
        coview += _tensor_bytes(checkpoint["coview_shared_mlp"])
        for attribute in active:
            coview += _tensor_bytes(checkpoint[f"coview_{attribute}_head"])
            coview += checkpoint["coview_gates"][attribute].numel() * 4
    return base, coview


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_bytes(bitstream):
    return {
        "anchor_bytes": (bitstream / "xyz_gpcc.npz").stat().st_size,
        "feature_bytes": _sum_files(bitstream, "feat*.b"),
        "scaling_bytes": _sum_files(bitstream, "scaling*.b"),
        "offset_bytes": _sum_files(bitstream, "offsets*.b"),
        "hash_bytes": (bitstream / "hash.b").stat().st_size,
        "mask_bytes": (bitstream / "masks.b").stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--shared_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    shared_log = (root / "playroom_shared_m15k" / "outputs.log").read_text(errors="replace")
    shared_time = float(_last_match(re.compile(r"Total Training time: ([0-9.]+)"), shared_log).group(1))
    runs = {}
    gate_rows = []
    for target in TARGETS:
        run = root / f"playroom_{target}"
        log = (run / "outputs.log").read_text(errors="replace")
        logged_sizes = {key: float(value) for key, value in _last_match(SIZE_RE, log).groupdict().items()}
        metrics = json.loads((run / "results.json").read_text())["ours_30000"]
        stream = _stream_bytes(run / "bitstreams")
        base_mlp, coview_mlp = _mlp_bytes(
            run / "float_model" / "iteration_30000" / "checkpoint.pth", target
        )
        exact_total = sum(stream.values()) + base_mlp + coview_mlp + 24
        branch_time = float(
            _last_match(re.compile(r"Total Training time: ([0-9.]+)"), log).group(1)
        )
        decode_time = float(
            _last_match(re.compile(r"DecTime ([0-9.]+)"), log).group(1)
        )
        gates = _last_literal(log, "CoView gates: ")
        residual = _last_literal(log, "CoView residual statistics: ")
        fresh_path = run / "fresh_decode.json"
        fresh = json.loads(fresh_path.read_text().splitlines()[-1]) if fresh_path.exists() else None
        row = {
            "target": target,
            **stream,
            "base_mlp_bytes": base_mlp,
            "active_coview_mlp_bytes": coview_mlp,
            "total_bytes": exact_total,
            "logged_sizes_mib": logged_sizes,
            "PSNR": metrics["PSNR"],
            "SSIM": metrics["SSIM"],
            "LPIPS": metrics["LPIPS"],
            "shared_training_seconds": shared_time,
            "branch_training_seconds": branch_time,
            "total_training_seconds": shared_time + branch_time,
            "encoding_seconds": logged_sizes["encode_time"],
            "decoding_seconds": decode_time,
            "gates": gates,
            "residual_statistics": residual,
            "fresh_decode": fresh,
        }
        off_stream = run / "bitstreams_off"
        if off_stream.exists():
            row["same_checkpoint_off_stream_bytes"] = _stream_bytes(off_stream)
        runs[target] = row
        for attribute in ATTRIBUTES:
            stats = residual.get(attribute, {})
            gate_rows.append({
                "target": target,
                "attribute": attribute,
                "gate": gates.get(attribute),
                "residual_mean": stats.get("mean"),
                "residual_std": stats.get("std"),
                "residual_mean_abs": stats.get("mean_abs"),
                "residual_count": stats.get("count"),
            })

    control = runs["none"]
    for target, row in runs.items():
        row["delta_vs_control"] = {
            "feature_bytes_saved": control["feature_bytes"] - row["feature_bytes"],
            "scaling_bytes_saved": control["scaling_bytes"] - row["scaling_bytes"],
            "offset_bytes_saved": control["offset_bytes"] - row["offset_bytes"],
            "raw_attribute_bytes_saved": sum(
                control[f"{name}_bytes"] - row[f"{name}_bytes"]
                for name in ATTRIBUTES
            ),
            "net_total_bytes_saved": control["total_bytes"] - row["total_bytes"],
            "PSNR": row["PSNR"] - control["PSNR"],
            "SSIM": row["SSIM"] - control["SSIM"],
            "LPIPS": row["LPIPS"] - control["LPIPS"],
        }

    summary = {
        "scene": "playroom",
        "shared_checkpoint": str(Path(args.shared_checkpoint).resolve()),
        "shared_checkpoint_sha256": _sha256(args.shared_checkpoint),
        "runs": runs,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    rate_fields = [
        "target", "anchor_bytes", "feature_bytes", "scaling_bytes", "offset_bytes",
        "hash_bytes", "mask_bytes", "base_mlp_bytes", "active_coview_mlp_bytes",
        "total_bytes", "feature_bytes_saved", "scaling_bytes_saved",
        "offset_bytes_saved", "raw_attribute_bytes_saved", "net_total_bytes_saved",
        "PSNR", "SSIM", "LPIPS",
    ]
    with open(output / "attribute_rate_ablation.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rate_fields)
        writer.writeheader()
        for row in runs.values():
            flat = {key: row.get(key) for key in rate_fields}
            flat.update(row["delta_vs_control"])
            flat["PSNR"], flat["SSIM"], flat["LPIPS"] = row["PSNR"], row["SSIM"], row["LPIPS"]
            writer.writerow(flat)
    with open(output / "gate_statistics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=gate_rows[0].keys())
        writer.writeheader()
        writer.writerows(gate_rows)


if __name__ == "__main__":
    main()
