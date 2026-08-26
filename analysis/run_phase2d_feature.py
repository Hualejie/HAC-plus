"""Run the cross-scene Phase 2D Frozen Feature A1/A2 replication."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.run_phase2d_rd import run_tasks


CONFIGURATIONS = (
    ("chunk_fp32", "fp32", "a1,a2"),
    ("chunk_fp16", "fp16", "a1"),
    ("chunk_int8", "int8", "a1"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--float_model", required=True)
    parser.add_argument("--camera_context", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", default="1")
    parser.add_argument(
        "--conda", default="/home/fansonglin/miniconda3/bin/conda"
    )
    parser.add_argument("--conda_env", default="HAC_5090_a100")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    repo = REPO_ROOT
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiment_config.json").write_text(json.dumps({
        "float_model": str(Path(args.float_model).resolve()),
        "camera_context": str(Path(args.camera_context).resolve()),
        "configurations": CONFIGURATIONS,
        "scene_representation": "frozen",
        "feature_mode": "chunk",
    }, indent=2, sort_keys=True))

    for label, storage, experiments in CONFIGURATIONS:
        output = root / label
        command = [
            args.conda,
            "run",
            "--no-capture-output",
            "-n",
            args.conda_env,
            "python",
            "analysis/train_frozen_feature_entropy.py",
            "--float_model",
            str(Path(args.float_model).resolve()),
            "--camera_context",
            str(Path(args.camera_context).resolve()),
            "--output",
            str(output),
            "--experiments",
            experiments,
            "--feature_mode",
            "chunk",
            "--serialization",
            storage,
        ]
        run_tasks([{
            "name": f"feature/{label}/train",
            "gpu": args.gpu,
            "command": command,
            "log": output / "train.log",
            "sentinels": [output / "manifest.json"],
        }], repo, args.dry_run)
        if args.dry_run:
            continue

        manifest = json.loads((output / "manifest.json").read_text())
        for package_name in manifest["results"]:
            package = output / package_name
            fresh_log = package / "fresh_decode.json"
            marker = package / "fresh_decode.done"
            run_tasks([{
                "name": f"feature/{label}/{package_name}/fresh-decode",
                "gpu": args.gpu,
                "command": [
                    args.conda,
                    "run",
                    "--no-capture-output",
                    "-n",
                    args.conda_env,
                    "python",
                    "analysis/fresh_frozen_feature_decode.py",
                    "--package",
                    str(package),
                ],
                "log": fresh_log,
                "sentinels": [fresh_log, marker],
                "completion_marker": marker,
            }], repo, args.dry_run)


if __name__ == "__main__":
    main()
