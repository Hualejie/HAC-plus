"""Run paired Phase 2D Scaling RD experiments.

For every scene/lambda pair this driver first creates one shared M15k
checkpoint, then resumes matched Control and Scaling branches on isolated CUDA
devices.  The Scaling float model is re-encoded with FP16 parameters and both
final packages are verified by a fresh-process decoder.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


DEFAULT_SCENES = ("playroom", "drjohnson")
DEFAULT_LAMBDAS = (0.003, 0.002, 0.001, 0.0005)


def lambda_tag(value):
    return format(value, ".8f").rstrip("0").rstrip(".")


def mask_lr(value):
    return 0.00008 * value / 0.001


def common_train_command(args, scene, lmbda, output, iterations, target):
    return [
        args.conda,
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        "train.py",
        "-s",
        str(Path(args.dataset_root) / scene),
        "--eval",
        "--lod",
        "0",
        "--voxel_size",
        "0.005",
        "--update_init_factor",
        "16",
        "--iterations",
        str(iterations),
        "-m",
        str(output),
        "--lmbda",
        str(lmbda),
        "--mask_lr_final",
        str(mask_lr(lmbda)),
        "--use_view_topology",
        "--coview_target",
        target,
    ]


def shared_command(args, scene, lmbda, output):
    command = common_train_command(
        args, scene, lmbda, output, iterations=15000, target="none"
    )
    command.extend([
        "--checkpoint_iterations",
        "15000",
        "--stop_after_checkpoint",
    ])
    return command


def branch_command(args, scene, lmbda, output, target, checkpoint):
    command = common_train_command(
        args, scene, lmbda, output, iterations=30000, target=target
    )
    command.extend(["--start_checkpoint", str(checkpoint)])
    return command


def task_complete(task):
    return all(path.exists() for path in task["sentinels"])


def run_tasks(tasks, repo, dry_run=False):
    pending = [task for task in tasks if not task_complete(task)]
    for task in tasks:
        if task_complete(task):
            print(f"skip complete task: {task['name']}", flush=True)
    if dry_run:
        for task in pending:
            print(
                f"CUDA_VISIBLE_DEVICES={task['gpu']} "
                + shlex.join(task["command"]),
                flush=True,
            )
        return

    processes = []
    for task in pending:
        task["log"].parent.mkdir(parents=True, exist_ok=True)
        handle = open(task["log"], "a", buffering=1)
        command_text = shlex.join(task["command"])
        handle.write(f"\n$ CUDA_VISIBLE_DEVICES={task['gpu']} {command_text}\n")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(task["gpu"])
        print(f"start {task['name']} on GPU {task['gpu']}", flush=True)
        process = subprocess.Popen(
            task["command"],
            cwd=repo,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((task, process, handle))

    failures = []
    for task, process, handle in processes:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append((task["name"], return_code, task["log"]))
        else:
            if task.get("completion_marker") is not None:
                task["completion_marker"].write_text("ok\n")
            print(f"completed {task['name']}", flush=True)
    if failures:
        details = ", ".join(
            f"{name} exit={code} log={log}" for name, code, log in failures
        )
        raise RuntimeError(f"Phase 2D task failure: {details}")
    for task in pending:
        if not task_complete(task):
            raise RuntimeError(
                f"{task['name']} exited successfully but missed sentinels: "
                f"{task['sentinels']}"
            )


def training_task(name, gpu, command, output, shared=False):
    sentinels = (
        [output / "chkpnt15000.pth"]
        if shared
        else [
            output / "results.json",
            output / "bitstreams" / "entropy_context.pth",
            output / "float_model" / "iteration_30000" / "checkpoint.pth",
        ]
    )
    return {
        "name": name,
        "gpu": gpu,
        "command": command,
        "log": output / "phase2d_driver.log",
        "sentinels": sentinels,
    }


def postprocess_tasks(args, scene, lmbda, root, gpus):
    tag = lambda_tag(lmbda)
    point = root / scene / f"lambda_{tag}"
    control = point / "control"
    scaling = point / "scaling"
    fp16 = scaling / "bitstreams_fp16"
    encode = {
        "name": f"{scene}/{tag}/scaling-fp16-encode",
        "gpu": gpus[1],
        "command": [
            args.conda,
            "run",
            "--no-capture-output",
            "-n",
            args.conda_env,
            "python",
            "analysis/encode_float_model.py",
            "--float_model",
            str(scaling / "float_model" / "iteration_30000"),
            "--output",
            str(fp16),
            "--coview_target",
            "scaling",
            "--camera_context",
            str(scaling / "bitstreams" / "entropy_context.pth"),
            "--coview_serialization",
            "fp16",
        ],
        "log": scaling / "fp16_encode.log",
        "sentinels": [fp16 / "coview_model.bin", fp16 / "entropy_context.pth"],
    }
    run_tasks([encode], args.repo, args.dry_run)
    fresh = []
    for target, bitstream, gpu, output in (
        ("none", control / "bitstreams", gpus[0], control / "fresh_decode.json"),
        ("scaling", fp16, gpus[1], scaling / "fresh_decode_fp16.json"),
    ):
        marker = output.with_suffix(output.suffix + ".done")
        fresh.append({
            "name": f"{scene}/{tag}/{target}-fresh-decode",
            "gpu": gpu,
            "command": [
                args.conda,
                "run",
                "--no-capture-output",
                "-n",
                args.conda_env,
                "python",
                "analysis/fresh_entropy_decode.py",
                "--bitstream",
                str(bitstream),
                "--coview_target",
                target,
            ],
            "log": output,
            "sentinels": [output, marker],
            "completion_marker": marker,
        })
    run_tasks(fresh, args.repo, args.dry_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument(
        "--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS
    )
    parser.add_argument("--gpus", nargs=2, default=("0", "1"))
    parser.add_argument(
        "--conda", default="/home/fansonglin/miniconda3/bin/conda"
    )
    parser.add_argument("--conda_env", default="HAC_5090_a100")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    args.repo = Path(__file__).resolve().parents[1]
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiment_config.json").write_text(json.dumps({
        "scenes": args.scenes,
        "lambdas": args.lambdas,
        "gpus": args.gpus,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "feature_mode": "full",
        "scaling_serialization": "fp16",
        "shared_iteration": 15000,
        "final_iteration": 30000,
    }, indent=2, sort_keys=True))

    for lmbda in args.lambdas:
        tag = lambda_tag(lmbda)
        shared_tasks = []
        for index, scene in enumerate(args.scenes):
            point = root / scene / f"lambda_{tag}"
            shared = point / "shared_m15k"
            shared_tasks.append(training_task(
                f"{scene}/{tag}/shared",
                args.gpus[index % 2],
                shared_command(args, scene, lmbda, shared),
                shared,
                shared=True,
            ))
        run_tasks(shared_tasks, args.repo, args.dry_run)

        for scene in args.scenes:
            point = root / scene / f"lambda_{tag}"
            shared = point / "shared_m15k" / "chkpnt15000.pth"
            branch_tasks = []
            for target, gpu in zip(("none", "scaling"), args.gpus):
                label = "control" if target == "none" else target
                output = point / label
                branch_tasks.append(training_task(
                    f"{scene}/{tag}/{label}",
                    gpu,
                    branch_command(
                        args, scene, lmbda, output, target, shared
                    ),
                    output,
                ))
            run_tasks(branch_tasks, args.repo, args.dry_run)
            postprocess_tasks(args, scene, lmbda, root, args.gpus)


if __name__ == "__main__":
    main()
