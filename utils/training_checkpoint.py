import random

import numpy as np
import torch


TRAINING_RUN_CHECKPOINT_VERSION = 1


def camera_checkpoint_key(camera):
    return (
        str(getattr(camera, "image_name", "")),
        int(getattr(camera, "colmap_id", -1)),
        int(getattr(camera, "uid", -1)),
    )


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def capture_training_run(gaussians, iteration, viewpoint_stack, config):
    return {
        "version": TRAINING_RUN_CHECKPOINT_VERSION,
        "iteration": iteration,
        "model": gaussians.training_checkpoint_state(),
        "config": dict(config),
        "rng": capture_rng_state(),
        "viewpoint_stack": [
            camera_checkpoint_key(camera) for camera in (viewpoint_stack or [])
        ],
    }


def restore_viewpoint_stack(cameras, saved_keys):
    camera_by_key = {}
    for camera in cameras:
        key = camera_checkpoint_key(camera)
        if key in camera_by_key:
            raise RuntimeError(f"duplicate training camera checkpoint key {key!r}")
        camera_by_key[key] = camera
    missing = [key for key in saved_keys if tuple(key) not in camera_by_key]
    if missing:
        raise RuntimeError(f"checkpoint training cameras are missing: {missing[:3]}")
    return [camera_by_key[tuple(key)] for key in saved_keys]
