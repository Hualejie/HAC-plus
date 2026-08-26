from pathlib import Path
from types import SimpleNamespace

from analysis.run_phase2d_rd import (
    branch_command,
    lambda_tag,
    mask_lr,
    shared_command,
)


def arguments(tmp_path):
    return SimpleNamespace(
        conda="/conda",
        conda_env="hac",
        dataset_root=tmp_path / "dataset",
    )


def test_lambda_tag_and_mask_schedule():
    assert lambda_tag(0.004) == "0.004"
    assert lambda_tag(0.0005) == "0.0005"
    assert mask_lr(0.004) == 0.00032
    assert mask_lr(0.0005) == 0.00004


def test_shared_command_stops_at_paired_checkpoint(tmp_path):
    command = shared_command(
        arguments(tmp_path), "playroom", 0.002, tmp_path / "shared"
    )
    assert command[command.index("--iterations") + 1] == "15000"
    assert command[command.index("--coview_target") + 1] == "none"
    assert command[command.index("--checkpoint_iterations") + 1] == "15000"
    assert "--stop_after_checkpoint" in command


def test_branch_command_uses_same_checkpoint_and_target(tmp_path):
    checkpoint = tmp_path / "shared" / "chkpnt15000.pth"
    command = branch_command(
        arguments(tmp_path),
        "drjohnson",
        0.001,
        tmp_path / "scaling",
        "scaling",
        checkpoint,
    )
    assert command[command.index("--iterations") + 1] == "30000"
    assert command[command.index("--coview_target") + 1] == "scaling"
    assert Path(command[command.index("--start_checkpoint") + 1]) == checkpoint
