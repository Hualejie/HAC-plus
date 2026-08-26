from collections import OrderedDict

import pytest
import torch

from utils.coview_serialization import (
    deserialize_named_tensors,
    serialize_named_tensors,
)


@pytest.fixture
def state():
    return OrderedDict([
        ("shared.0.weight", torch.linspace(-1.0, 1.0, 60).reshape(4, 15)),
        ("shared.0.bias", torch.tensor([0.0, -0.25, 0.5, 1.0])),
        ("head.scaling.weight", torch.zeros(12, 4)),
        ("gate.scaling", torch.tensor(0.75)),
    ])


@pytest.mark.parametrize("storage_format", ["fp32", "fp16", "int8"])
def test_coview_serialization_is_deterministic_and_round_trips(
    tmp_path, state, storage_format
):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first_metadata = serialize_named_tensors(state, first, storage_format)
    second_metadata = serialize_named_tensors(reversed(list(state.items())), second, storage_format)
    decoded, decoded_metadata = deserialize_named_tensors(first)

    assert first.read_bytes() == second.read_bytes()
    assert first_metadata == second_metadata == decoded_metadata
    assert first_metadata["bytes"] == first.stat().st_size
    assert list(decoded) == sorted(state)
    tolerance = {"fp32": 0.0, "fp16": 5e-4, "int8": 8e-3}[storage_format]
    for name, expected in state.items():
        torch.testing.assert_close(decoded[name], expected, rtol=0, atol=tolerance)


def test_coview_serialization_rejects_truncation(tmp_path, state):
    path = tmp_path / "model.bin"
    serialize_named_tensors(state, path, "fp32")
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(RuntimeError, match="truncated"):
        deserialize_named_tensors(path)
