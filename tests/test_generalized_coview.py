from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scene.gaussian_model import GaussianModel
from utils.training_checkpoint import (
    camera_checkpoint_key,
    capture_rng_state,
    restore_rng_state,
    restore_viewpoint_stack,
)


ATTRIBUTE_DIMS = {"feature": 50, "scaling": 6, "offset": 30}


def _context_model(target, feature_mode="full"):
    model = GaussianModel.__new__(GaussianModel)
    nn.Module.__init__(model)
    model.use_view_topology = True
    model.coview_target = target
    model.coview_feature_mode = feature_mode
    model._coview_residual_stats = {}
    model._coview_residual_accumulators = {}
    model._collect_coview_statistics = True
    model.mlp_coview_shared = nn.Sequential(nn.Linear(15, 4), nn.ReLU())
    model.mlp_coview_feature = nn.Linear(4, 100 if feature_mode == "full" else 10)
    model.mlp_coview_scaling = nn.Linear(4, 12)
    model.mlp_coview_offset = nn.Linear(4, 60)
    model.coview_gates = nn.ParameterDict({
        name: nn.Parameter(torch.ones(())) for name in ATTRIBUTE_DIMS
    })
    model.mlp_dummy_base = nn.Linear(2, 3)
    with torch.no_grad():
        model.mlp_coview_shared[0].weight.fill_(0.1)
        model.mlp_coview_shared[0].bias.fill_(0.2)
        for name in ATTRIBUTE_DIMS:
            head = getattr(model, f"mlp_coview_{name}")
            head.weight.fill_(0.1)
            head.bias.fill_(0.05)
    return model


def _apply_all(model):
    topology = torch.ones(3, 15)
    outputs = {}
    for attribute, dim in ATTRIBUTE_DIMS.items():
        mean = torch.zeros(3, dim)
        scale = torch.ones(3, dim)
        outputs[attribute] = model.apply_coview_entropy_context(
            mean, scale, topology, attribute
        )
    return outputs


def test_none_is_exact_baseline_path_without_topology():
    model = _context_model("none")
    mean = torch.randn(2, 50)
    scale = torch.rand(2, 50)
    output_mean, output_scale = model.apply_coview_entropy_context(
        mean, scale, None, "feature"
    )
    assert output_mean is mean
    assert output_scale is scale
    assert not model.coview_enabled


@pytest.mark.parametrize("target", ["feature", "scaling", "offset"])
def test_single_target_only_changes_requested_attribute(target):
    outputs = _apply_all(_context_model(target))
    for attribute, (mean, scale) in outputs.items():
        changed = not torch.equal(mean, torch.zeros_like(mean))
        assert changed is (attribute == target)
        assert (not torch.equal(scale, torch.ones_like(scale))) is (attribute == target)


def test_all_target_changes_every_attribute_and_records_gates():
    model = _context_model("all")
    outputs = _apply_all(model)
    assert all(not torch.equal(mean, torch.zeros_like(mean)) for mean, _ in outputs.values())
    assert set(model._coview_residual_stats) == set(ATTRIBUTE_DIMS)
    assert all(stats["gate"] == 1.0 for stats in model._coview_residual_stats.values())


def test_chunk_feature_head_shares_one_residual_per_ten_channels():
    model = _context_model("feature", feature_mode="chunk")
    mean, scale = model.apply_coview_entropy_context(
        torch.zeros(3, 50), torch.ones(3, 50), torch.ones(3, 15), "feature"
    )
    for chunk in range(5):
        chunk_mean = mean[:, chunk * 10:(chunk + 1) * 10]
        chunk_scale = scale[:, chunk * 10:(chunk + 1) * 10]
        torch.testing.assert_close(
            chunk_mean, chunk_mean[:, :1].expand_as(chunk_mean), rtol=0, atol=0
        )
        torch.testing.assert_close(
            chunk_scale, chunk_scale[:, :1].expand_as(chunk_scale), rtol=0, atol=0
        )


def test_entropy_parameter_prediction_is_state_dict_deterministic():
    encoder = _context_model("all")
    decoder = _context_model("all")
    decoder.load_state_dict(encoder.state_dict())
    encoder_outputs = _apply_all(encoder)
    decoder_outputs = _apply_all(decoder)
    for attribute in ATTRIBUTE_DIMS:
        for first, second in zip(encoder_outputs[attribute], decoder_outputs[attribute]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_active_coview_bytes_only_count_shared_and_selected_heads():
    model = _context_model("feature")
    sizes = model.get_mlp_size_breakdown()
    expected_coview_params = (
        sum(p.numel() for p in model.mlp_coview_shared.parameters())
        + sum(p.numel() for p in model.mlp_coview_feature.parameters())
        + 1
    )
    assert sizes["active_coview_bits"] == expected_coview_params * 32
    model.coview_target = "none"
    none_sizes = model.get_mlp_size_breakdown()
    assert none_sizes["active_coview_bits"] == 0
    assert none_sizes["base_bits"] == sizes["base_bits"]


def _checkpoint_model():
    model = _context_model("none")
    model.feat_dim = 50
    model.n_offsets = 10
    model.use_feat_bank = False
    model.view_topology_k = 8
    model.view_topology_candidates = 16
    model.coview_feature_mode = "full"
    for name in (
        "mlp_opacity", "mlp_cov", "mlp_color", "encoding_xyz",
        "mlp_grid", "mlp_deform",
    ):
        setattr(model, name, nn.Linear(2, 2))
    shapes = {
        "_anchor": (2, 3), "_offset": (2, 10, 3), "_mask": (2, 11, 1),
        "_anchor_feat": (2, 50), "_scaling": (2, 6),
        "_rotation": (2, 4), "_opacity": (2, 1),
    }
    for name, shape in shapes.items():
        setattr(model, name, nn.Parameter(torch.randn(*shape)))
    model.max_radii2D = torch.randn(2)
    model.opacity_accum = torch.randn(2, 1)
    model.offset_gradient_accum = torch.randn(20, 1)
    model.offset_denom = torch.randn(20, 1)
    model.anchor_demon = torch.randn(2, 1)
    model.spatial_lr_scale = 3.0
    model.percent_dense = 0.01
    model.x_bound_min = torch.randn(1, 3)
    model.x_bound_max = torch.randn(1, 3)

    def training_setup(instance, _):
        instance.optimizer = torch.optim.Adam(instance.parameters(), lr=1e-3)

    object.__setattr__(model, "training_setup", MethodType(training_setup, model))
    model.training_setup(None)
    sum(parameter.sum() for parameter in model.parameters()).backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    return model


def test_full_model_optimizer_and_buffers_checkpoint_round_trip():
    source = _checkpoint_model()
    state = source.training_checkpoint_state()
    restored = _checkpoint_model()
    restored.restore_training_checkpoint(state, None)

    for name, saved in state["gaussian_parameters"].items():
        torch.testing.assert_close(getattr(restored, name), saved["tensor"])
    for name, saved in state["training_buffers"].items():
        torch.testing.assert_close(getattr(restored, name), saved)
    for name, module in restored._checkpoint_modules().items():
        for key, tensor in module.state_dict().items():
            torch.testing.assert_close(tensor, state["module_state_dicts"][name][key])
    assert restored.optimizer.state_dict()["param_groups"] == state["optimizer"]["param_groups"]


def test_rng_and_camera_stack_round_trip():
    cameras = [
        SimpleNamespace(image_name="a", colmap_id=1, uid=11),
        SimpleNamespace(image_name="b", colmap_id=2, uid=12),
        SimpleNamespace(image_name="c", colmap_id=3, uid=13),
    ]
    saved_rng = capture_rng_state()
    expected_numpy = np.random.rand(4)
    expected_torch = torch.rand(4)
    restore_rng_state(saved_rng)
    np.testing.assert_array_equal(np.random.rand(4), expected_numpy)
    torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)

    saved_keys = [camera_checkpoint_key(cameras[2]), camera_checkpoint_key(cameras[0])]
    restored = restore_viewpoint_stack(list(reversed(cameras)), saved_keys)
    assert [camera.image_name for camera in restored] == ["c", "a"]
