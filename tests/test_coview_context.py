from types import SimpleNamespace

import numpy as np
import torch

from utils.general_utils import get_expon_lr_func

from scene.coview_context import (
    ObservationRelation,
    build_geometric_observation_descriptors,
    build_geometry_observations,
    build_view_topology_context,
    camera_geometry_from_state,
    camera_geometry_state,
    canonicalize_codec_anchors,
    coview_lsh_candidates,
    coview_topk,
    pair_geometric_view_scores,
    pair_distance_depth_scores,
    spatial_topk,
    spatial_topk_queries,
)


def _camera(name, uid, zfar=1.0):
    return SimpleNamespace(
        image_name=name,
        colmap_id=uid + 10,
        uid=uid,
        full_proj_transform=torch.eye(4),
        world_view_transform=torch.eye(4),
        image_width=100,
        image_height=80,
        znear=0.01,
        zfar=zfar,
    )


def _relation(camera_sets, num_cameras):
    camera_to_anchor = []
    for camera_id in range(num_cameras):
        camera_to_anchor.append(
            np.asarray([idx for idx, cameras in enumerate(camera_sets) if camera_id in cameras], dtype=np.int64)
        )
    ids = []
    indptr = [0]
    for cameras in camera_sets:
        ids.extend(sorted(cameras))
        indptr.append(len(ids))
    return ObservationRelation(
        camera_keys=tuple((str(idx), idx, idx) for idx in range(num_cameras)),
        camera_to_anchor=tuple(camera_to_anchor),
        anchor_indptr=np.asarray(indptr, dtype=np.int64),
        anchor_camera_ids=np.asarray(ids, dtype=np.int32),
        num_anchors=len(camera_sets),
    )


def test_codec_anchor_attribute_alignment_and_quantization():
    anchor = torch.tensor([
        [2.04, 0.0, 0.0],
        [9.00, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [1.02, 0.0, 0.0],
    ])
    mask = torch.tensor([[1], [0], [1], [1]], dtype=torch.bool)
    attribute = torch.arange(4)[:, None]
    aligned = canonicalize_codec_anchors(anchor, mask, 1.0, {"id": attribute})

    assert aligned.valid_original_idx.tolist() == [0, 2, 3]
    assert aligned.codec_original_idx.tolist() == [2, 3, 0]
    assert aligned.attributes["id"].reshape(-1).tolist() == [2, 3, 0]
    torch.testing.assert_close(aligned.xyz, torch.round(aligned.xyz / 1.0) * 1.0)
    torch.testing.assert_close(aligned.xyz, aligned.integer_xyz * 1.0)


def test_camera_permutation_invariance():
    xyz = torch.tensor([
        [0.0, 0.0, 0.5],
        [2.0, 0.0, 0.5],
        [0.0, 0.0, 2.0],
    ])
    cameras = [_camera("b", 1, zfar=3.0), _camera("a", 0, zfar=1.0)]
    forward = build_geometry_observations(xyz, cameras, chunk_size=2)
    reverse = build_geometry_observations(xyz, list(reversed(cameras)), chunk_size=1)

    assert forward.camera_keys == reverse.camera_keys
    np.testing.assert_array_equal(forward.anchor_indptr, reverse.anchor_indptr)
    np.testing.assert_array_equal(forward.anchor_camera_ids, reverse.anchor_camera_ids)
    for first, second in zip(forward.camera_to_anchor, reverse.camera_to_anchor):
        np.testing.assert_array_equal(first, second)


def test_coview_topk_deterministic_tie_break():
    relation = _relation([{0}, {0}, {0}, {0}], num_cameras=1)
    neighbors, scores, diagnostics = coview_topk(relation, k=2, score_block_size=1)

    np.testing.assert_array_equal(neighbors, np.asarray([[1, 2], [0, 2], [0, 1], [0, 1]]))
    np.testing.assert_allclose(scores, 1.0)
    assert diagnostics["dense_anchor_pair_matrix_created"] is False


def test_coview_uses_bounded_signature_score_blocks_not_anchor_square():
    relation = _relation(
        [{0}, {1}, {0, 1}, {2}, {0, 2}, {1, 2}, {0, 1, 2}, set()],
        num_cameras=3,
    )
    _, _, diagnostics = coview_topk(relation, k=2, score_block_size=2)

    rows, columns = diagnostics["max_score_block_shape"]
    assert rows <= 2
    assert columns == diagnostics["num_unique_signatures"]
    assert [rows, columns] != [relation.num_anchors, relation.num_anchors]
    assert diagnostics["dense_anchor_pair_matrix_created"] is False


def test_coview_lsh_candidates_are_bounded_and_deterministic():
    relation = _relation(
        [{0, 1}, {0}, {0, 1}, {1}, {1, 2}, {2}, {0, 2}, set()],
        num_cameras=3,
    )
    first_rows, first_scores, first_diagnostics = coview_lsh_candidates(
        relation,
        k=3,
        bucket_window=2,
    )
    second_rows, second_scores, second_diagnostics = coview_lsh_candidates(
        relation,
        k=3,
        bucket_window=2,
    )

    for first, second in zip(first_rows, second_rows):
        np.testing.assert_array_equal(first, second)
        assert first.size <= 3
    for first, second in zip(first_scores, second_scores):
        np.testing.assert_array_equal(first, second)
    assert first_diagnostics == second_diagnostics
    assert first_diagnostics["quadratic_pair_enumeration"] is False
    assert first_diagnostics["dense_anchor_pair_matrix_created"] is False


def test_spatial_topk_ties_resolve_by_anchor_index():
    xyz = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    neighbors, distances = spatial_topk(xyz, k=2, query_chunk_size=2)
    np.testing.assert_array_equal(neighbors[0], [1, 2])
    np.testing.assert_allclose(distances[0], [1.0, 1.0])

    subset_neighbors, subset_distances = spatial_topk_queries(xyz, [0, 4], k=2)
    np.testing.assert_array_equal(subset_neighbors, neighbors[[0, 4]])
    np.testing.assert_allclose(subset_distances, distances[[0, 4]])


def test_geometric_descriptors_are_camera_permutation_invariant():
    xyz = torch.tensor([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.8, 0.0, 0.5]])
    cameras = [_camera("b", 1), _camera("a", 0)]
    first_relation = build_geometry_observations(xyz, cameras)
    second_relation = build_geometry_observations(xyz, list(reversed(cameras)))
    first = build_geometric_observation_descriptors(xyz, cameras, first_relation)
    second = build_geometric_observation_descriptors(xyz, list(reversed(cameras)), second_relation)

    np.testing.assert_allclose(first.distance, second.distance)
    np.testing.assert_allclose(first.view_direction, second.view_direction)
    np.testing.assert_allclose(first.image_xy, second.image_xy)
    np.testing.assert_allclose(first.depth, second.depth)


def test_geometric_view_score_resolves_saturated_binary_jaccard():
    xyz = torch.tensor([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.8, 0.0, 0.5]])
    cameras = [_camera("a", 0), _camera("b", 1)]
    relation = build_geometry_observations(xyz, cameras)
    descriptors = build_geometric_observation_descriptors(xyz, cameras, relation)
    scores = pair_geometric_view_scores(
        descriptors,
        source=np.asarray([0, 0]),
        target=np.asarray([1, 2]),
        batch_size=1,
    )

    np.testing.assert_allclose(scores["binary_jaccard"], [1.0, 1.0])
    assert scores["geometric_image"][0] > scores["geometric_image"][1]
    assert scores["geometric_composite"][0] > scores["geometric_composite"][1]
    np.testing.assert_array_equal(scores["common_camera_count"], [2.0, 2.0])


def test_view_topology_is_camera_permutation_invariant_and_quantized():
    xyz = torch.tensor([
        [-0.8, 0.0, 0.5],
        [-0.2, 0.0, 0.5],
        [0.1, 0.0, 0.5],
        [0.7, 0.0, 0.5],
    ])
    cameras = [_camera("b", 1), _camera("a", 0)]
    first = build_view_topology_context(
        xyz, cameras, candidate_k=3, topk=2, feature_quantization=1e-4,
    )
    second = build_view_topology_context(
        xyz, list(reversed(cameras)), candidate_k=3, topk=2, feature_quantization=1e-4,
    )

    np.testing.assert_array_equal(first.neighbors, second.neighbors)
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_allclose(first.features / 1e-4, np.round(first.features / 1e-4), atol=1e-3)
    assert first.features.shape == (4, 15)
    assert first.diagnostics["dense_anchor_pair_matrix_created"] is False


def test_hybrid_view_candidates_escape_the_spatial_candidate_ceiling():
    xyz = torch.tensor([
        [-0.90, 0.0, 0.50],
        [-0.85, 0.0, 1.00],
        [-0.80, 0.0, 1.00],
        [-0.75, 0.0, 1.00],
        [0.00, 0.0, 1.00],
        [0.90, 0.0, 0.50],
    ])
    cameras = [
        _camera("near", 0, zfar=0.75),
        _camera("far", 1, zfar=3.0),
    ]

    spatial = build_view_topology_context(
        xyz,
        cameras,
        candidate_k=3,
        topk=1,
        candidate_mode="spatial",
    )
    hybrid = build_view_topology_context(
        xyz,
        list(reversed(cameras)),
        candidate_k=3,
        topk=1,
        candidate_mode="hybrid",
        view_candidate_k=2,
    )

    assert 5 not in spatial.neighbors[0]
    assert hybrid.neighbors[0, 0] == 5
    assert hybrid.diagnostics["candidate_mode"] == "hybrid"
    assert hybrid.diagnostics["selected_outside_spatial_fraction"] > 0.0
    assert hybrid.diagnostics["dense_anchor_pair_matrix_created"] is False


def test_hybrid_view_topology_is_camera_permutation_invariant():
    xyz = torch.tensor([
        [-0.90, 0.0, 0.50],
        [-0.85, 0.0, 1.00],
        [-0.80, 0.0, 1.00],
        [-0.75, 0.0, 1.00],
        [0.00, 0.0, 1.00],
        [0.90, 0.0, 0.50],
    ])
    cameras = [
        _camera("near", 0, zfar=0.75),
        _camera("far", 1, zfar=3.0),
    ]
    kwargs = {
        "candidate_k": 3,
        "topk": 2,
        "candidate_mode": "hybrid",
        "view_candidate_k": 2,
    }

    forward = build_view_topology_context(xyz, cameras, **kwargs)
    reverse = build_view_topology_context(xyz, list(reversed(cameras)), **kwargs)

    np.testing.assert_array_equal(forward.neighbors, reverse.neighbors)
    np.testing.assert_array_equal(forward.features, reverse.features)


def test_view_topology_camera_package_round_trip():
    cameras = [_camera("b", 1), _camera("a", 0)]
    state = camera_geometry_state(cameras)
    restored = camera_geometry_from_state(state)

    assert state["format"] == "packed_v1"
    assert state["full_proj_transform"].shape == (2, 4, 4)
    assert state["world_view_transform"].shape == (2, 4, 4)
    assert [camera.image_name for camera in restored] == ["a", "b"]
    for before, after in zip(sorted(cameras, key=lambda camera: camera.image_name), restored):
        torch.testing.assert_close(before.full_proj_transform, after.full_proj_transform)
        torch.testing.assert_close(before.world_view_transform, after.world_view_transform)


def test_legacy_camera_package_remains_decodable():
    cameras = [_camera("b", 1), _camera("a", 0)]
    legacy = tuple({
        "image_name": camera.image_name,
        "colmap_id": camera.colmap_id,
        "uid": camera.uid,
        "full_proj_transform": camera.full_proj_transform,
        "world_view_transform": camera.world_view_transform,
        "image_width": camera.image_width,
        "image_height": camera.image_height,
        "znear": camera.znear,
        "zfar": camera.zfar,
    } for camera in cameras)
    restored = camera_geometry_from_state(legacy)
    assert [camera.image_name for camera in restored] == ["a", "b"]


def test_distance_depth_fast_path_matches_full_geometric_scores():
    xyz = torch.tensor([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.8, 0.0, 0.5]])
    cameras = [_camera("a", 0), _camera("b", 1)]
    descriptors = build_geometric_observation_descriptors(xyz, cameras)
    source = np.asarray([0, 0, 1])
    target = np.asarray([1, 2, 2])

    full = pair_geometric_view_scores(descriptors, source, target, batch_size=2)
    fast = pair_distance_depth_scores(descriptors, source, target, batch_size=2)
    np.testing.assert_allclose(fast["geometric_distance"], full["geometric_distance"])
    np.testing.assert_allclose(fast["geometric_depth"], full["geometric_depth"])
    np.testing.assert_array_equal(fast["common_camera_count"], full["common_camera_count"])


def test_delayed_learning_rate_uses_absolute_endpoint():
    schedule = get_expon_lr_func(
        lr_init=1e-3,
        lr_final=1e-5,
        max_steps=30_000,
        step_sub=15_000,
    )

    assert np.isclose(schedule(1), 1e-3)
    assert np.isclose(schedule(15_000), 1e-3)
    assert np.isclose(schedule(30_000), 1e-5)
