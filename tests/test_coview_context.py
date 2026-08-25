from types import SimpleNamespace

import numpy as np
import torch

from scene.coview_context import (
    ObservationRelation,
    build_geometry_observations,
    canonicalize_codec_anchors,
    coview_topk,
    spatial_topk,
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
