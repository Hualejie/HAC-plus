"""Geometry-only camera/anchor observation and neighbour utilities.

This module is intentionally independent from the renderer and entropy model.
It mirrors the anchor filtering, quantisation, and canonical sorting performed by
``GaussianModel.conduct_encoding`` without changing codec behaviour.
"""

from dataclasses import dataclass
import heapq
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from utils.gpcc_utils import calculate_morton_order


@dataclass(frozen=True)
class CodecAlignedAnchors:
    xyz: torch.Tensor
    integer_xyz: torch.Tensor
    valid_original_idx: torch.Tensor
    codec_original_idx: torch.Tensor
    attributes: Dict[str, torch.Tensor]


@dataclass(frozen=True)
class ObservationRelation:
    """Sparse camera-to-anchor relation plus its anchor-to-camera CSR transpose."""

    camera_keys: Tuple[Tuple[str, int, int], ...]
    camera_to_anchor: Tuple[np.ndarray, ...]
    anchor_indptr: np.ndarray
    anchor_camera_ids: np.ndarray
    num_anchors: int


def canonicalize_codec_anchors(
    anchor: torch.Tensor,
    mask_anchor: torch.Tensor,
    voxel_size: float,
    attributes: Optional[Mapping[str, torch.Tensor]] = None,
) -> CodecAlignedAnchors:
    """Apply the exact valid-mask, quantisation, and sort used by the codec."""
    if anchor.ndim != 2 or anchor.shape[1] != 3:
        raise ValueError("anchor must have shape [N, 3]")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive for codec-equivalent analysis")

    mask = mask_anchor.reshape(-1).to(dtype=torch.bool, device=anchor.device)
    if mask.numel() != anchor.shape[0]:
        raise ValueError("mask_anchor length must equal the number of anchors")

    valid_original_idx = torch.nonzero(mask, as_tuple=False)[:, 0]
    integer_xyz = torch.round(anchor[mask] / voxel_size)
    codec_order = calculate_morton_order(integer_xyz)
    integer_xyz = integer_xyz[codec_order]
    codec_original_idx = valid_original_idx[codec_order]

    aligned_attributes: Dict[str, torch.Tensor] = {}
    for name, value in (attributes or {}).items():
        if value.shape[0] != anchor.shape[0]:
            raise ValueError(f"attribute {name!r} is not anchor-aligned")
        aligned_attributes[name] = value[mask][codec_order]

    return CodecAlignedAnchors(
        xyz=integer_xyz * voxel_size,
        integer_xyz=integer_xyz,
        valid_original_idx=valid_original_idx,
        codec_original_idx=codec_original_idx,
        attributes=aligned_attributes,
    )


def camera_sort_key(camera) -> Tuple[str, int, int]:
    """Stable camera identity used to make input permutation irrelevant."""
    return (
        str(getattr(camera, "image_name", "")),
        int(getattr(camera, "colmap_id", -1)),
        int(getattr(camera, "uid", -1)),
    )


def build_geometry_observations(
    codec_xyz: torch.Tensor,
    cameras: Sequence,
    chunk_size: int = 262_144,
    epsilon: float = 1e-7,
) -> ObservationRelation:
    """Project codec anchors into train cameras without rasterisation or occlusion.

    Only anchor xyz and camera geometry are read. Projection is chunked and only
    sparse CPU index arrays are retained; no dense camera-by-anchor GPU tensor is
    kept after a camera has been processed.
    """
    if codec_xyz.ndim != 2 or codec_xyz.shape[1] != 3:
        raise ValueError("codec_xyz must have shape [N, 3]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    sorted_cameras = sorted(cameras, key=camera_sort_key)
    keys = tuple(camera_sort_key(camera) for camera in sorted_cameras)
    if len(set(keys)) != len(keys):
        raise ValueError("camera stable keys must be unique")

    device = codec_xyz.device
    dtype = codec_xyz.dtype
    num_anchors = int(codec_xyz.shape[0])
    camera_to_anchor = []

    for camera in sorted_cameras:
        full_proj = torch.as_tensor(camera.full_proj_transform, device=device, dtype=dtype)
        world_view = torch.as_tensor(camera.world_view_transform, device=device, dtype=dtype)
        width = int(camera.image_width)
        height = int(camera.image_height)
        znear = float(camera.znear)
        zfar = float(camera.zfar)
        observed_chunks = []

        for start in range(0, num_anchors, chunk_size):
            end = min(start + chunk_size, num_anchors)
            xyz = codec_xyz[start:end]
            ones = torch.ones((xyz.shape[0], 1), device=device, dtype=dtype)
            xyz_h = torch.cat((xyz, ones), dim=1)
            clip = xyz_h @ full_proj
            camera_space = xyz_h @ world_view
            w = clip[:, 3]
            safe_w = torch.where(w.abs() > epsilon, w, torch.ones_like(w))
            ndc = clip[:, :3] / safe_w[:, None]

            # Pixel bounds are equivalent to NDC [-1, 1] while explicitly using
            # the camera image dimensions from the observation contract.
            pixel_x = ((ndc[:, 0] + 1.0) * width - 1.0) * 0.5
            pixel_y = ((ndc[:, 1] + 1.0) * height - 1.0) * 0.5
            visible = (
                (w > epsilon)
                & (camera_space[:, 2] >= znear)
                & (camera_space[:, 2] <= zfar)
                & (ndc[:, 2] >= 0.0)
                & (ndc[:, 2] <= 1.0)
                & (pixel_x >= -0.5)
                & (pixel_x <= width - 0.5)
                & (pixel_y >= -0.5)
                & (pixel_y <= height - 0.5)
            )
            local = torch.nonzero(visible, as_tuple=False)[:, 0]
            if local.numel():
                observed_chunks.append((local + start).cpu().numpy().astype(np.int64, copy=False))

        if observed_chunks:
            camera_to_anchor.append(np.concatenate(observed_chunks))
        else:
            camera_to_anchor.append(np.empty(0, dtype=np.int64))

    if camera_to_anchor:
        edge_anchor = np.concatenate(camera_to_anchor)
        edge_camera = np.concatenate(
            [np.full(indices.size, camera_id, dtype=np.int32)
             for camera_id, indices in enumerate(camera_to_anchor)]
        )
        edge_order = np.lexsort((edge_camera, edge_anchor))
        edge_anchor = edge_anchor[edge_order]
        anchor_camera_ids = edge_camera[edge_order]
        counts = np.bincount(edge_anchor, minlength=num_anchors)
    else:
        anchor_camera_ids = np.empty(0, dtype=np.int32)
        counts = np.zeros(num_anchors, dtype=np.int64)

    anchor_indptr = np.empty(num_anchors + 1, dtype=np.int64)
    anchor_indptr[0] = 0
    np.cumsum(counts, out=anchor_indptr[1:])
    return ObservationRelation(
        camera_keys=keys,
        camera_to_anchor=tuple(camera_to_anchor),
        anchor_indptr=anchor_indptr,
        anchor_camera_ids=anchor_camera_ids,
        num_anchors=num_anchors,
    )


def observation_signatures(
    relation: ObservationRelation,
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], np.ndarray]:
    """Group anchors that have identical sparse camera observation sets."""
    signature_to_group = {}
    group_of_anchor = np.empty(relation.num_anchors, dtype=np.int64)
    members = []
    signatures = []

    for anchor_idx in range(relation.num_anchors):
        begin = relation.anchor_indptr[anchor_idx]
        end = relation.anchor_indptr[anchor_idx + 1]
        signature = tuple(int(x) for x in relation.anchor_camera_ids[begin:end])
        group_idx = signature_to_group.get(signature)
        if group_idx is None:
            group_idx = len(signatures)
            signature_to_group[signature] = group_idx
            signatures.append(signature)
            members.append([])
        group_of_anchor[anchor_idx] = group_idx
        members[group_idx].append(anchor_idx)

    group_members = tuple(np.asarray(group, dtype=np.int64) for group in members)
    incidence = np.zeros((len(signatures), len(relation.camera_keys)), dtype=np.uint8)
    for group_idx, signature in enumerate(signatures):
        if signature:
            incidence[group_idx, np.asarray(signature, dtype=np.int64)] = 1
    return group_of_anchor, group_members, incidence


def _smallest_group_members(
    group_ids: np.ndarray,
    group_members: Tuple[np.ndarray, ...],
    limit: int,
) -> np.ndarray:
    heap = []
    for group_idx in group_ids:
        member = group_members[int(group_idx)]
        if member.size:
            heapq.heappush(heap, (int(member[0]), int(group_idx), 0))
    selected = []
    while heap and len(selected) < limit:
        anchor_idx, group_idx, offset = heapq.heappop(heap)
        selected.append(anchor_idx)
        next_offset = offset + 1
        member = group_members[group_idx]
        if next_offset < member.size:
            heapq.heappush(heap, (int(member[next_offset]), group_idx, next_offset))
    return np.asarray(selected, dtype=np.int64)


def coview_topk(
    relation: ObservationRelation,
    k: int = 8,
    score_block_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Return deterministic Jaccard Top-K without constructing an N-by-N matrix."""
    if k <= 0 or k >= relation.num_anchors:
        raise ValueError("k must satisfy 0 < k < num_anchors")
    if score_block_size <= 0:
        raise ValueError("score_block_size must be positive")

    group_of_anchor, group_members, incidence = observation_signatures(relation)
    num_groups = len(group_members)
    group_sizes = np.asarray([group.size for group in group_members], dtype=np.int64)
    signature_sizes = incidence.sum(axis=1, dtype=np.int32)
    neighbors = np.empty((relation.num_anchors, k), dtype=np.int64)
    neighbor_scores = np.empty((relation.num_anchors, k), dtype=np.float32)
    max_shape = (0, num_groups)

    for block_start in range(0, num_groups, score_block_size):
        block_end = min(block_start + score_block_size, num_groups)
        intersection = incidence[block_start:block_end].astype(np.int32) @ incidence.T.astype(np.int32)
        union = (
            signature_sizes[block_start:block_end, None]
            + signature_sizes[None, :]
            - intersection
        )
        score_block = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0,
        )
        max_shape = max(max_shape, score_block.shape, key=lambda shape: shape[0] * shape[1])

        for local_idx, source_group in enumerate(range(block_start, block_end)):
            scores = score_block[local_idx]
            group_order = np.argsort(-scores, kind="stable")
            ranked_candidates = []
            ranked_scores = []
            cursor = 0
            needed = k + 1  # include the source anchor, then filter it per row
            while cursor < num_groups and len(ranked_candidates) < needed:
                level_score = scores[group_order[cursor]]
                level_end = cursor + 1
                while level_end < num_groups and scores[group_order[level_end]] == level_score:
                    level_end += 1
                level_groups = group_order[cursor:level_end]
                take = min(needed - len(ranked_candidates), int(group_sizes[level_groups].sum()))
                level_members = _smallest_group_members(level_groups, group_members, take)
                ranked_candidates.extend(level_members.tolist())
                ranked_scores.extend([float(level_score)] * len(level_members))
                cursor = level_end

            base_indices = np.asarray(ranked_candidates, dtype=np.int64)
            base_scores = np.asarray(ranked_scores, dtype=np.float32)
            for source_anchor in group_members[source_group]:
                keep = base_indices != source_anchor
                chosen = base_indices[keep][:k]
                chosen_scores = base_scores[keep][:k]
                if chosen.size != k:
                    raise RuntimeError("failed to produce k non-self CoView neighbours")
                neighbors[source_anchor] = chosen
                neighbor_scores[source_anchor] = chosen_scores

    diagnostics = {
        "num_anchors": relation.num_anchors,
        "num_cameras": len(relation.camera_keys),
        "num_edges": int(relation.anchor_camera_ids.size),
        "num_unique_signatures": num_groups,
        "max_score_block_shape": list(max_shape),
        "dense_anchor_pair_matrix_created": False,
    }
    return neighbors, neighbor_scores, diagnostics


def spatial_topk(
    xyz: np.ndarray,
    k: int = 8,
    query_chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic Euclidean Top-K, including exact boundary ties."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if k <= 0 or k >= xyz.shape[0]:
        raise ValueError("k must satisfy 0 < k < num_anchors")

    tree = cKDTree(xyz)
    initial_dist, _ = tree.query(xyz, k=k + 1, workers=-1)
    boundary = initial_dist[:, k]
    neighbors = np.empty((xyz.shape[0], k), dtype=np.int64)
    distances = np.empty((xyz.shape[0], k), dtype=np.float64)

    for start in range(0, xyz.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, xyz.shape[0])
        radii = boundary[start:end] + np.maximum(1.0, boundary[start:end]) * 1e-12
        candidate_lists = tree.query_ball_point(xyz[start:end], radii, workers=-1)
        for local_idx, candidates in enumerate(candidate_lists):
            anchor_idx = start + local_idx
            candidate_idx = np.asarray(candidates, dtype=np.int64)
            candidate_idx = candidate_idx[candidate_idx != anchor_idx]
            delta = xyz[candidate_idx] - xyz[anchor_idx]
            candidate_dist = np.linalg.norm(delta, axis=1)
            order = np.lexsort((candidate_idx, candidate_dist))[:k]
            if order.size != k:
                raise RuntimeError("failed to produce k spatial neighbours")
            neighbors[anchor_idx] = candidate_idx[order]
            distances[anchor_idx] = candidate_dist[order]
    return neighbors, distances


def pair_coview_scores(
    relation: ObservationRelation,
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Jaccard scores for arbitrary anchor pairs from sparse signatures."""
    group_of_anchor, _, incidence = observation_signatures(relation)
    source_groups = group_of_anchor[np.asarray(source, dtype=np.int64)]
    target_groups = group_of_anchor[np.asarray(target, dtype=np.int64)]
    source_incidence = incidence[source_groups]
    target_incidence = incidence[target_groups]
    intersection = np.logical_and(source_incidence, target_incidence).sum(axis=1)
    union = np.logical_or(source_incidence, target_incidence).sum(axis=1)
    return np.divide(
        intersection,
        union,
        out=np.zeros(intersection.shape, dtype=np.float32),
        where=union > 0,
    )
