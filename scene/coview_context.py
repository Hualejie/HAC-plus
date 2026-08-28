"""Geometry-only camera/anchor observation and neighbour utilities.

This module is intentionally independent from the renderer and entropy model.
It mirrors the anchor filtering, quantisation, and canonical sorting performed by
``GaussianModel.conduct_encoding`` without changing codec behaviour.
"""

from dataclasses import dataclass
import hashlib
import heapq
import struct
from types import SimpleNamespace
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


@dataclass(frozen=True)
class GeometricObservationDescriptors:
    """Decoder-reconstructable geometry stored in Anchor→Camera CSR edge order."""

    relation: ObservationRelation
    distance: np.ndarray
    view_direction: np.ndarray
    image_xy: np.ndarray
    depth: np.ndarray


@dataclass(frozen=True)
class ViewTopologyContext:
    """Fixed-width topology features derived from a sparse anchor graph."""

    features: np.ndarray
    neighbors: np.ndarray
    distance_scores: np.ndarray
    depth_scores: np.ndarray
    diagnostics: Dict[str, object]


VIEW_TOPOLOGY_FEATURE_DIM = 15
VIEW_TOPOLOGY_CANDIDATE_MODES = ("spatial", "hybrid")
CAMERA_GEOMETRY_MAGIC = b"CVCAM001"
CAMERA_PROTOTYPE_SELECTION = "canonical_uniform_v1"


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


def extract_camera_geometry(cameras: Sequence) -> Tuple[SimpleNamespace, ...]:
    """Detach the small decoder-side camera contract from image-bearing cameras."""
    geometry = []
    for camera in sorted(cameras, key=camera_sort_key):
        geometry.append(SimpleNamespace(
            image_name=str(getattr(camera, "image_name", "")),
            colmap_id=int(getattr(camera, "colmap_id", -1)),
            uid=int(getattr(camera, "uid", -1)),
            full_proj_transform=torch.as_tensor(camera.full_proj_transform).detach().cpu(),
            world_view_transform=torch.as_tensor(camera.world_view_transform).detach().cpu(),
            image_width=int(camera.image_width),
            image_height=int(camera.image_height),
            znear=float(camera.znear),
            zfar=float(camera.zfar),
        ))
    return tuple(geometry)


def select_camera_prototypes(
    cameras: Sequence,
    count: int = 0,
) -> Tuple[SimpleNamespace, ...]:
    """Select an input-order-invariant uniform subset of canonical cameras.

    ``count == 0`` preserves all cameras.  A positive count samples inclusive
    endpoints from the camera-sort-key order, matching the c4 Frozen studies.
    """
    geometry = extract_camera_geometry(cameras)
    count = int(count)
    if count < 0:
        raise ValueError("camera prototype count must be non-negative")
    if not geometry:
        if count:
            raise ValueError("cannot select camera prototypes from an empty set")
        return geometry
    if count == 0 or count >= len(geometry):
        return geometry
    indices = np.linspace(0, len(geometry) - 1, count, dtype=np.int64)
    return tuple(geometry[int(index)] for index in indices)


def camera_geometry_state(cameras: Sequence) -> Dict[str, object]:
    """Pack canonical camera geometry into one float32 tensor."""
    geometry = extract_camera_geometry(cameras)
    if not geometry:
        raise ValueError("at least one camera is required for CoView geometry")
    rows = []
    for camera in geometry:
        rows.append(torch.cat((
            camera.full_proj_transform.to(dtype=torch.float32).reshape(-1),
            camera.world_view_transform.to(dtype=torch.float32).reshape(-1),
            torch.tensor([
                camera.image_width,
                camera.image_height,
                camera.znear,
                camera.zfar,
            ], dtype=torch.float32),
        )))
    return {
        "format": "packed_v2",
        "data": torch.stack(rows),
    }


def camera_geometry_from_state(state) -> Tuple[SimpleNamespace, ...]:
    """Restore packed-v2, packed-v1, or legacy camera geometry packages."""
    if isinstance(state, Mapping) and state.get("format") == "packed_v2":
        data = torch.as_tensor(state["data"])
        if data.ndim != 2 or data.shape[1] != 36 or data.shape[0] == 0:
            raise ValueError(
                "packed camera geometry data must have shape [M, 36] with M > 0"
            )
        cameras = [SimpleNamespace(
            image_name=f"camera_{index:08d}",
            colmap_id=index,
            uid=index,
            full_proj_transform=data[index, :16].reshape(4, 4),
            world_view_transform=data[index, 16:32].reshape(4, 4),
            image_width=int(data[index, 32]),
            image_height=int(data[index, 33]),
            znear=float(data[index, 34]),
            zfar=float(data[index, 35]),
        ) for index in range(data.shape[0])]
        return extract_camera_geometry(cameras)
    if isinstance(state, Mapping) and state.get("format") == "packed_v1":
        names = tuple(state["image_names"])
        colmap_ids = torch.as_tensor(state["colmap_ids"])
        uids = torch.as_tensor(state["uids"])
        full_proj = torch.as_tensor(state["full_proj_transform"])
        world_view = torch.as_tensor(state["world_view_transform"])
        image_size = torch.as_tensor(state["image_size"])
        clip = torch.as_tensor(state["clip"])
        count = len(names)
        expected_shapes = {
            "colmap_ids": (count,),
            "uids": (count,),
            "full_proj_transform": (count, 4, 4),
            "world_view_transform": (count, 4, 4),
            "image_size": (count, 2),
            "clip": (count, 2),
        }
        values = {
            "colmap_ids": colmap_ids,
            "uids": uids,
            "full_proj_transform": full_proj,
            "world_view_transform": world_view,
            "image_size": image_size,
            "clip": clip,
        }
        for name, expected in expected_shapes.items():
            if tuple(values[name].shape) != expected:
                raise ValueError(
                    f"packed camera geometry {name} has shape "
                    f"{tuple(values[name].shape)}, expected {expected}"
                )
        cameras = [SimpleNamespace(
            image_name=str(names[index]),
            colmap_id=int(colmap_ids[index]),
            uid=int(uids[index]),
            full_proj_transform=full_proj[index],
            world_view_transform=world_view[index],
            image_width=int(image_size[index, 0]),
            image_height=int(image_size[index, 1]),
            znear=float(clip[index, 0]),
            zfar=float(clip[index, 1]),
        ) for index in range(count)]
        return extract_camera_geometry(cameras)
    return extract_camera_geometry([
        SimpleNamespace(**dict(item)) for item in state
    ])


def serialize_camera_geometry(cameras: Sequence, path) -> Dict[str, object]:
    """Write canonical camera geometry without pickle or tensor-header overhead."""
    geometry = extract_camera_geometry(cameras)
    if not geometry:
        raise ValueError("at least one camera is required for CoView geometry")
    payload = bytearray(struct.pack("<8sI", CAMERA_GEOMETRY_MAGIC, len(geometry)))
    for camera in geometry:
        for matrix in (
            camera.full_proj_transform,
            camera.world_view_transform,
        ):
            array = np.asarray(
                torch.as_tensor(matrix).detach().cpu(), dtype="<f4"
            ).reshape(4, 4)
            payload.extend(array.tobytes(order="C"))
        payload.extend(struct.pack(
            "<IIdd",
            camera.image_width,
            camera.image_height,
            camera.znear,
            camera.zfar,
        ))
    payload = bytes(payload)
    with open(path, "wb") as handle:
        handle.write(payload)
    return {
        "format": "raw_v1",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "count": len(geometry),
    }


def deserialize_camera_geometry(path, metadata) -> Tuple[SimpleNamespace, ...]:
    """Read and verify a raw-v1 decoder camera package."""
    with open(path, "rb") as handle:
        payload = handle.read()
    if metadata.get("format") != "raw_v1":
        raise RuntimeError(
            f"unsupported camera geometry format {metadata.get('format')!r}"
        )
    actual = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    for key, value in actual.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"camera geometry {key} mismatch: {value!r} != {metadata.get(key)!r}"
            )
    header_size = struct.calcsize("<8sI")
    if len(payload) < header_size:
        raise RuntimeError("camera geometry payload is truncated")
    magic, count = struct.unpack_from("<8sI", payload, 0)
    if magic != CAMERA_GEOMETRY_MAGIC:
        raise RuntimeError("camera geometry magic mismatch")
    if count != metadata.get("count"):
        raise RuntimeError(
            f"camera geometry count mismatch: {count} != {metadata.get('count')}"
        )
    matrix_bytes = 16 * np.dtype("<f4").itemsize
    scalar_size = struct.calcsize("<IIdd")
    record_size = 2 * matrix_bytes + scalar_size
    if len(payload) != header_size + count * record_size:
        raise RuntimeError("camera geometry payload length is inconsistent")
    cameras = []
    offset = header_size
    for index in range(count):
        matrices = []
        for _ in range(2):
            array = np.frombuffer(
                payload, dtype="<f4", count=16, offset=offset
            ).copy().reshape(4, 4)
            matrices.append(torch.from_numpy(array))
            offset += matrix_bytes
        width, height, znear, zfar = struct.unpack_from(
            "<IIdd", payload, offset
        )
        offset += scalar_size
        cameras.append(SimpleNamespace(
            image_name=f"camera_{index:08d}",
            colmap_id=index,
            uid=index,
            full_proj_transform=matrices[0],
            world_view_transform=matrices[1],
            image_width=width,
            image_height=height,
            znear=znear,
            zfar=zfar,
        ))
    return extract_camera_geometry(cameras)


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


def coview_lsh_candidates(
    relation: ObservationRelation,
    k: int = 16,
    num_hashes: int = 8,
    band_size: int = 2,
    bucket_window: Optional[int] = None,
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Dict[str, object]]:
    """Generate bounded non-local CoView candidates with deterministic MinHash.

    The exact global Jaccard Top-K above is useful for analysis but has
    quadratic total work in the number of unique observation signatures. This
    codec path instead applies MinHash LSH to sparse camera sets. Only a bounded
    canonical-index window in each matching band is evaluated with exact
    Jaccard, so neither resident memory nor pair work is anchor-square.

    Rows may contain fewer than ``k`` candidates. The hybrid topology always
    unions them with at least ``topk`` Euclidean candidates before selection.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if num_hashes <= 0 or band_size <= 0 or num_hashes % band_size:
        raise ValueError("num_hashes must be positive and divisible by band_size")
    num_bands = num_hashes // band_size
    if bucket_window is None:
        bucket_window = max(2, int(np.ceil(k / num_bands)))
    if bucket_window <= 0:
        raise ValueError("bucket_window must be positive")

    group_of_anchor, group_members, incidence = observation_signatures(relation)
    num_groups = len(group_members)
    prime = np.int64(2_147_483_647)
    multipliers = np.asarray(
        [1_103_515_245, 1_664_525, 22_695_477, 214_013,
         1_347_758_131, 48_271, 69_069, 16_807],
        dtype=np.int64,
    )
    offsets = np.asarray(
        [12_345, 1_013_904_223, 1, 2_531_011,
         668_265_263, 0, 362_437, 97_531],
        dtype=np.int64,
    )
    if num_hashes > multipliers.size:
        raise ValueError(f"num_hashes cannot exceed {multipliers.size}")

    minhash = np.full((num_groups, num_hashes), prime, dtype=np.int64)
    for group_idx in range(num_groups):
        camera_ids = np.flatnonzero(incidence[group_idx]).astype(np.int64) + 1
        if camera_ids.size:
            hashed = (
                camera_ids[:, None] * multipliers[None, :num_hashes]
                + offsets[None, :num_hashes]
            ) % prime
            minhash[group_idx] = hashed.min(axis=0)

    anchor_ids = np.arange(relation.num_anchors, dtype=np.int64)
    candidate_sets = [set() for _ in range(relation.num_anchors)]
    for band_idx in range(num_bands):
        start = band_idx * band_size
        end = start + band_size
        keys = minhash[group_of_anchor, start:end]
        # key column 0 is primary, followed by the remaining columns and the
        # canonical anchor index for a deterministic within-bucket order.
        lex_keys = (anchor_ids,) + tuple(
            keys[:, axis] for axis in reversed(range(band_size))
        )
        order = np.lexsort(lex_keys)
        sorted_keys = keys[order]
        boundaries = np.flatnonzero(
            np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
        ) + 1
        bucket_starts = np.concatenate((np.asarray([0]), boundaries))
        bucket_ends = np.concatenate((boundaries, np.asarray([order.size])))

        for bucket_start, bucket_end in zip(bucket_starts, bucket_ends):
            for position in range(int(bucket_start), int(bucket_end)):
                anchor_idx = int(order[position])
                begin = max(int(bucket_start), position - bucket_window)
                finish = min(int(bucket_end), position + bucket_window + 1)
                candidate_sets[anchor_idx].update(
                    int(value) for value in order[begin:finish]
                    if int(value) != anchor_idx
                )

    candidate_rows = []
    score_rows = []
    raw_counts = np.empty(relation.num_anchors, dtype=np.int64)
    for anchor_idx, candidate_set in enumerate(candidate_sets):
        candidates = np.asarray(sorted(candidate_set), dtype=np.int64)
        raw_counts[anchor_idx] = candidates.size
        if candidates.size:
            source_incidence = incidence[group_of_anchor[anchor_idx]]
            target_incidence = incidence[group_of_anchor[candidates]]
            intersection = np.logical_and(
                target_incidence,
                source_incidence[None, :],
            ).sum(axis=1)
            union = np.logical_or(
                target_incidence,
                source_incidence[None, :],
            ).sum(axis=1)
            scores = np.divide(
                intersection,
                union,
                out=np.zeros(intersection.shape, dtype=np.float32),
                where=union > 0,
            )
            selected = np.lexsort((candidates, -scores))[:k]
            candidates = candidates[selected]
            scores = scores[selected]
        else:
            scores = np.empty(0, dtype=np.float32)
        candidate_rows.append(candidates)
        score_rows.append(scores.astype(np.float32, copy=False))

    retained_counts = np.asarray([row.size for row in candidate_rows])
    diagnostics = {
        "num_anchors": relation.num_anchors,
        "num_cameras": len(relation.camera_keys),
        "num_edges": int(relation.anchor_camera_ids.size),
        "num_unique_signatures": num_groups,
        "num_hashes": num_hashes,
        "band_size": band_size,
        "num_bands": num_bands,
        "bucket_window": bucket_window,
        "mean_raw_candidate_count": float(raw_counts.mean()),
        "max_raw_candidate_count": int(raw_counts.max()),
        "mean_retained_candidate_count": float(retained_counts.mean()),
        "max_retained_candidate_count": int(retained_counts.max()),
        "dense_anchor_pair_matrix_created": False,
        "quadratic_pair_enumeration": False,
    }
    return tuple(candidate_rows), tuple(score_rows), diagnostics


def spatial_topk(
    xyz: np.ndarray,
    k: int = 8,
    query_chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic Euclidean Top-K, including exact boundary ties."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return spatial_topk_queries(
        xyz,
        np.arange(xyz.shape[0], dtype=np.int64),
        k=k,
        query_chunk_size=query_chunk_size,
    )


def spatial_topk_queries(
    xyz: np.ndarray,
    query_indices: np.ndarray,
    k: int = 8,
    query_chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic Euclidean Top-K for a bounded anchor subset."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64)
    query_indices = np.asarray(query_indices, dtype=np.int64).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if k <= 0 or k >= xyz.shape[0]:
        raise ValueError("k must satisfy 0 < k < num_anchors")
    if query_indices.size and (query_indices.min() < 0 or query_indices.max() >= xyz.shape[0]):
        raise ValueError("query index outside xyz")

    tree = cKDTree(xyz)
    query_xyz = xyz[query_indices]
    initial_dist, _ = tree.query(query_xyz, k=k + 1, workers=-1)
    boundary = initial_dist[:, k]
    neighbors = np.empty((query_indices.size, k), dtype=np.int64)
    distances = np.empty((query_indices.size, k), dtype=np.float64)

    for start in range(0, query_indices.size, query_chunk_size):
        end = min(start + query_chunk_size, query_indices.size)
        radii = boundary[start:end] + np.maximum(1.0, boundary[start:end]) * 1e-12
        candidate_lists = tree.query_ball_point(query_xyz[start:end], radii, workers=-1)
        for local_idx, candidates in enumerate(candidate_lists):
            query_row = start + local_idx
            anchor_idx = query_indices[query_row]
            candidate_idx = np.asarray(candidates, dtype=np.int64)
            candidate_idx = candidate_idx[candidate_idx != anchor_idx]
            delta = xyz[candidate_idx] - xyz[anchor_idx]
            candidate_dist = np.linalg.norm(delta, axis=1)
            order = np.lexsort((candidate_idx, candidate_dist))[:k]
            if order.size != k:
                raise RuntimeError("failed to produce k spatial neighbours")
            neighbors[query_row] = candidate_idx[order]
            distances[query_row] = candidate_dist[order]
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


def build_geometric_observation_descriptors(
    codec_xyz: torch.Tensor,
    cameras: Sequence,
    relation: Optional[ObservationRelation] = None,
) -> GeometricObservationDescriptors:
    """Compute sparse per-observation geometry without appearance or rasterisation.

    Descriptors contain camera-anchor distance, world-space unit viewing
    direction, normalized image coordinates (NDC x/y), and camera-space depth.
    All values depend only on codec xyz and train-camera geometry.
    """
    if codec_xyz.ndim != 2 or codec_xyz.shape[1] != 3:
        raise ValueError("codec_xyz must have shape [N, 3]")
    if relation is None:
        relation = build_geometry_observations(codec_xyz, cameras)
    if relation.num_anchors != codec_xyz.shape[0]:
        raise ValueError("relation and codec_xyz anchor counts differ")

    sorted_cameras = sorted(cameras, key=camera_sort_key)
    keys = tuple(camera_sort_key(camera) for camera in sorted_cameras)
    if keys != relation.camera_keys:
        raise ValueError("camera set does not match the observation relation")

    device = codec_xyz.device
    dtype = codec_xyz.dtype
    edge_anchor_parts = []
    edge_camera_parts = []
    distance_parts = []
    direction_parts = []
    image_xy_parts = []
    depth_parts = []

    for camera_id, (camera, anchor_indices) in enumerate(
        zip(sorted_cameras, relation.camera_to_anchor)
    ):
        if anchor_indices.size == 0:
            continue
        index_tensor = torch.as_tensor(anchor_indices, device=device, dtype=torch.long)
        xyz = codec_xyz[index_tensor]
        ones = torch.ones((xyz.shape[0], 1), device=device, dtype=dtype)
        xyz_h = torch.cat((xyz, ones), dim=1)
        world_view = torch.as_tensor(camera.world_view_transform, device=device, dtype=dtype)
        full_proj = torch.as_tensor(camera.full_proj_transform, device=device, dtype=dtype)
        camera_center = torch.linalg.inv(world_view)[3, :3]

        camera_delta = xyz - camera_center[None, :]
        distance = torch.linalg.norm(camera_delta, dim=1)
        direction = camera_delta / torch.clamp(distance[:, None], min=1e-12)
        camera_space = xyz_h @ world_view
        clip = xyz_h @ full_proj
        image_xy = clip[:, :2] / torch.clamp(clip[:, 3:4], min=1e-12)

        edge_anchor_parts.append(anchor_indices.astype(np.int64, copy=False))
        edge_camera_parts.append(np.full(anchor_indices.size, camera_id, dtype=np.int32))
        distance_parts.append(distance.cpu().numpy().astype(np.float32, copy=False))
        direction_parts.append(direction.cpu().numpy().astype(np.float32, copy=False))
        image_xy_parts.append(image_xy.cpu().numpy().astype(np.float32, copy=False))
        depth_parts.append(camera_space[:, 2].cpu().numpy().astype(np.float32, copy=False))

    if not edge_anchor_parts:
        empty = np.empty(0, dtype=np.float32)
        return GeometricObservationDescriptors(
            relation=relation,
            distance=empty,
            view_direction=np.empty((0, 3), dtype=np.float32),
            image_xy=np.empty((0, 2), dtype=np.float32),
            depth=empty,
        )

    edge_anchor = np.concatenate(edge_anchor_parts)
    edge_camera = np.concatenate(edge_camera_parts)
    edge_order = np.lexsort((edge_camera, edge_anchor))
    ordered_camera = edge_camera[edge_order]
    if not np.array_equal(ordered_camera, relation.anchor_camera_ids):
        raise RuntimeError("descriptor edge order does not match relation CSR")

    return GeometricObservationDescriptors(
        relation=relation,
        distance=np.concatenate(distance_parts)[edge_order],
        view_direction=np.concatenate(direction_parts, axis=0)[edge_order],
        image_xy=np.concatenate(image_xy_parts, axis=0)[edge_order],
        depth=np.concatenate(depth_parts)[edge_order],
    )


def pair_geometric_view_scores(
    descriptors: GeometricObservationDescriptors,
    source: np.ndarray,
    target: np.ndarray,
    image_sigma: float = 0.25,
    batch_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """Compare observation geometry for arbitrary anchor pairs.

    Each component is averaged over common cameras. Distance and depth use a
    relative Laplacian kernel, direction uses rescaled cosine similarity, and
    normalized image coordinates use an RBF kernel. ``geometric_composite`` is
    their equal-weight mean. Sparse CSR matrices are densified only for bounded
    pair batches, never for all anchors and cameras.
    """
    from scipy.sparse import csr_matrix

    source = np.asarray(source, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape")
    if image_sigma <= 0 or batch_size <= 0:
        raise ValueError("image_sigma and batch_size must be positive")

    relation = descriptors.relation
    shape = (relation.num_anchors, len(relation.camera_keys))
    indices = relation.anchor_camera_ids
    indptr = relation.anchor_indptr

    def sparse(values):
        return csr_matrix((values, indices, indptr), shape=shape, copy=False)

    incidence = sparse(np.ones(indices.size, dtype=np.float32))
    distance = sparse(descriptors.distance)
    direction = tuple(sparse(descriptors.view_direction[:, axis]) for axis in range(3))
    image_xy = tuple(sparse(descriptors.image_xy[:, axis]) for axis in range(2))
    depth = sparse(descriptors.depth)

    names = (
        "binary_jaccard",
        "geometric_distance",
        "geometric_direction",
        "geometric_image",
        "geometric_depth",
        "geometric_composite",
        "common_camera_count",
    )
    output = {name: np.empty(source.size, dtype=np.float32) for name in names}
    epsilon = 1e-8

    for start in range(0, source.size, batch_size):
        end = min(start + batch_size, source.size)
        src = source[start:end]
        dst = target[start:end]
        src_mask = incidence[src].toarray() > 0
        dst_mask = incidence[dst].toarray() > 0
        common = src_mask & dst_mask
        common_count = common.sum(axis=1)
        union_count = (src_mask | dst_mask).sum(axis=1)
        denominator = np.maximum(common_count, 1)

        src_distance = distance[src].toarray()
        dst_distance = distance[dst].toarray()
        distance_scale = 0.5 * (np.abs(src_distance) + np.abs(dst_distance)) + epsilon
        distance_kernel = np.exp(-np.abs(src_distance - dst_distance) / distance_scale)
        distance_score = (distance_kernel * common).sum(axis=1) / denominator

        direction_dot = np.zeros(common.shape, dtype=np.float32)
        for matrix in direction:
            direction_dot += matrix[src].toarray() * matrix[dst].toarray()
        direction_kernel = 0.5 * (np.clip(direction_dot, -1.0, 1.0) + 1.0)
        direction_score = (direction_kernel * common).sum(axis=1) / denominator

        image_sq_distance = np.zeros(common.shape, dtype=np.float32)
        for matrix in image_xy:
            delta = matrix[src].toarray() - matrix[dst].toarray()
            image_sq_distance += delta * delta
        image_kernel = np.exp(-image_sq_distance / (image_sigma * image_sigma))
        image_score = (image_kernel * common).sum(axis=1) / denominator

        src_depth = depth[src].toarray()
        dst_depth = depth[dst].toarray()
        depth_scale = 0.5 * (np.abs(src_depth) + np.abs(dst_depth)) + epsilon
        depth_kernel = np.exp(-np.abs(src_depth - dst_depth) / depth_scale)
        depth_score = (depth_kernel * common).sum(axis=1) / denominator

        valid = common_count > 0
        component_scores = (distance_score, direction_score, image_score, depth_score)
        for values in component_scores:
            values[~valid] = 0.0
        composite = np.mean(np.stack(component_scores, axis=1), axis=1)

        output["binary_jaccard"][start:end] = np.divide(
            common_count,
            union_count,
            out=np.zeros(common_count.shape, dtype=np.float32),
            where=union_count > 0,
        )
        output["geometric_distance"][start:end] = distance_score
        output["geometric_direction"][start:end] = direction_score
        output["geometric_image"][start:end] = image_score
        output["geometric_depth"][start:end] = depth_score
        output["geometric_composite"][start:end] = composite
        output["common_camera_count"][start:end] = common_count
    return output


def pair_distance_depth_scores(
    descriptors: GeometricObservationDescriptors,
    source: np.ndarray,
    target: np.ndarray,
    batch_size: int = 8192,
) -> Dict[str, np.ndarray]:
    """Evaluate only the two Phase 1.5-supported edge components.

    This avoids materializing the unused direction and image-plane kernels when
    building the codec graph. Dense arrays are bounded by ``batch_size`` times
    the camera count; an anchor-square matrix is never constructed.
    """
    from scipy.sparse import csr_matrix

    source = np.asarray(source, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    relation = descriptors.relation
    shape = (relation.num_anchors, len(relation.camera_keys))
    indices = relation.anchor_camera_ids
    indptr = relation.anchor_indptr

    def sparse(values):
        return csr_matrix((values, indices, indptr), shape=shape, copy=False)

    incidence = sparse(np.ones(indices.size, dtype=np.float32))
    distance = sparse(descriptors.distance)
    depth = sparse(descriptors.depth)
    output = {
        "geometric_distance": np.empty(source.size, dtype=np.float32),
        "geometric_depth": np.empty(source.size, dtype=np.float32),
        "common_camera_count": np.empty(source.size, dtype=np.float32),
    }
    epsilon = 1e-8

    for start in range(0, source.size, batch_size):
        end = min(start + batch_size, source.size)
        src = source[start:end]
        dst = target[start:end]
        src_mask = incidence[src].toarray() > 0
        dst_mask = incidence[dst].toarray() > 0
        common = src_mask & dst_mask
        common_count = common.sum(axis=1)
        denominator = np.maximum(common_count, 1)

        src_distance = distance[src].toarray()
        dst_distance = distance[dst].toarray()
        distance_scale = 0.5 * (np.abs(src_distance) + np.abs(dst_distance)) + epsilon
        distance_kernel = np.exp(-np.abs(src_distance - dst_distance) / distance_scale)
        distance_score = (distance_kernel * common).sum(axis=1) / denominator

        src_depth = depth[src].toarray()
        dst_depth = depth[dst].toarray()
        depth_scale = 0.5 * (np.abs(src_depth) + np.abs(dst_depth)) + epsilon
        depth_kernel = np.exp(-np.abs(src_depth - dst_depth) / depth_scale)
        depth_score = (depth_kernel * common).sum(axis=1) / denominator

        invalid = common_count == 0
        distance_score[invalid] = 0.0
        depth_score[invalid] = 0.0
        output["geometric_distance"][start:end] = distance_score
        output["geometric_depth"][start:end] = depth_score
        output["common_camera_count"][start:end] = common_count
    return output


def build_view_topology_context(
    codec_xyz: torch.Tensor,
    cameras: Sequence,
    candidate_k: int = 16,
    topk: int = 8,
    candidate_mode: str = "spatial",
    view_candidate_k: int = 16,
    feature_quantization: float = 1e-5,
    pair_batch_size: int = 8192,
) -> ViewTopologyContext:
    """Build a deterministic distance/depth-induced inter-anchor graph.

    ``spatial`` mode ranks deterministic Euclidean candidates. ``hybrid`` mode
    unions those candidates with global sparse-signature Jaccard candidates,
    then applies the same distance/depth score. The latter lets a strong
    camera-induced relation enter the graph even when it is not among the
    nearest Euclidean anchors, without constructing an anchor-square matrix.

    The output reads no anchor attribute: its fixed-width feature consists only
    of edge score statistics, common-camera support, and weighted relative xyz.
    Final quantization is part of the codec contract and suppresses
    insignificant cross-device floating-point differences.
    """
    if codec_xyz.ndim != 2 or codec_xyz.shape[1] != 3:
        raise ValueError("codec_xyz must have shape [N, 3]")
    if not 0 < topk <= candidate_k < codec_xyz.shape[0]:
        raise ValueError("require 0 < topk <= candidate_k < num_anchors")
    if candidate_mode not in VIEW_TOPOLOGY_CANDIDATE_MODES:
        raise ValueError(
            f"candidate_mode must be one of {VIEW_TOPOLOGY_CANDIDATE_MODES}, "
            f"got {candidate_mode!r}"
        )
    if candidate_mode == "hybrid" and not 0 < view_candidate_k < codec_xyz.shape[0]:
        raise ValueError("hybrid mode requires 0 < view_candidate_k < num_anchors")
    if feature_quantization <= 0:
        raise ValueError("feature_quantization must be positive")

    camera_geometry = extract_camera_geometry(cameras)
    xyz = codec_xyz.detach().cpu().numpy().astype(np.float64, copy=False)
    relation = build_geometry_observations(codec_xyz, camera_geometry)
    descriptors = build_geometric_observation_descriptors(codec_xyz, camera_geometry, relation)
    spatial_neighbors, spatial_distances = spatial_topk(xyz, k=candidate_k)
    view_diagnostics = None

    if candidate_mode == "spatial":
        candidate_rows = tuple(spatial_neighbors[row] for row in range(xyz.shape[0]))
        candidate_distance_rows = tuple(
            spatial_distances[row] for row in range(xyz.shape[0])
        )
    else:
        view_neighbors, _, view_diagnostics = coview_lsh_candidates(
            relation,
            k=view_candidate_k,
        )
        candidate_rows = []
        candidate_distance_rows = []
        for anchor_idx in range(xyz.shape[0]):
            # Sorted canonical indices make the union independent of the order
            # returned by either candidate generator. Scoring below determines
            # the final order.
            candidates = np.unique(np.concatenate((
                spatial_neighbors[anchor_idx],
                view_neighbors[anchor_idx],
            )))
            candidates = candidates[candidates != anchor_idx]
            candidate_rows.append(candidates)
            candidate_distance_rows.append(
                np.linalg.norm(xyz[candidates] - xyz[anchor_idx], axis=1)
            )
        candidate_rows = tuple(candidate_rows)
        candidate_distance_rows = tuple(candidate_distance_rows)

    candidate_counts = np.asarray(
        [row.size for row in candidate_rows],
        dtype=np.int64,
    )
    candidate_indptr = np.empty(xyz.shape[0] + 1, dtype=np.int64)
    candidate_indptr[0] = 0
    np.cumsum(candidate_counts, out=candidate_indptr[1:])
    source = np.repeat(
        np.arange(xyz.shape[0], dtype=np.int64),
        candidate_counts,
    )
    target = np.concatenate(candidate_rows)
    candidate_distances = np.concatenate(candidate_distance_rows)

    pair_scores = pair_distance_depth_scores(
        descriptors,
        source,
        target,
        batch_size=pair_batch_size,
    )
    distance_score = pair_scores["geometric_distance"]
    depth_score = pair_scores["geometric_depth"]
    common_count = pair_scores["common_camera_count"]
    joint_score = 0.5 * (distance_score + depth_score)
    valid = common_count > 0

    neighbors = np.empty((xyz.shape[0], topk), dtype=np.int64)
    selected_spatial_distance = np.empty((xyz.shape[0], topk), dtype=np.float64)
    selected_distance = np.empty((xyz.shape[0], topk), dtype=np.float32)
    selected_depth = np.empty((xyz.shape[0], topk), dtype=np.float32)
    selected_common = np.empty((xyz.shape[0], topk), dtype=np.float32)
    for anchor_idx in range(xyz.shape[0]):
        begin = candidate_indptr[anchor_idx]
        end = candidate_indptr[anchor_idx + 1]
        candidates = target[begin:end]
        # Common-camera pairs rank before unsupported pairs; ties use the
        # canonical anchor index, independent of scipy query order.
        ranking_score = np.where(valid[begin:end], joint_score[begin:end], -1.0)
        selected = np.lexsort((
            candidates,
            -ranking_score,
        ))[:topk]
        neighbors[anchor_idx] = candidates[selected]
        selected_spatial_distance[anchor_idx] = candidate_distances[begin:end][selected]
        selected_distance[anchor_idx] = distance_score[begin:end][selected]
        selected_depth[anchor_idx] = depth_score[begin:end][selected]
        selected_common[anchor_idx] = common_count[begin:end][selected]
    selected_valid = selected_common > 0
    selected_joint = 0.5 * (selected_distance + selected_depth)

    valid_float = selected_valid.astype(np.float64)
    valid_count = np.maximum(valid_float.sum(axis=1, keepdims=True), 1.0)

    def supported_stats(values):
        values = values.astype(np.float64, copy=False)
        mean = (values * valid_float).sum(axis=1, keepdims=True) / valid_count
        variance = (((values - mean) ** 2) * valid_float).sum(axis=1, keepdims=True) / valid_count
        maximum = np.where(selected_valid, values, -np.inf).max(axis=1, keepdims=True)
        maximum[~np.isfinite(maximum)] = 0.0
        return mean, np.sqrt(np.maximum(variance, 0.0)), maximum

    distance_mean, distance_std, distance_max = supported_stats(selected_distance)
    depth_mean, depth_std, depth_max = supported_stats(selected_depth)
    joint_mean, joint_std, joint_max = supported_stats(selected_joint)
    valid_fraction = valid_float.mean(axis=1, keepdims=True)
    camera_denominator = max(len(camera_geometry), 1)
    common_fraction = (
        selected_common.astype(np.float64) * valid_float
    ).sum(axis=1, keepdims=True) / valid_count / camera_denominator

    delta = xyz[neighbors] - xyz[:, None, :]
    local_scale = np.maximum(
        (selected_spatial_distance * valid_float).sum(axis=1, keepdims=True) / valid_count,
        1e-12,
    )
    weights = selected_joint.astype(np.float64) * valid_float
    weight_sum = np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    normalized_delta = delta / local_scale[:, :, None]
    weighted_delta = (normalized_delta * weights[:, :, None]).sum(axis=1) / weight_sum
    weighted_distance = (
        selected_spatial_distance * weights
    ).sum(axis=1, keepdims=True) / weight_sum / local_scale

    features = np.concatenate((
        distance_mean, distance_std, distance_max,
        depth_mean, depth_std, depth_max,
        joint_mean, joint_std, joint_max,
        valid_fraction, common_fraction,
        weighted_delta, weighted_distance,
    ), axis=1)
    if features.shape[1] != VIEW_TOPOLOGY_FEATURE_DIM:
        raise RuntimeError("unexpected topology feature dimension")
    features = (
        np.round(features / feature_quantization) * feature_quantization
    ).astype(np.float32)

    selected_outside_spatial = np.empty_like(neighbors, dtype=np.bool_)
    for anchor_idx in range(xyz.shape[0]):
        selected_outside_spatial[anchor_idx] = ~np.isin(
            neighbors[anchor_idx],
            spatial_neighbors[anchor_idx],
        )

    diagnostics = {
        "num_anchors": int(xyz.shape[0]),
        "num_cameras": len(camera_geometry),
        "candidate_mode": candidate_mode,
        "candidate_k": int(candidate_k),
        "view_candidate_k": int(view_candidate_k) if candidate_mode == "hybrid" else 0,
        "topk": int(topk),
        "num_candidate_pairs": int(candidate_counts.sum()),
        "mean_candidate_count": float(candidate_counts.mean()),
        "max_candidate_count": int(candidate_counts.max()),
        "selected_outside_spatial_fraction": float(selected_outside_spatial.mean()),
        "num_observation_edges": int(relation.anchor_camera_ids.size),
        "mean_valid_neighbor_fraction": float(valid_fraction.mean()),
        "feature_quantization": float(feature_quantization),
        "feature_dim": VIEW_TOPOLOGY_FEATURE_DIM,
        "dense_anchor_pair_matrix_created": False,
    }
    if view_diagnostics is not None:
        diagnostics["view_candidate_diagnostics"] = view_diagnostics
    return ViewTopologyContext(
        features=features,
        neighbors=neighbors,
        distance_scores=selected_distance.astype(np.float32),
        depth_scores=selected_depth.astype(np.float32),
        diagnostics=diagnostics,
    )
