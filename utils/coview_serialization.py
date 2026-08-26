"""Deterministic binary serialization for active CoView entropy parameters."""

from collections import OrderedDict
from pathlib import Path
import hashlib
import struct

import numpy as np
import torch


MAGIC = b"HACCVW1\0"
FORMAT_CODES = {"fp32": 1, "fp16": 2, "int8": 3}
CODE_FORMATS = {value: key for key, value in FORMAT_CODES.items()}


def _tensor_payload(tensor, storage_format):
    value = tensor.detach().cpu().contiguous().to(torch.float32)
    scale = None
    if storage_format == "fp32":
        payload = value.numpy().astype("<f4", copy=False).tobytes(order="C")
    elif storage_format == "fp16":
        payload = value.numpy().astype("<f2").tobytes(order="C")
    elif storage_format == "int8":
        max_abs = float(value.abs().max()) if value.numel() else 0.0
        scale = max_abs / 127.0 if max_abs else 1.0
        quantized = torch.clamp(torch.round(value / scale), -127, 127).to(torch.int8)
        payload = quantized.numpy().tobytes(order="C")
    else:
        raise ValueError(f"unsupported CoView serialization format {storage_format!r}")
    return payload, scale


def serialize_named_tensors(named_tensors, path, storage_format="fp32"):
    """Serialize a mapping of floating tensors and return real blob metadata."""
    if storage_format not in FORMAT_CODES:
        raise ValueError(f"unsupported CoView serialization format {storage_format!r}")
    items = named_tensors.items() if hasattr(named_tensors, "items") else named_tensors
    ordered = OrderedDict(sorted(items))
    blob = bytearray(MAGIC)
    blob.extend(struct.pack("<BI", FORMAT_CODES[storage_format], len(ordered)))
    for name, tensor in ordered.items():
        if not torch.is_floating_point(tensor):
            raise TypeError(f"CoView tensor {name!r} must be floating point")
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 65535 or tensor.ndim > 255:
            raise ValueError(f"CoView tensor metadata is too large for {name!r}")
        payload, scale = _tensor_payload(tensor, storage_format)
        blob.extend(struct.pack("<HB", len(name_bytes), tensor.ndim))
        blob.extend(name_bytes)
        for dimension in tensor.shape:
            blob.extend(struct.pack("<I", int(dimension)))
        if storage_format == "int8":
            blob.extend(struct.pack("<f", scale))
        blob.extend(struct.pack("<Q", len(payload)))
        blob.extend(payload)
    blob = bytes(blob)
    destination = Path(path)
    destination.write_bytes(blob)
    return {
        "format": storage_format,
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "tensor_count": len(ordered),
    }


def deserialize_named_tensors(path):
    """Read a CoView blob and dequantize values to CPU float32 tensors."""
    blob = Path(path).read_bytes()
    view = memoryview(blob)
    cursor = 0

    def take(size):
        nonlocal cursor
        if cursor + size > len(view):
            raise RuntimeError("truncated CoView model blob")
        result = view[cursor:cursor + size]
        cursor += size
        return result

    if bytes(take(len(MAGIC))) != MAGIC:
        raise RuntimeError("invalid CoView model magic")
    format_code, tensor_count = struct.unpack("<BI", take(5))
    if format_code not in CODE_FORMATS:
        raise RuntimeError(f"unsupported CoView model format code {format_code}")
    storage_format = CODE_FORMATS[format_code]
    tensors = OrderedDict()
    for _ in range(tensor_count):
        name_length, ndim = struct.unpack("<HB", take(3))
        name = bytes(take(name_length)).decode("utf-8")
        if name in tensors:
            raise RuntimeError(f"duplicate CoView tensor {name!r}")
        shape = tuple(struct.unpack("<I", take(4))[0] for _ in range(ndim))
        scale = struct.unpack("<f", take(4))[0] if storage_format == "int8" else None
        payload_length = struct.unpack("<Q", take(8))[0]
        payload = take(payload_length)
        element_count = int(np.prod(shape, dtype=np.int64))
        if storage_format == "fp32":
            array = np.frombuffer(payload, dtype="<f4", count=element_count)
        elif storage_format == "fp16":
            array = np.frombuffer(payload, dtype="<f2", count=element_count).astype(np.float32)
        else:
            array = np.frombuffer(payload, dtype=np.int8, count=element_count).astype(np.float32)
            array *= scale
        expected_bytes = element_count * {"fp32": 4, "fp16": 2, "int8": 1}[storage_format]
        if payload_length != expected_bytes:
            raise RuntimeError(f"invalid payload length for CoView tensor {name!r}")
        tensors[name] = torch.from_numpy(array.copy()).reshape(shape)
    if cursor != len(view):
        raise RuntimeError("trailing bytes in CoView model blob")
    return tensors, {
        "format": storage_format,
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "tensor_count": tensor_count,
    }
