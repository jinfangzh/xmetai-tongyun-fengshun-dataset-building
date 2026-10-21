"""Alignment and value validation for refreshed Zarr stores."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from .source import LegacyZarrSource
from .store import CONTENT_VERSION, MASK_CHANNELS, SCHEMA_VERSION
EXPECTED_CHILDREN = {
    "data",
    "channel",
    "lat",
    "lon",
    "time",
    "mean",
    "std",
    "mask",
    "mask_channel",
}


def default_sample_indices(size: int) -> list[int]:
    if size <= 3:
        return list(range(size))
    return sorted({0, size // 2, size - 1})


def _strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values.tolist()]


def validate_refreshed_store(
    path: Path,
    source: LegacyZarrSource,
    source_indices: list[int],
    *,
    full_data_check: bool = False,
) -> dict[str, Any]:
    group = zarr.open_group(str(path), mode="r", use_consolidated=False)
    if set(group.keys()) != EXPECTED_CHILDREN:
        raise ValueError(f"unexpected root nodes: {sorted(group.keys())}")
    attrs = dict(group.attrs)
    if attrs.get("schema_version") != SCHEMA_VERSION or attrs.get("content_version") != CONTENT_VERSION:
        raise ValueError("root schema/content version mismatch")
    expected_shape = (
        len(source_indices),
        len(source.channels),
        len(source.lat),
        len(source.lon),
    )
    if tuple(group["data"].shape) != expected_shape:
        raise ValueError(f"target data shape {group['data'].shape} != {expected_shape}")
    if group["data"].dtype != np.dtype("f2"):
        raise ValueError("target data dtype is not float16")
    if tuple(group["data"].chunks) != (1, *expected_shape[1:]):
        raise ValueError("target data chunks do not follow the v2 contract")
    if _strings(group["channel"][:]) != list(source.channels):
        raise ValueError("channel values/order changed")
    np.testing.assert_array_equal(group["lat"][:], source.lat.astype("f4"))
    np.testing.assert_array_equal(group["lon"][:], source.lon.astype("f4"))
    np.testing.assert_array_equal(
        np.asarray(group["time"][:]).astype("datetime64[ns]"),
        source.times[source_indices].astype("datetime64[ns]"),
    )
    np.testing.assert_array_equal(group["mean"][:], source.mean.astype("f4"))
    np.testing.assert_array_equal(group["std"][:], source.std.astype("f4"))
    if _strings(group["mask_channel"][:]) != list(MASK_CHANNELS):
        raise ValueError("mask channel registry mismatch")
    masks = np.asarray(group["mask"][:], dtype="u1")
    if masks.shape != (2, len(source.lat), len(source.lon)):
        raise ValueError("mask shape mismatch")
    if not np.isin(masks, (0, 1)).all() or not np.array_equal(
        masks[0] + masks[1], np.ones_like(masks[0])
    ):
        raise ValueError("masks are not binary complements")
    np.testing.assert_array_equal(masks[0], source.mask)

    output_indices: Iterable[int]
    output_indices = range(len(source_indices)) if full_data_check else default_sample_indices(len(source_indices))
    checked = []
    radiation_report: dict[str, Any] = {}
    for output_index in output_indices:
        source_index = source_indices[output_index]
        expected = source.read_data(source_index)
        actual = np.asarray(group["data"][output_index], dtype="f2")
        if not np.array_equal(actual, expected, equal_nan=True):
            difference = np.nanmax(np.abs(actual.astype("f4") - expected.astype("f4")))
            raise ValueError(
                f"normalized data mismatch output={output_index} source={source_index}; max_abs={difference}"
            )
        checked.append({"output_index": output_index, "source_index": source_index})

    for name in ("ssr", "ssrd", "fdir", "ttr"):
        if name not in source.channels:
            continue
        ci = source.channels.index(name)
        samples = []
        for output_index in default_sample_indices(len(source_indices)):
            normalized = np.asarray(group["data"][output_index, ci], dtype="f4")
            physical = normalized * source.std[ci] + source.mean[ci]
            finite = physical[np.isfinite(physical)]
            if finite.size:
                samples.append(
                    {
                        "output_index": output_index,
                        "min": float(np.min(finite)),
                        "mean": float(np.mean(finite)),
                        "max": float(np.max(finite)),
                    }
                )
        radiation_report[name] = samples

    return {
        "path": str(path),
        "shape": list(expected_shape),
        "checked_data_chunks": checked,
        "full_data_check": full_data_check,
        "radiation_physical_samples": radiation_report,
        "status": "ok",
    }


def read_manifest(path: Path) -> dict[str, Any]:
    manifest = path / "refresh_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return json.loads(manifest.read_text(encoding="utf-8"))
