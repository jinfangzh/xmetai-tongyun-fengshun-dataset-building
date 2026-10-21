"""Strict local Zarr inspection without legacy fallback or data modification.

Physical coverage is supported for nonsharded v2/v3 directory stores. Sharded
or unknown layouts retain raw metadata but are explicitly unsupported for this
audit, even if the installed Zarr library could decode them.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zarr

from ._common import local_path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected metadata object: {path}")
    return value


def read_zarr_metadata(path: str | Path) -> dict[str, Any]:
    root = local_path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    version = 3 if (root / "zarr.json").is_file() else 2 if (root / ".zgroup").is_file() else None
    if version is None:
        raise ValueError(f"Not a Zarr directory: {root}")
    root_meta = _json(root / ("zarr.json" if version == 3 else ".zgroup"))
    if root_meta.get("zarr_format") != version:
        raise ValueError("Root metadata format contradicts its canonical metadata file")
    if version == 3 and root_meta.get("node_type") != "group":
        raise ValueError("The dataset root must be a Zarr group")
    arrays, nested = {}, []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not child.resolve().is_relative_to(root):
            raise ValueError(f"Array path escapes the store: {child}")
        metadata_path = child / ("zarr.json" if version == 3 else ".zarray")
        if version == 2 and (child / ".zgroup").is_file():
            nested.append(child.name)
            continue
        if metadata_path.is_file():
            raw = _json(metadata_path)
            if raw.get("zarr_format") != version:
                raise ValueError(f"Mixed array metadata format: {child.name}")
            if version == 3 and raw.get("node_type") != "array":
                nested.append(child.name)
                continue
            attrs = (
                raw.get("attributes", {})
                if version == 3
                else _json(child / ".zattrs") if (child / ".zattrs").exists() else {}
            )
            arrays[child.name] = {
                "raw": raw,
                "attributes": attrs,
                "shape": tuple(raw["shape"]),
                "dimensions": (
                    tuple(raw.get("dimension_names") or ())
                    if version == 3
                    else tuple(attrs.get("_ARRAY_DIMENSIONS", ()))
                ),
                "chunks": (
                    tuple(raw.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape", ()))
                    if version == 3
                    else tuple(raw["chunks"])
                ),
            }
    attrs = (
        root_meta.get("attributes", {})
        if version == 3
        else _json(root / ".zattrs") if (root / ".zattrs").exists() else {}
    )
    return {
        "path": str(root),
        "zarr_format": version,
        "root": root_meta,
        "attributes": attrs,
        "arrays": arrays,
        "nested_groups": nested,
    }


def metadata_digest(metadata: dict[str, Any]) -> str:
    root = Path(metadata["path"])
    files = [
        root / name
        for name in ("zarr.json", ".zgroup", ".zattrs", ".zmetadata")
        if (root / name).is_file()
    ]
    for name in metadata["arrays"]:
        files.extend(
            root / name / file
            for file in ("zarr.json", ".zarray", ".zattrs")
            if (root / name / file).is_file()
        )
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def consolidated_mismatches(metadata: dict[str, Any]) -> list[str] | None:
    root = Path(metadata["path"])
    if metadata["zarr_format"] == 3:
        consolidated = metadata["root"].get("consolidated_metadata")
        if consolidated is None:
            return None
        if consolidated.get("kind") != "inline" or not isinstance(
            consolidated.get("metadata"), dict
        ):
            return ["unsupported consolidated metadata representation"]
        cached = consolidated["metadata"]
        mismatches = [
            name for name, item in metadata["arrays"].items() if cached.get(name) != item["raw"]
        ]
        mismatches.extend(
            name
            for name, item in cached.items()
            if item.get("node_type") == "array" and name not in metadata["arrays"]
        )
        return sorted(set(mismatches))
    file = root / ".zmetadata"
    if not file.exists():
        return None
    cached = _json(file).get("metadata", {})
    mismatches = []
    for name, item in metadata["arrays"].items():
        if (
            cached.get(f"{name}/.zarray") != item["raw"]
            or cached.get(f"{name}/.zattrs", {}) != item["attributes"]
        ):
            mismatches.append(name)
    if cached.get(".zattrs", {}) != metadata["attributes"]:
        mismatches.append("root attributes")
    actual = {f"{name}/.zarray" for name in metadata["arrays"]}
    mismatches.extend(key for key in cached if key.endswith("/.zarray") and key not in actual)
    return sorted(set(mismatches))


def physical_layout_supported(metadata: dict[str, Any], name: str = "data") -> tuple[bool, str]:
    item = metadata["arrays"][name]
    raw = item["raw"]
    if metadata["zarr_format"] == 3:
        if raw.get("chunk_grid", {}).get("name") != "regular":
            return False, "Only regular nonsharded chunk grids have physical coverage support"
        if raw.get("storage_transformers"):
            return False, "Storage transformers have no physical coverage audit"
        codecs = raw.get("codecs", [])
        allowed = {"bytes", "blosc", "gzip", "zstd", "crc32c", "transpose"}
        if name != "data":
            allowed.update({"vlen-utf8", "vlen-bytes", "fixed_length_utf32"})
        if any(codec.get("name") not in allowed for codec in codecs):
            return False, "Sharding or unverified data codecs have no physical coverage audit"
        key = raw.get("chunk_key_encoding", {})
        if key.get("name") not in ("default", "v2") or key.get("configuration", {}).get(
            "separator", "/"
        ) not in ("/", "."):
            return False, "Unknown chunk key encoding"
    else:
        if raw.get("dimension_separator", ".") not in ("/", "."):
            return False, "Unknown v2 chunk key separator"
        compressor = raw.get("compressor")
        if compressor and compressor.get("id") not in ("blosc", "gzip", "zstd"):
            return False, "Unverified v2 compressor"
        if raw.get("filters"):
            return False, "Filtered v2 data has no audited support"
    return True, "local-directory/nonsharded/regular"


def zarr_chunk_path(metadata: dict[str, Any], name: str, index: tuple[int, ...]) -> Path:
    raw = metadata["arrays"][name]["raw"]
    if metadata["zarr_format"] == 2:
        key = raw.get("dimension_separator", ".").join(map(str, index))
    else:
        encoding = raw["chunk_key_encoding"]
        separator = encoding.get("configuration", {}).get(
            "separator", "/" if encoding["name"] == "default" else "."
        )
        key = (
            separator.join(("c", *map(str, index)))
            if encoding["name"] == "default"
            else separator.join(map(str, index))
        )
    target = Path(metadata["path"]) / name / key
    if not target.resolve().is_relative_to(Path(metadata["path"]).resolve()):
        raise ValueError("Chunk path escapes the local store")
    return target


def expected_zarr_blocks(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> Iterator[tuple[tuple[int, ...], tuple[slice, ...]]]:
    if len(shape) != len(chunks) or any(s <= 0 for s in shape) or any(c <= 0 for c in chunks):
        raise ValueError("Expected block grid must have positive dimensions")
    ranges = [range((s + c - 1) // c) for s, c in zip(shape, chunks)]
    for index in itertools.product(*ranges):
        yield index, tuple(
            slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(index, chunks, shape)
        )


@contextmanager
def open_zarr_array_readonly(path: str | Path, name: str) -> Iterator[Any]:
    root = local_path(path).resolve()
    target = root / name
    if (
        name in (".", "..")
        or "/" in name
        or "\\" in name
        or not target.resolve().is_relative_to(root)
    ):
        raise ValueError("Array name must be a direct child")
    version = 3 if (root / "zarr.json").is_file() else 2 if (root / ".zgroup").is_file() else None
    if version is None:
        raise ValueError("Expected a Zarr group parent with an explicit format")
    array = zarr.open_array(str(target), mode="r", zarr_format=version)
    try:
        yield array
    finally:
        store = getattr(array, "store", None)
        close = getattr(store, "close", None)
        if close is not None:
            close()


def read_zarr_block(array: Any, selection: tuple[slice, ...]) -> np.ndarray:
    """Decode exactly the requested logical slice; errors propagate unchanged."""
    return np.asarray(array[selection])
