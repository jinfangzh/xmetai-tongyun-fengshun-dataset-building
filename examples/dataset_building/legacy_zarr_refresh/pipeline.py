"""Orchestration for refreshing legacy Zarr stores."""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import zarr

from .config import RefreshProfile
from .source import LegacyZarrSource
from .store import create_refreshed_store
from .validation import validate_refreshed_store
def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def selection_indices(size: int, start: int, count: int | None) -> list[int]:
    if start < 0 or start >= size:
        raise ValueError(f"start index {start} outside [0, {size})")
    stop = size if count is None else min(size, start + count)
    if count is not None and count <= 0:
        raise ValueError("count must be positive")
    return list(range(start, stop))


def output_path(
    profile: RefreshProfile,
    output_root: Path,
    *,
    pilot: bool,
    start: int,
    count: int | None,
) -> Path:
    name = profile.output_name
    if pilot:
        assert count is not None
        name += f".pilot.{start}_{start + count - 1}"
    return output_root / name


def build_refresh_plan(
    source: LegacyZarrSource,
    destination: Path,
    indices: list[int],
) -> dict[str, Any]:
    """Return a read-only summary of transformations that would be applied."""
    identity = tuple(range(len(source.channel_indices)))
    return {
        "input": str(source.path),
        "output": str(destination),
        "source_indices": {
            "start": indices[0],
            "end": indices[-1],
            "count": len(indices),
        },
        "actions": {
            "channel_rename": bool(source.profile.channel_rename),
            "channel_reorder": source.channel_indices != identity,
            "latitude_flip": source.flip_latitude,
            "longitude_reorder": False,
            "statistics_divide": source.profile.statistics_divide,
            "statistics_multiply": source.profile.statistics_multiply,
            "data_dtype_conversion": str(source.data.dtype) != "float16",
            "mask_binarization": source.mask_binarized,
        },
    }


def refresh_legacy_zarr(
    profile: RefreshProfile,
    output_root: Path,
    *,
    start: int = 0,
    count: int | None = None,
    pilot: bool = False,
    workers: int = 1,
    resume: bool = True,
    overwrite: bool = False,
    full_data_check: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    source = LegacyZarrSource(profile)
    indices = selection_indices(len(source.times), start, count)
    final = output_path(profile, output_root, pilot=pilot, start=start, count=len(indices))
    if final.resolve() == source.path.resolve():
        raise ValueError("output path must not equal the read-only source path")
    staging = final.with_name("." + final.name + ".staging")
    state_path = final.with_name("." + final.name + ".refresh_state.json")
    fingerprint_payload = {
        "source_fingerprint": source.fingerprint(),
        "source_indices": [indices[0], indices[-1]],
        "count": len(indices),
        "output_name": final.name,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    completed = 0
    if staging.exists():
        if not resume:
            raise FileExistsError(f"staging exists and resume is disabled: {staging}")
        if not state_path.is_file():
            raise RuntimeError(f"staging has no state file: {staging}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError("staging fingerprint differs from current source/config")
        completed = int(state.get("completed", 0))
        group = zarr.open_group(str(staging), mode="r+", use_consolidated=False)
    else:
        revision = datetime.now(timezone.utc).strftime("%Y%m%d")
        group = create_refreshed_store(
            staging,
            dataset_id=final.name.removesuffix(".zarr"),
            data_revision=revision,
            times=source.times[indices],
            channels=source.channels,
            lat=source.lat,
            lon=source.lon,
            mean=source.mean,
            std=source.std,
            land_mask=source.mask,
            radiation_seconds=profile.radiation_seconds,
        )
        atomic_json(
            state_path,
            {
                "fingerprint": fingerprint,
                "completed": 0,
                "total": len(indices),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    target = group["data"]
    session_start = completed
    started = time.monotonic()

    def load(item: tuple[int, int]) -> tuple[int, np.ndarray]:
        output_index, source_index = item
        return output_index, source.read_data(source_index)

    while completed < len(indices):
        block_stop = min(len(indices), completed + workers)
        work = [(i, indices[i]) for i in range(completed, block_stop)]
        if workers == 1:
            loaded = [load(work[0])]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                loaded = list(executor.map(load, work))
        for output_index, values in sorted(loaded):
            target[output_index] = values
        completed = block_stop
        atomic_json(
            state_path,
            {
                "fingerprint": fingerprint,
                "completed": completed,
                "total": len(indices),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        elapsed = time.monotonic() - started
        processed = completed - session_start
        rate = processed / elapsed if elapsed else 0.0
        eta_seconds = (len(indices) - completed) / rate if rate else float("inf")
        print(
            f"[{profile.key}] {completed}/{len(indices)} "
            f"({completed / len(indices):.2%}) rate={rate:.3f} steps/s "
            f"eta={eta_seconds / 3600:.2f}h",
            flush=True,
        )

    del target, group
    zarr.consolidate_metadata(str(staging))
    validation = validate_refreshed_store(
        staging, source, indices, full_data_check=full_data_check
    )
    manifest = {
        "tool": "data-format-refresh",
        "profile": profile.serializable(),
        "source_fingerprint": source.fingerprint(),
        "source_indices": {"start": indices[0], "end": indices[-1], "count": len(indices)},
        "source_time": {"start": str(source.times[indices[0]]), "end": str(source.times[indices[-1]])},
        "transformations": {
            "normalized_data": "unchanged",
            "latitude_reversed": source.flip_latitude,
            "statistics_divide": profile.statistics_divide,
            "statistics_multiply": profile.statistics_multiply,
            "channel_rename": profile.channel_rename,
            "channel_order": "template_intersection",
            "channel_source_indices": list(source.channel_indices),
            "codec": "blosc-zstd-clevel5-bitshuffle",
        },
        "validation": validation,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(staging / "refresh_manifest.json", manifest)
    state_path.unlink(missing_ok=True)
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"output exists: {final}")
        backup = final.with_name(
            final.name + ".backup." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        os.replace(final, backup)
    os.replace(staging, final)
    return final, manifest
