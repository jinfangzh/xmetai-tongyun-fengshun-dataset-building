"""Regression checks for standard dataset cli."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import zarr

from pipelines.dataset_building.definitions.dataset_products import get_dataset_product
from pipelines.dataset_building.definitions.standard_dataset import (
    DatasetValidationSpec,
    QualityRule,
    build_standard_channel_info,
)

from ._helpers import _receipt


def test_cli_runs_independent_published_land_fixture(published, tmp_path):
    path, original = published
    product = get_dataset_product("era5-land.c8")
    spec = DatasetValidationSpec(
        "land-fixture",
        product,
        "1p5.h24",
        "2024-02-28T00",
        "2024-02-29T00",
        "r1",
        quality=tuple(
            QualityRule(n, "all", 0, True, "approved fixture quality", (-1, 1), "stored")
            for n in product.channel_names
        ),
        mask_provenance="approved mask",
        auxiliary_provenance={"weight.npy": "approved latitude weight"},
    )
    destination = tmp_path / "land.zarr"
    group = zarr.open_group(str(destination), mode="w", zarr_format=3)
    lat, lon = spec.variant.grid.coordinates()
    group.attrs.update(
        dataset_id=spec.dataset_id,
        schema_version="2.0",
        content_version="v3",
        data_revision="r1",
        geospatial_lat_range=[-90, 90],
        geospatial_lon_range=[0, 358.5],
    )
    old = zarr.open_group(str(path), mode="r")
    mean, std = np.zeros(8, dtype="f4"), np.ones(8, dtype="f4")
    land = np.zeros((121, 240), dtype="u1")
    land[:60] = 1
    arrays = {
        "data": (
            np.ones((2, 8, 121, 240), dtype="f2") / 2,
            ("time", "channel", "lat", "lon"),
            dict(old["data"].attrs),
            (1, 8, 121, 240),
        ),
        "time": (spec.expected_times(), ("time",), {"timezone": "UTC"}, (2,)),
        "channel": (
            np.array(product.channel_names),
            ("channel",),
            {
                "channel_info_schema_version": "1.0",
                "channel_info": build_standard_channel_info(product, mean=mean, std=std),
            },
            (8,),
        ),
        "lat": (lat, ("lat",), {}, (121,)),
        "lon": (lon, ("lon",), {}, (240,)),
        "mean": (mean, ("channel",), {}, (8,)),
        "std": (std, ("channel",), {}, (8,)),
        "mask": (np.stack([land, 1 - land]), ("mask_channel", "lat", "lon"), {}, (2, 121, 240)),
        "mask_channel": (np.array(["land_mask", "sea_mask"]), ("mask_channel",), {}, (2,)),
    }
    for name, (values, dims, attrs, chunks) in arrays.items():
        group.create_array(
            name,
            data=values,
            chunks=chunks,
            attributes=attrs,
            dimension_names=dims,
            config={"write_empty_chunks": True},
        )
    zarr.consolidate_metadata(str(destination))
    np.save(destination / "weight.npy", np.ones((1, 121, 1), dtype="f4"))
    spec = _receipt(destination, spec)
    from dataclasses import asdict

    document = {
        "dataset_id": spec.dataset_id,
        "product_id": product.product_id,
        "product_version": "1.0",
        "variant_id": "1p5.h24",
        "data_revision": "r1",
        "dataset_path": str(destination),
        "expected_time": {"first": spec.first_time, "last": spec.last_time, "count": 2},
        "quality": [asdict(q) for q in spec.quality],
        "mask_provenance": spec.mask_provenance,
        "auxiliary_provenance": dict(spec.auxiliary_provenance),
        "snapshot_id": spec.snapshot_id,
        "production_evidence": {
            "path": str(spec.production_evidence.path),
            "sha256": spec.production_evidence.sha256,
        },
    }
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(document))
    output = tmp_path / "reports"
    command = [
        sys.executable,
        "-m",
        "examples.dataset_building.validate_standard_dataset",
        "--profile",
        str(profile),
        "--mode",
        "full",
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reports = [p for p in output.glob("*.json") if p.name != "batch.json"]
    assert json.loads(reports[0].read_text())["release_passed"]
    ledger = Path(json.loads(reports[0].read_text())["identity"]["block_ledger"])
    assert len(ledger.read_text().splitlines()) == 2
    frozen_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
    repeated = subprocess.run(
        command, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2]
    )
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == frozen_hash
    assert len(list(output.glob("*-blocks.jsonl"))) == 2
    assert (output / "summary.csv").exists() and (output / "summary.md").exists()
    forbidden = destination / "reports"
    command[-1] = str(forbidden)
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2]
    )
    assert result.returncode == 3 and not forbidden.exists()
