"""Regression checks for standard dataset cli."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
import zarr

from datakit.schema.datasets.specifications import (
    QualityRule,
    build_standard_channel_info,
)
from pipelines.dataset_building.support.inputs import load_validation_schema

from ._helpers import SCHEMA_DIRECTORY, _receipt


def test_cli_runs_independent_published_land_fixture(published, tmp_path):
    path, original = published
    product = load_validation_schema(SCHEMA_DIRECTORY / "era5_land_c8_1p5_h24.yaml")[0].product
    spec = replace(
        original,
        dataset_id="land-fixture",
        product=product,
        variant_name="1p5.h24",
        first_time="2024-02-28T00",
        last_time="2024-02-29T00",
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
                "channel_info": build_standard_channel_info(
                    product, mean=mean, std=std, channel_metadata=spec.channel_metadata
                ),
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

    document = yaml.safe_load((SCHEMA_DIRECTORY / "era5_land_c8_1p5_h24.yaml").read_text(encoding="utf-8"))
    meta = document["meta_info"]
    meta.update(
        dataset_id="fixture.land", expected_root_dataset_id=spec.dataset_id,
        dataset_path=str(destination), auxiliary_directory=str(destination), data_revision="r1",
        validation_scope="strict-standard",
        identity_mapping_evidence="Independently declared fixture directory mapping",
        quality=[asdict(q) for q in spec.quality], mask_provenance=spec.mask_provenance,
    )
    meta["time_coverage"].update(first=spec.first_time, last=spec.last_time, time_count=2)
    schema = tmp_path / "schema.yaml"
    schema.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    run = tmp_path / "run.json"
    run.write_text(json.dumps({
        "schema": schema.name,
        "execution": {
            "snapshot_id": spec.snapshot_id,
            "production_evidence": {
                "path": str(spec.production_evidence.path),
                "sha256": spec.production_evidence.sha256,
            },
        },
    }), encoding="utf-8")
    output = tmp_path / "reports"
    command = [
        sys.executable,
        "-m",
        "examples.dataset_building.validate_standard_dataset",
        "--run-config",
        str(run),
        "--mode",
        "full",
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[3]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reports = [p for p in output.glob("*.json") if p.name != "batch.json"]
    assert json.loads(reports[0].read_text())["release_passed"]
    identity = json.loads(reports[0].read_text())["identity"]
    assert identity["config_id"] == "fixture.land"
    assert identity["template_version"] == "v1"
    assert identity["schema_sha256"] == hashlib.sha256(schema.read_bytes()).hexdigest()
    ledger = Path(json.loads(reports[0].read_text())["identity"]["block_ledger"])
    assert len(ledger.read_text().splitlines()) == 2
    frozen_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
    repeated = subprocess.run(
        command, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[3]
    )
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == frozen_hash
    assert len(list(output.glob("*-blocks.jsonl"))) == 2
    assert (output / "summary.csv").exists() and (output / "summary.md").exists()
    forbidden = destination / "reports"
    command[-1] = str(forbidden)
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[3]
    )
    assert result.returncode == 3 and not forbidden.exists()
