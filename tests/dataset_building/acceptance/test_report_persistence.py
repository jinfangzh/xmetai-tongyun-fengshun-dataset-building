"""Regression checks for explicit report persistence through the public API."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import yaml
import zarr

from datakit.schema.specifications import (
    QualityRule,
    build_standard_channel_info,
)
from pipelines.dataset import validate_zarr
from pipelines.dataset.configuration import load_validation_run, load_validation_schema

from ._helpers import SCHEMA_DIRECTORY, _receipt


def test_public_api_validates_independent_published_land_fixture(published, tmp_path):
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

    document = yaml.safe_load(
        (SCHEMA_DIRECTORY / "era5_land_c8_1p5_h24.yaml").read_text(encoding="utf-8")
    )
    meta = document["meta_info"]
    meta.update(
        dataset_id="fixture.land",
        expected_root_dataset_id=spec.dataset_id,
        dataset_path=str(destination),
        auxiliary_directory=str(destination),
        data_revision="r1",
        validation_scope="strict-standard",
        identity_mapping_evidence="Independently declared fixture directory mapping",
        quality=[asdict(q) for q in spec.quality],
        mask_provenance=spec.mask_provenance,
    )
    meta["time_coverage"].update(first=spec.first_time, last=spec.last_time, time_count=2)
    schema = tmp_path / "schema.yaml"
    schema.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {
                "schema": schema.name,
                "execution": {
                    "snapshot_id": spec.snapshot_id,
                    "production_evidence": {
                        "path": str(spec.production_evidence.path),
                        "sha256": spec.production_evidence.sha256,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.yaml"
    runtime_spec, locations = load_validation_run(run)
    report = validate_zarr(locations["dataset_path"], runtime_spec, output=output)
    assert report.release_passed
    saved = yaml.safe_load(output.read_text())
    assert saved["release_passed"]
    identity = saved["identity"]
    assert identity["config_id"] == "fixture.land"
    assert identity["template_version"] == "v1"
    assert identity["schema_sha256"] == hashlib.sha256(schema.read_bytes()).hexdigest()
    assert saved["coverage"]["value_checked"] == 2
    frozen = output.read_bytes()
    with pytest.raises(FileExistsError):
        validate_zarr(destination, runtime_spec, output=output)
    assert output.read_bytes() == frozen
    assert not list(tmp_path.glob("*.jsonl"))
    assert not (tmp_path / "summary.csv").exists()
    forbidden = destination / "reports.yaml"
    with pytest.raises(ValueError):
        validate_zarr(destination, runtime_spec, output=forbidden)
    assert not forbidden.exists()
