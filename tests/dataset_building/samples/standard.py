"""Physically written v3 samples, independent of production metadata builders."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml
import zarr

from datakit.schema.datasets.specifications import (
    ChannelDefinition,
    DatasetProduct,
    DatasetVariant,
    EvidenceReference,
    GridDefinition,
    QualityRule,
)
from pipelines.dataset_building.support.inputs import load_validation_schema
from pipelines.dataset_building.support.specifications import DatasetValidationSpec


def write_production_evidence(path: Path, spec: DatasetValidationSpec) -> DatasetValidationSpec:
    receipt = path.parent / "production.json"
    receipt.write_text(
        json.dumps(
            {
                "dataset_id": spec.dataset_id,
                "data_revision": spec.data_revision,
                "snapshot_id": "frozen-fixture",
                "shape": list(spec.expected_shape),
                "logical_blocks_expected": spec.expected_shape[0],
                "complete": True,
                "allowed_omitted": [list(i) for i in spec.allowed_omitted],
                "stability": "writes-stopped",
            }
        ),
        encoding="utf-8",
    )
    return replace(
        spec,
        snapshot_id="frozen-fixture",
        production_evidence=EvidenceReference(
            receipt, hashlib.sha256(receipt.read_bytes()).hexdigest()
        ),
    )


def write_published(
    root: Path,
    *,
    dataset_id: str = "fixture",
    channel_names: tuple[str, ...] = ("t2m", "q1000", "tp"),
    time_shift_days: int = 0,
) -> tuple[Path, DatasetValidationSpec]:
    # Expectations are independent of the metadata builder under test.
    declarations, _ = load_validation_schema(Path(__file__).with_name("standard.yaml"))
    definitions = (
        ChannelDefinition(
            "t2m",
            "t2m",
            "K",
            "K",
            "height_above_ground",
            2,
            "m",
            variable_type="instantaneous",
            preprocessing=({"operation": "zscore", "mean": "/mean", "std": "/std"},),
        ),
        ChannelDefinition(
            "q1000",
            "q",
            "g kg-1",
            "kg kg-1",
            "pressure",
            1000,
            "hPa",
            variable_type="instantaneous",
            preprocessing=(
                {
                    "operation": "multiply",
                    "factor": 1000.0,
                    "input_units": "kg kg-1",
                    "output_units": "g kg-1",
                },
                {"operation": "zscore", "mean": "/mean", "std": "/std"},
            ),
        ),
        ChannelDefinition(
            "tp",
            "tp",
            "1",
            "m",
            "surface",
            variable_type="accumulated",
            preprocessing=(
                {
                    "operation": "multiply",
                    "factor": 1000.0,
                    "input_units": "m",
                    "output_units": "mm",
                },
                {"operation": "clip_min", "minimum": 0.0},
                {"operation": "log1p"},
            ),
        ),
    )
    by_name = {channel.name: channel for channel in definitions}
    product = DatasetProduct(
        f"{dataset_id}.c{len(channel_names)}",
        "1.0",
        tuple(by_name[name] for name in channel_names),
        (
            DatasetVariant(
                "tiny.h6", GridDefinition("tiny", 3, 4, 1.5, 90.0, 0.0, "north-to-south"), 6
            ),
        ),
        mask=declarations.product.mask,
        effective_statistics={"tp": (0.0, 1.0)} if "tp" in channel_names else {},
    )
    times = (
        np.datetime64("2024-02-28T00", "ns")
        + np.timedelta64(time_shift_days, "D")
        + np.arange(12) * np.timedelta64(6, "h")
    )
    spec = DatasetValidationSpec(
        dataset_id,
        product,
        "tiny.h6",
        str(times[0].astype("datetime64[h]")),
        str(times[-1].astype("datetime64[h]")),
        "r1",
        template_version=declarations.template_version,
        schema_version=declarations.schema_version,
        content_version=declarations.content_version,
        zarr_format=declarations.zarr_format,
        validation_scope=declarations.validation_scope,
        schema_status=declarations.schema_status,
        arrays=declarations.arrays,
        data_attributes=declarations.data_attributes,
        channel_metadata=declarations.channel_metadata,
        unregistered_auxiliary_status=declarations.unregistered_auxiliary_status,
        mask_provenance="independent-fixture-mask",
        quality=tuple(
            QualityRule(
                name, "all", 0.0, False, "independent-fixture-limits", (-10.0, 10.0), "stored"
            )
            for name in product.channel_names
        ),
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "data.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.attrs.update(
        dataset_id=dataset_id,
        schema_version="2.0",
        content_version="v3",
        data_revision="r1",
        geospatial_lat_range=[87.0, 90.0],
        geospatial_lon_range=[0.0, 4.5],
    )
    count = len(channel_names)
    data = np.tile(np.arange(12, dtype="f2").reshape(1, 1, 3, 4) / 12, (12, count, 1, 1))
    dataattrs = {
        "data_representation": "normalized",
        "normalization_method": "channel_dependent",
        "mean_variable": "mean",
        "std_variable": "std",
        "scale_factor_location": "channel.channel_info[*].scale_factor",
        "add_offset_location": "channel.channel_info[*].add_offset",
        "inverse_transform": "preprocessed_value = data * scale_factor + add_offset",
    }
    zscore = {"operation": "zscore", "mean": "/mean", "std": "/std"}
    info = {
        "t2m": {
            "units": "K",
            "source_units": "K",
            "level_type": "height_above_ground",
            "level": 2,
            "level_units": "m",
            "variable_type": "instantaneous",
            "preprocessing": [zscore],
        },
        "q1000": {
            "units": "g kg-1",
            "source_units": "kg kg-1",
            "level_type": "pressure",
            "level": 1000,
            "level_units": "hPa",
            "variable_type": "instantaneous",
            "preprocessing": [
                {
                    "operation": "multiply",
                    "factor": 1000.0,
                    "input_units": "kg kg-1",
                    "output_units": "g kg-1",
                },
                zscore,
            ],
        },
        "tp": {
            "units": "1",
            "source_units": "m",
            "level_type": "surface",
            "variable_type": "accumulated",
            "preprocessing": [
                {
                    "operation": "multiply",
                    "factor": 1000.0,
                    "input_units": "m",
                    "output_units": "mm",
                },
                {"operation": "clip_min", "minimum": 0.0},
                {"operation": "log1p"},
            ],
        },
    }
    info = {name: info[name] for name in channel_names}
    for name, fields in info.items():
        fields.update(source_name="q" if name == "q1000" else name, long_name=name)
        fields.update(
            scale_factor=1.0,
            add_offset=0.0,
            scale_offset_formula="preprocessed_value = normalized_value * scale_factor + add_offset",
        )
    land = np.zeros((3, 4), dtype="u1")
    land[:, :2] = 1
    arrays = {
        "data": (data, ("time", "channel", "lat", "lon"), dataattrs, (1, count, 3, 4)),
        "time": (times, ("time",), {"timezone": "UTC"}, (12,)),
        "channel": (
            np.array(channel_names),
            ("channel",),
            {"channel_info_schema_version": "1.0", "channel_info": info},
            (count,),
        ),
        "lat": (np.array([90, 88.5, 87], dtype="f4"), ("lat",), {}, (3,)),
        "lon": (np.array([0, 1.5, 3, 4.5], dtype="f4"), ("lon",), {}, (4,)),
        "mean": (np.zeros(count, dtype="f4"), ("channel",), {}, (count,)),
        "std": (np.ones(count, dtype="f4"), ("channel",), {}, (count,)),
        "mask": (np.stack([land, 1 - land]), ("mask_channel", "lat", "lon"), {}, (2, 3, 4)),
        "mask_channel": (np.array(["land_mask", "sea_mask"]), ("mask_channel",), {}, (2,)),
    }
    for name, (values, dims, attrs, chunks) in arrays.items():
        group.create_array(
            name,
            data=values,
            chunks=chunks,
            dimension_names=dims,
            attributes=attrs,
            config={"write_empty_chunks": True},
        )
    zarr.consolidate_metadata(str(path))
    return path, write_production_evidence(path, spec)


def write_published_schema(published: tuple[Path, DatasetValidationSpec]):
    path, spec = published
    document = yaml.safe_load(Path(__file__).with_name("standard.yaml").read_text(encoding="utf-8"))
    meta = document["meta_info"]
    meta.update(
        dataset_id=f"{spec.product.product_id}.tiny.h6",
        expected_root_dataset_id=spec.dataset_id,
        product={"id": spec.product.product_id, "version": "1.0", "variant": "tiny.h6"},
        dataset_path=str(path),
        auxiliary_directory=str(path),
        data_revision="r1",
        time_coverage={
            "first": spec.first_time,
            "last": spec.last_time,
            "time_count": 12,
            "time_resolution": {"value": 6, "unit": "hour"},
            "timezone": "UTC",
        },
        channels=list(spec.product.channel_names),
        quality=[asdict(rule) for rule in spec.quality],
    )
    meta["normalization"]["effective_statistics"] = {
        name: {"mean": mean, "std": std}
        for name, (mean, std) in spec.product.effective_statistics.items()
    }
    if "q1000" not in spec.product.channel_names:
        meta.pop("pressure")
    variables = {channel.variable for channel in spec.product.channels}
    document["variable_metadata"] = {
        name: value for name, value in document["variable_metadata"].items() if name in variables
    }
    schema = path.parent / "fixture.yaml"
    schema.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    loaded = replace(
        load_validation_schema(schema)[0],
        snapshot_id=spec.snapshot_id,
        production_evidence=spec.production_evidence,
    )
    return path, loaded, schema, document


def evidence_reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_validation_run(schema: Path, spec: DatasetValidationSpec) -> Path:
    assert spec.production_evidence is not None
    run = schema.with_name("run.json")
    run.write_text(
        json.dumps(
            {
                "schema": schema.name,
                "execution": {
                    "snapshot_id": spec.snapshot_id,
                    "production_evidence": evidence_reference(spec.production_evidence.path),
                },
            }
        ),
        encoding="utf-8",
    )
    return run


def write_conversion_reference(schema: Path, spec: DatasetValidationSpec) -> tuple[Path, Path]:
    source, _ = write_published(
        schema.parent / "reference",
        dataset_id="reference",
        channel_names=("tp", "t2m", "q1000"),
    )
    group = zarr.open_group(str(source), mode="r+")
    grid = (np.arange(12, dtype="f2") / 12).reshape(3, 4)
    values = np.stack([grid * 8, grid * 2, grid * 4])[:, ::-1, :]
    group["data"][:] = np.tile(values[None], (12, 1, 1, 1))
    group["lat"][:] = [87.0, 88.5, 90.0]
    group["std"][:] = [8.0, 2.0, 4.0]
    info = group["channel"].attrs["channel_info"]
    for name, factor in (("tp", 8.0), ("t2m", 2.0), ("q1000", 4.0)):
        info[name]["scale_factor"] = factor
    group["channel"].attrs["channel_info"] = info
    zarr.consolidate_metadata(str(source))
    config = schema.with_name("conversion.json")
    config.write_text(
        json.dumps(
            {
                "target_schema": schema.name,
                "source_path": str(source.relative_to(schema.parent)),
                "execution": {"snapshot_id": spec.snapshot_id},
                "conversion": {
                    "source_identity": "independent-linear-reference",
                    "source_snapshot_id": "frozen-reference",
                    "rules_version": "known-powers-of-two-1",
                    "source_first_time": "2024-02-28T00",
                    "source_last_time": "2024-03-01T18",
                    "source_cadence_hours": 6,
                    "source_channels": ["tp", "t2m", "q1000"],
                    "latitude_flip": True,
                    "rules": [
                        {
                            "target_channel": name,
                            "source_channel": name,
                            "value_factor": factor,
                            "std_factor": factor,
                            "atol": 0,
                            "rtol": 0,
                        }
                        for name, factor in (("t2m", 0.5), ("q1000", 0.25), ("tp", 0.125))
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return source, config
