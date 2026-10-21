"""Independently constructed, physically written Zarr fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from datakit.schema.standard_dataset import (
    ChannelDefinition,
    DatasetProduct,
    DatasetValidationSpec,
    DatasetVariant,
    GridDefinition,
    QualityRule,
)

from ._helpers import _receipt


@pytest.fixture
def published(tmp_path):
    # Expectations are independent of the metadata builder under test.
    product = DatasetProduct(
        "fixture.c3",
        "1.0",
        (
            ChannelDefinition("t2m", "t2m", "K", "K", "height_above_ground", 2, "m"),
            ChannelDefinition("q1000", "q", "g kg-1", "kg kg-1", "pressure", 1000, "hPa"),
            ChannelDefinition("tp", "tp", "1", "m", "surface", variable_type="accumulated"),
        ),
        (DatasetVariant("tiny.h6", GridDefinition("tiny", 3, 4, 1.5), 6),),
    )
    spec = DatasetValidationSpec(
        "fixture",
        product,
        "tiny.h6",
        "2024-02-28T00",
        "2024-03-01T18",
        "r1",
        mask_provenance="independent-fixture-mask",
        quality=tuple(
            QualityRule(
                name, "all", 0.0, False, "independent-fixture-limits", (-10.0, 10.0), "stored"
            )
            for name in product.channel_names
        ),
    )
    path = tmp_path / "data.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.attrs.update(
        dataset_id="fixture",
        schema_version="2.0",
        content_version="v3",
        data_revision="r1",
        geospatial_lat_range=[87.0, 90.0],
        geospatial_lon_range=[0.0, 4.5],
    )
    data = np.tile(np.arange(12, dtype="f2").reshape(1, 1, 3, 4) / 12, (12, 3, 1, 1))
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
        "data": (data, ("time", "channel", "lat", "lon"), dataattrs, (1, 3, 3, 4)),
        "time": (spec.expected_times(), ("time",), {"timezone": "UTC"}, (12,)),
        "channel": (
            np.array(["t2m", "q1000", "tp"]),
            ("channel",),
            {"channel_info_schema_version": "1.0", "channel_info": info},
            (3,),
        ),
        "lat": (np.array([90, 88.5, 87], dtype="f4"), ("lat",), {}, (3,)),
        "lon": (np.array([0, 1.5, 3, 4.5], dtype="f4"), ("lon",), {}, (4,)),
        "mean": (np.zeros(3, dtype="f4"), ("channel",), {}, (3,)),
        "std": (np.ones(3, dtype="f4"), ("channel",), {}, (3,)),
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
    return path, _receipt(path, spec)
