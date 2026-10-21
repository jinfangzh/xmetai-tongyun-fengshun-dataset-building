"""Target Zarr v2 schema for refreshed datasets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numcodecs import Blosc
import zarr

from .channels import build_channel_info
SCHEMA_VERSION = "2.0"
CONTENT_VERSION = "v3"
MASK_CHANNELS = ("land_mask", "sea_mask")
DATA_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["time", "channel", "lat", "lon"],
    "long_name": "meteorological fields",
    "data_representation": "normalized",
    "normalization_method": "channel_dependent",
    "mean_variable": "mean",
    "std_variable": "std",
    "scale_factor_location": "channel.channel_info[*].scale_factor",
    "add_offset_location": "channel.channel_info[*].add_offset",
    "inverse_transform": "preprocessed_value = data * scale_factor + add_offset",
}
LAT_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["lat"],
    "standard_name": "latitude",
    "long_name": "latitude",
    "units": "degrees_north",
    "axis": "Y",
}
LON_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["lon"],
    "standard_name": "longitude",
    "long_name": "longitude",
    "units": "degrees_east",
    "axis": "X",
}
TIME_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["time"],
    "standard_name": "time",
    "long_name": "time",
    "axis": "T",
    "timezone": "UTC",
}
MEAN_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["channel"],
    "alignment": "index-aligned with /channel",
    "long_name": "Channel normalization mean",
}
STD_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["channel"],
    "alignment": "index-aligned with /channel",
    "long_name": "Channel normalization standard deviation",
}
MASK_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["mask_channel", "lat", "lon"],
    "long_name": "Binary spatial masks",
    "mask_type": "binary",
    "mask_channel_variable": "mask_channel",
    "source_variable": "lsm",
    "derivation": "land_mask = lsm > 0.5; sea_mask = 1 - land_mask",
}
MASK_CHANNEL_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["mask_channel"],
    "long_name": "Mask channel name",
    "description": "Labels aligned with mask[mask_channel, :, :]",
}


def codec() -> Blosc:
    return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


def channel_info(
    channels: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    radiation_seconds: int | None,
) -> dict[str, dict[str, Any]]:
    return build_channel_info(channels, mean, std, radiation_seconds)


def create_refreshed_store(
    path: Path,
    *,
    dataset_id: str,
    data_revision: str,
    times: np.ndarray,
    channels: Sequence[str],
    lat: np.ndarray,
    lon: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    land_mask: np.ndarray,
    radiation_seconds: int | None,
) -> zarr.Group:
    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    group.attrs.update(
        {
            "dataset_id": dataset_id,
            "schema_version": SCHEMA_VERSION,
            "content_version": CONTENT_VERSION,
            "data_revision": data_revision,
            "geospatial_lat_range": [float(np.min(lat)), float(np.max(lat))],
            "geospatial_lon_range": [float(np.min(lon)), float(np.max(lon))],
        }
    )
    channel_count = len(channels)
    ny, nx = len(lat), len(lon)
    group.create_array(
        "data",
        shape=(len(times), channel_count, ny, nx),
        chunks=(1, channel_count, ny, nx),
        dtype="f2",
        fill_value=np.nan,
        compressor=codec(),
        chunk_key_encoding={"name": "v2", "configuration": {"separator": "."}},
        attributes=DATA_ATTRIBUTES,
    )
    group.create_array(
        "channel",
        data=np.asarray(channels, dtype=str),
        chunks=(channel_count,),
        fill_value="",
        attributes={
            "_ARRAY_DIMENSIONS": ["channel"],
            "long_name": "Weather variable channel code",
            "channel_info_schema_version": "1.0",
            "channel_info": channel_info(channels, mean, std, radiation_seconds),
        },
    )
    group.create_array(
        "lat",
        data=np.asarray(lat, dtype="f4"),
        chunks=(ny,),
        fill_value=np.nan,
        attributes=LAT_ATTRIBUTES,
    )
    group.create_array(
        "lon",
        data=np.asarray(lon, dtype="f4"),
        chunks=(nx,),
        fill_value=np.nan,
        attributes=LON_ATTRIBUTES,
    )
    times_ns = np.asarray(times).astype("datetime64[ns]")
    group.create_array(
        "time",
        data=times_ns,
        chunks=times_ns.shape,
        fill_value=np.datetime64("NaT", "ns"),
        attributes=TIME_ATTRIBUTES,
    )
    group.create_array(
        "mean",
        data=np.asarray(mean, dtype="f4"),
        chunks=(channel_count,),
        compressor=codec(),
        fill_value=np.nan,
        attributes=MEAN_ATTRIBUTES,
    )
    group.create_array(
        "std",
        data=np.asarray(std, dtype="f4"),
        chunks=(channel_count,),
        compressor=codec(),
        fill_value=np.nan,
        attributes=STD_ATTRIBUTES,
    )
    group.create_array(
        "mask_channel",
        data=np.asarray(MASK_CHANNELS, dtype=str),
        chunks=(2,),
        fill_value="",
        attributes=MASK_CHANNEL_ATTRIBUTES,
    )
    masks = np.stack((land_mask.astype("u1"), np.uint8(1) - land_mask.astype("u1")))
    group.create_array(
        "mask",
        data=masks,
        chunks=(1, ny, nx),
        compressor=codec(),
        fill_value=np.uint8(0),
        attributes=MASK_ATTRIBUTES,
    )
    return group
