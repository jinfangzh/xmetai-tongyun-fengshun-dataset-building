"""Standard assembly for the build pipeline, plus optional preparation tools."""

from dataclasses import replace

import numpy as np
import xarray as xr

from datakit.schema import build_standard_channel_info


def assemble_standard_dataset(product: xr.Dataset, spec) -> xr.Dataset:
    """Assemble already prepared standard arrays and declare target metadata.

    Does not select/reorder channels, cast types or repair coordinates. Shared
    coordinates must align exactly. Contract-owned root/data/channel attributes
    are replaced on the result; unrelated variable/coordinate attributes remain.
    UTC and geographic bounds are contract declarations, not source verification.
    A result still requires the full target guard before writing.
    """
    if not isinstance(product, xr.Dataset):
        raise TypeError("process must return an xarray Dataset")
    if not {"data", "mean", "std"}.issubset(product.data_vars):
        raise ValueError("Processed Dataset must contain data, mean and std")
    expected = {"data", "mean", "std", spec.product.mask.array_name}
    if set(product.data_vars) != expected:
        raise ValueError(f"Processed Dataset must contain exactly {sorted(expected)}")
    data, mean, std = product["data"], product["mean"], product["std"]
    mask = product[spec.product.mask.array_name]
    arrays = {name: product[name] for name in expected}
    layouts = spec.array_layouts()
    for name, array in arrays.items():
        dims, shape, dtype = layouts[name]
        if tuple(array.dims) != dims or array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"{name} must already have target dimensions, shape and dtype")
    if tuple(str(v) for v in data.channel.values) != spec.product.channel_names:
        raise ValueError("Data channels must already be in target order")
    mask_coord = spec.product.mask.channel_array_name
    if tuple(str(v) for v in mask[mask_coord].values) != spec.product.mask.channels:
        raise ValueError("Mask channels must already be in target order")
    for name in ("time", "lat", "lon", "channel", mask_coord):
        array = mask if name == mask_coord else data
        dims, shape, dtype = layouts[name]
        coord = array[name]
        dtype_ok = coord.dtype.kind in "UST" if dtype == "string" else coord.dtype == np.dtype(dtype)
        if coord.dims != dims or coord.shape != shape or not dtype_ok:
            raise ValueError(f"{name} must already have target layout and dtype")
    data, mean, std, mask = xr.align(data, mean, std, mask, join="exact", copy=False)
    # Deep-copy metadata-bearing coordinates only; main data can remain lazy.
    result = data.rename("data").to_dataset().copy(deep=False)
    result["mean"], result["std"], result[spec.product.mask.array_name] = mean, std, mask
    # Retain extra coordinates so the final guard can reject undeclared objects.
    result = result.assign_coords({name: coord.copy(deep=True) for name, coord in product.coords.items()})
    lat, lon = spec.variant.grid.coordinates()
    result.attrs = {
        "dataset_id": spec.dataset_id, "schema_version": spec.schema_version,
        "content_version": spec.content_version, "data_revision": spec.data_revision,
        "geospatial_lat_range": [float(lat.min()), float(lat.max())],
        "geospatial_lon_range": [float(lon[0]), float(lon[-1])],
    }
    result.data.attrs = dict(spec.data_attributes)
    result.time.attrs["timezone"] = "UTC"
    metadata = spec.channel_metadata
    result.channel.attrs = {
        metadata.version_attribute: metadata.version,
        metadata.attribute: build_standard_channel_info(
            # Describe the actual statistics; the unchanged target guard checks
            # fixed-statistic identity, returning a report on a mismatch.
            replace(spec.product, effective_statistics={}),
            mean=mean.values, std=std.values, channel_metadata=metadata,
        ),
    }
    return result


def generate_mask(dataset, spec, variable, threshold):
    """Optional helper for recipes that explicitly use a source land variable.

    Not automatically invoked by the built-in ERA5/ORAS5 conversions.
    """
    if variable is None:
        raise ValueError("mask_variable is required when mask_path is None")
    if variable in dataset.data_vars:
        land = dataset[variable]
    elif "data" in dataset and "channel" in dataset.data.dims:
        land = dataset.data.sel(channel=variable, drop=True)
    else:
        raise ValueError(f"Source land-mask variable is missing: {variable}")
    # A single time slice may describe a static field; never guess across times.
    if "time" in land.dims and land.sizes["time"] == 1:
        land = land.isel(time=0, drop=True)
    if set(land.dims) != {"lat", "lon"}:
        raise ValueError("Source land mask must be static with lat/lon dimensions")
    land = land.transpose("lat", "lon")
    values = land.values
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Source land mask must contain finite values in [0, 1]")
    binary = np.isin(values, [0, 1]).all()
    if threshold is not None:
        if isinstance(threshold, (bool, np.bool_)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("mask_threshold must be finite and in [0, 1]")
    if binary:
        land = land.astype("uint8")
    else:
        if threshold is None:
            raise ValueError("Fractional source mask requires an explicit mask_threshold")
        land = (land >= threshold).astype("uint8")
    schema = spec.product.mask
    fields = {schema.land_channel: land, schema.sea_channel: 1 - land}
    return xr.concat([fields[name] for name in schema.channels],
                     dim=xr.IndexVariable(schema.channel_array_name, list(schema.channels)),
                     join="exact").astype(schema.dtype).rename(schema.array_name)
