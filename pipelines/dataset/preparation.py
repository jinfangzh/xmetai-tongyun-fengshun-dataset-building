"""Standard assembly for the build pipeline, plus optional preparation tools."""

from dataclasses import replace

import numpy as np
import xarray as xr

from datakit import io
from datakit.schema import build_standard_channel_info, load_mapping
from datakit.transforms import compute_mean_std


def load_external_array(path, variable):
    """Load one auxiliary array completely before closing its source file."""
    if path is None:
        raise ValueError(f"An external {variable} path is required")
    with io.open(path).read(load=False) as source:
        return source[variable].load()


def prepare_statistics(data, spec, *, mean_path=None, std_path=None):
    """Load statistics or compute missing ones in the caller's processing domain.

    Computation requires exact target time coverage and reduces time/lat/lon
    with skipna=True, ddof=0. Select target channels and return float32 arrays;
    apply_zscore remains responsible for numeric and label validation.
    """
    mean = load_external_array(mean_path, "mean") if mean_path is not None else None
    std = load_external_array(std_path, "std") if std_path is not None else None
    channels = list(spec.product.channel_names)
    if mean is None or std is None:
        if not np.array_equal(data.time.values.astype("datetime64[ns]"), spec.expected_times()):
            raise ValueError("Source time coverage must exactly match target before computing statistics")
        computed_mean, computed_std = compute_mean_std(data, dim=("time", "lat", "lon"))
        mean = computed_mean if mean is None else mean
        std = computed_std if std is None else std
    return mean.sel(channel=channels).astype("float32"), std.sel(channel=channels).astype("float32")


def prepare_standard_arrays(data, mean, std, mask, spec):
    """Align prepared values and auxiliaries, then apply target types and chunks.

    Does not normalize, select coverage or generate standard metadata. The caller
    supplies processed values; assembly and the final guard check the result.
    """
    # Check before uint8 casting so values such as 257 cannot become valid masks.
    if mask.dtype.kind not in "biu" or not np.isin(mask.values, spec.product.mask.allowed_values).all():
        raise ValueError("Mask must contain only the declared binary integer values")
    data, mean, std, mask = xr.align(data, mean, std, mask, join="exact", copy=False)
    definition = spec.arrays["data"]
    data = data.transpose(*definition.dimensions).astype(definition.dtype)
    data = data.assign_coords(channel=data.channel.astype(str),
                              time=data.time.astype("datetime64[ns]"),
                              lat=data.lat.astype("float32"), lon=data.lon.astype("float32"))
    mean = mean.assign_coords(channel=data.channel)
    std = std.assign_coords(channel=data.channel)
    mask_coord = spec.product.mask.channel_array_name
    mask = mask.astype(spec.product.mask.dtype).assign_coords(
        {mask_coord: mask[mask_coord].astype(str), "lat": data.lat, "lon": data.lon})
    if data.chunks is not None and definition.chunks is not None:
        data = data.chunk({dim: data.sizes[dim] if size == "full" else size
                           for dim, size in zip(definition.dimensions, definition.chunks, strict=True)})
    # Explicit coordinate order controls Dataset's dimension display order;
    # each variable keeps its own contract-defined dimensions.
    coords = dict(data.coords)
    coords.update(mean.coords)
    coords.update(std.coords)
    coords.update(mask.coords)
    order = (*definition.dimensions, mask_coord)
    ordered_coords = {name: coords[name] for name in order}
    ordered_coords.update({name: coord for name, coord in coords.items() if name not in order})
    return xr.Dataset({"data": data.variable, "mean": mean.variable, "std": std.variable,
                       spec.product.mask.array_name: mask.variable}, coords=ordered_coords)


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
    order = (*spec.arrays["data"].dimensions, mask_coord)
    coords = {name: product.coords[name].copy(deep=True) for name in order}
    # Retain extra coordinates so the final guard can reject undeclared objects.
    coords.update({name: coord.copy(deep=True) for name, coord in product.coords.items() if name not in order})
    result = xr.Dataset({"data": data.variable, "mean": mean.variable, "std": std.variable,
                         spec.product.mask.array_name: mask.variable}, coords=coords)
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


def pack_source_channels(dataset, spec, source_mapping, *, depth_tolerance=0.1):
    """Select declared variables/levels and pack them in target channel order.

    Depth matching uses physical coordinates in metres, never channel-name codes
    or fixed file indices. Values retain their source units; mapping conversions
    must be performed in a separate explicit stage. Exact grid reindexing leaves
    absent locations as NaN (including the ORAS5 southern polar gap); this is not
    interpolation. The caller owns the source Dataset and its lazy file handles.
    """
    if source_mapping is None:
        raise ValueError("Raw input requires an explicit source mapping")
    if not np.isfinite(depth_tolerance) or depth_tolerance < 0:
        raise ValueError("depth_tolerance must be finite and nonnegative")
    mapping = load_mapping(source_mapping)
    renames = {
        alias: standard for alias, standard in mapping.coordinate_aliases.items()
        if alias != standard and (alias in dataset.dims or alias in dataset.coords)
    }
    source = dataset.rename(renames)
    for dimension in ("time", "lat", "lon"):
        if dimension not in source.indexes or source[dimension].dims != (dimension,):
            raise ValueError(f"Source requires labeled one-dimensional {dimension!r}")
        if not source.get_index(dimension).is_unique:
            raise ValueError(f"Source has duplicate {dimension!r} labels")
    if not np.isfinite(source.lat.values).all() or not np.isfinite(source.lon.values).all():
        raise ValueError("Source latitude/longitude must be finite")
    source = source.assign_coords(lon=((source.lon + 360.0) % 360.0)).sortby("lon")
    if not source.get_index("lon").is_unique:
        raise ValueError("Longitude wrapping produced duplicate labels")
    expected_times = spec.expected_times()
    missing_times = expected_times[~np.isin(expected_times, source.time.values)]
    if missing_times.size:
        raise ValueError(
            "Source does not contain the complete target time coverage; "
            f"missing {missing_times.size} timestamp(s), first={missing_times[0]}"
        )
    source = source.sel(time=expected_times)

    arrays = []
    for channel in spec.product.channels:
        declaration = mapping.spec_for(channel.variable)
        if declaration is None or declaration.source is None:
            raise ValueError(f"{channel.name}: source mapping must declare an explicit variable")
        if declaration.scale != 1 or declaration.offset != 0:
            raise ValueError(f"{channel.name}: packing does not execute mapping unit conversions")
        name = declaration.source
        if mapping.case_insensitive:
            matches = [key for key in source.data_vars if key.lower() == name.lower()]
            if len(matches) != 1:
                raise ValueError(f"{channel.name}: source variable {name!r} is absent or ambiguous")
            name = matches[0]
        if name not in source.data_vars:
            raise KeyError(f"{channel.name}: source variable {name!r} is missing")
        value = source[name]
        expected_units = declaration.source_unit or channel.source_units
        if expected_units and value.attrs.get("units") != expected_units:
            raise ValueError(f"{channel.name}: expected source units {expected_units!r}, "
                             f"observed {value.attrs.get('units')!r}")
        declared_units = declaration.unit if declaration.unit != "unknown" else declaration.source_unit
        if declared_units and channel.source_units != declared_units:
            raise ValueError(f"{channel.name}: source and target contracts disagree on source units")
        if channel.level_type in {"depth_below_sea_surface", "depth_below_sea"}:
            if channel.level is None or channel.level_units != "m":
                raise ValueError(f"{channel.name}: target depth must be declared in metres")
            if "depth" not in value.indexes or value.depth.dims != ("depth",):
                raise ValueError(f"{channel.name}: source requires a labeled depth coordinate")
            depths = np.asarray(value.depth.values, dtype="float64")
            if not depths.size or not np.isfinite(depths).all() or not value.get_index("depth").is_unique:
                raise ValueError(f"{channel.name}: source depths must be finite, nonempty and unique")
            if value.depth.attrs.get("units") != "m":
                raise ValueError(f"{channel.name}: source depth units must be 'm'")
            distances = np.abs(depths - channel.level)
            nearest = np.flatnonzero(np.isclose(distances, distances.min(), rtol=0, atol=1e-10))
            if nearest.size != 1 or distances[nearest[0]] > depth_tolerance:
                raise ValueError(f"{channel.name}: cannot uniquely match depth {channel.level} m "
                                 f"within {depth_tolerance} m")
            value = value.isel(depth=int(nearest[0]), drop=True)
        elif channel.level_type == "pressure":
            if channel.level_units != "hPa" or "level" not in value.indexes:
                raise ValueError(f"{channel.name}: source requires labeled pressure levels in hPa")
            if value.level.attrs.get("units") != "hPa":
                raise ValueError(f"{channel.name}: source pressure units must be 'hPa'")
            try:
                value = value.sel(level=channel.level, drop=True)
            except KeyError as exc:
                raise ValueError(f"{channel.name}: pressure level {channel.level} hPa is missing") from exc
        elif channel.level_type not in {"surface", "height_above_ground"}:
            raise ValueError(f"{channel.name}: unsupported packing level type {channel.level_type!r}")
        if set(value.dims) != {"time", "lat", "lon"}:
            raise ValueError(f"{channel.name}: expected source dimensions time/lat/lon, got {value.dims}")
        arrays.append(value.expand_dims(channel=[channel.name]))

    lat, lon = spec.variant.grid.coordinates()
    packed = xr.concat(arrays, dim="channel", join="exact", combine_attrs="drop")
    return packed.reindex(lat=lat, lon=lon).transpose("time", "channel", "lat", "lon").astype("float32")
