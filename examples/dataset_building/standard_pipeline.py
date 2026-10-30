"""Small-data building example with explicit pipeline stages.

Load contracts -> read -> process -> write -> validate. Pass a processor returning
an entire target Dataset; mean/std/mask paths belong to that processor, not to the
pipeline. Source mapping is not applied implicitly. Inputs to the built-in ERA5
and ORAS5 processors are packed physical data(time, channel, lat, lon).

Use matching target coverage/channels; legacy ORAS5 so19979 is not os19978.
No overwrite, streaming append or enforced memory budget. Final validation is a
quick read-back check, including declared external auxiliaries, not full release
certification. dry_run only checks the in-memory product and writes nothing.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import xarray as xr

from datakit import io
from datakit.schema import DatasetSchema, build_standard_channel_info, load_mapping, load_schema
from datakit.transforms import normalize_xarray, preprocess_channel_units
from pipelines.dataset import create_zarr, validate_zarr


def _load_array(path, variable):
    with io.open(path).read(load=True) as opened:
        return opened[variable].load().copy(deep=True)


def _statistic(array, name, channels):
    if array.dims != ("channel",) or "channel" not in array.coords:
        raise ValueError(f"{name} must have labeled, one-dimensional channels")
    if not array.get_index("channel").is_unique:
        raise ValueError(f"{name} channels must be unique")
    result = array.sel(channel=list(channels)).astype("float32")
    if not np.isfinite(result.values).all() or (name == "std" and not (result.values > 0).all()):
        raise ValueError(f"{name} must be finite; std must also be positive")
    return result


def _packed_data(dataset):
    data = dataset["data"]
    if set(data.dims) != {"time", "channel", "lat", "lon"}:
        raise ValueError("Expected packed data(time, channel, lat, lon)")
    if "channel" not in data.coords or not data.get_index("channel").is_unique:
        raise ValueError("Source must have unique labeled channels")
    return data.astype("float32")


def _era5_values(dataset, spec, mean, std, *, accumulation_hours=6.0):
    """Derive wind speed, convert humidity/radiation/rain, then normalize.

    The accumulation window is an explicit processing choice. Target recipes must
    agree with these fixed operations; arbitrary YAML steps are not executed.
    """
    if not np.isfinite(accumulation_hours) or accumulation_hours <= 0:
        raise ValueError("accumulation_hours must be finite and positive")
    data = _packed_data(dataset)
    zscore = {"operation": "zscore", "mean": "/mean", "std": "/std"}
    derived = []
    for channel in spec.product.channels:
        expected = [zscore]
        if channel.variable in {"ws10m", "ws100m"}:
            height = "10m" if channel.variable == "ws10m" else "100m"
            left, right = f"u{height}", f"v{height}"
            expected = [{"operation": "hypot", "inputs": [left, right]}, zscore]
            wind = np.hypot(data.sel(channel=left, drop=True), data.sel(channel=right, drop=True))
            derived.append(wind.expand_dims(channel=[channel.name]))
        elif channel.variable == "q":
            expected = [{"operation": "multiply", "factor": 1000.0,
                         "input_units": "kg kg-1", "output_units": "g kg-1"}, zscore]
        elif channel.variable in {"ssr", "ssrd", "fdir", "ttr"}:
            seconds = accumulation_hours * 3600
            expected = [{"operation": "divide", "divisor": seconds,
                         "accumulation_window_seconds": seconds,
                         "input_units": "J m-2", "output_units": "W m-2"}, zscore]
        elif channel.variable == "tp":
            expected = [{"operation": "multiply", "factor": 1000.0,
                         "input_units": "m", "output_units": "mm"},
                        {"operation": "clip_min", "minimum": 0.0}, {"operation": "log1p"}]
        if list(channel.preprocessing) != expected:
            raise ValueError(f"{channel.name}: target recipe differs from process_era5")
    if derived:
        names = [str(array.channel.values[0]) for array in derived]
        data = xr.concat([data.drop_sel(channel=[n for n in names if n in data.channel]),
                          *derived], dim="channel", join="exact")
    data = data.sel(channel=list(spec.product.channel_names))
    data = preprocess_channel_units(data, accumulation_hours=accumulation_hours)
    return normalize_xarray(data, mean, std, channel_dim="channel", log_channels=())


def _oras5_values(dataset, spec, mean, std):
    """Select matching ocean channels and apply zscore, without unit conversion."""
    expected = [{"operation": "zscore", "mean": "/mean", "std": "/std"}]
    if any(list(channel.preprocessing) != expected for channel in spec.product.channels):
        raise ValueError("process_oras5 supports only unconverted zscore channels")
    data = _packed_data(dataset).sel(channel=list(spec.product.channel_names))
    return normalize_xarray(data, mean, std, channel_dim="channel", log_channels=())


def assemble_product(data, spec, mean, std, mask):
    """Attach schema-bound metadata without relabeling times, grids or masks."""
    channels = list(spec.product.channel_names)
    data = data.sel(channel=channels).transpose(*spec.arrays["data"].dimensions)
    lat, lon = spec.variant.grid.coordinates()
    # Classic NetCDF may decode binary masks as signed integers. Check values
    # before casting so invalid values cannot wrap into a valid uint8 label.
    if mask.dtype.kind not in "biu" or not np.isin(mask.values, spec.product.mask.allowed_values).all():
        raise ValueError("Mask must contain only the declared binary integer values")
    mask = mask.astype(spec.product.mask.dtype)
    # Exact alignment prevents Xarray from silently unioning shifted coordinates.
    data, mean, std, mask = xr.align(data, mean, std, mask, join="exact", copy=False)
    result = data.astype(spec.arrays["data"].dtype).rename("data").to_dataset()
    result["mean"], result["std"], result["mask"] = mean, std, mask
    result = result.assign_coords(
        channel=result.channel.astype(str), mask_channel=result.mask_channel.astype(str),
        time=result.time.astype("datetime64[ns]"),
        lat=result.lat.astype("float32"), lon=result.lon.astype("float32"),
    )
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
            spec.product, mean=mean.values, std=std.values, channel_metadata=metadata,
        ),
    }
    return result


def _prepare_product(dataset, spec, transform, mean_path, std_path, mask_path):
    channels = spec.product.channel_names
    mean = _statistic(_load_array(mean_path, "mean"), "mean", channels)
    std = _statistic(_load_array(std_path, "std"), "std", channels)
    mask = _load_array(mask_path, "mask")
    data = transform(dataset, spec, mean, std)
    return assemble_product(data, spec, mean, std, mask)


def process_era5(dataset, spec, *, mean_path, std_path, mask_path, accumulation_hours=6.0):
    """ERA5 processing stage: physical values and auxiliaries -> complete product."""
    transform = partial(_era5_values, accumulation_hours=accumulation_hours)
    return _prepare_product(dataset, spec, transform, mean_path, std_path, mask_path)


def process_oras5(dataset, spec, *, mean_path, std_path, mask_path):
    """ORAS5 processing stage: matching ocean channels -> complete product."""
    return _prepare_product(dataset, spec, _oras5_values, mean_path, std_path, mask_path)


def load_source_contract(source_config):
    """Load the source declaration and check its IO layout."""
    description = io.get_description(source_config)
    io.get_layout(description)
    return description


def load_target_contract(target_spec):
    """Load the target schema and check its configuration."""
    spec = target_spec if isinstance(target_spec, DatasetSchema) else load_schema(target_spec)[0]
    spec.validate_configuration()
    return spec


def read_source(source_path, description):
    """Stage 2: use the declared IO layout; caller must close the Dataset.

    File discovery/format checks belong to IO. A complete semantic source-data
    check is not yet available from the current source-layout declarations.
    """
    return io.open_dataset(source_path, description).read(all_files=True)


def validate_source(source_path, source_config):
    """Check actual source files before processing; return a scoped summary.

    Version 1 checks path/layout discovery, readability, unique dimension labels,
    declared packed storage and source units for present mapped variables. Mapping
    entries are optional mappings, not a required-variable list. No transformations
    or value scans are performed. Undeclared coverage, missing expected dates,
    temporal semantics and physical ranges are not certified by this interface.
    Raise on a mismatch so the pipeline cannot proceed to process/write.
    """
    path = Path(source_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source path does not exist: {path}")
    description = io.get_description(source_config)
    access = io.open_dataset(path, description)
    records = access.discover(all_files=True)
    if not records:
        raise ValueError(f"No files match source declaration: {path}")
    mapping = load_mapping(description["mapping"]) if "mapping" in description else None
    checked = 0
    # Closing the iterator also closes the current file if a check raises.
    from contextlib import closing

    with closing(access.iter_datasets(all_files=True)) as datasets:
        for dataset in datasets:
            file = records[checked].path
            try:
                for name, index in dataset.indexes.items():
                    if not index.is_unique:
                        raise ValueError(f"Duplicate coordinate labels: {name}")
                if mapping is not None:
                    packed = mapping.packed_variable
                    if packed and (packed not in dataset or mapping.packed_dimension not in dataset[packed].dims):
                        raise ValueError(f"Expected {packed!r} with dimension {mapping.packed_dimension!r}")
                    for standard_name, variable in mapping.variables.items():
                        name = variable.source
                        if name is None:
                            if len(dataset.data_vars) != 1:
                                continue
                            name = next(iter(dataset.data_vars))
                        if mapping.case_insensitive:
                            name = next((n for n in dataset.data_vars if str(n).lower() == name.lower()), name)
                        if name not in dataset.data_vars:
                            continue
                        units = variable.source_unit
                        if not units and variable.scale == 1 and variable.offset == 0:
                            units = variable.unit if variable.unit != "unknown" else ""
                        if units and dataset[name].attrs.get("units") != units:
                            raise ValueError(
                                f"{standard_name}: expected source units {units!r}, "
                                f"observed {dataset[name].attrs.get('units')!r}"
                            )
            except ValueError as exc:
                raise ValueError(f"Source contract mismatch in {file}: {exc}") from exc
            checked += 1
    if checked != len(records):
        raise ValueError(f"Source contains empty/unvalidated files: {checked}/{len(records)} checked at {path}")
    return {"path": str(path), "files_checked": checked,
            "scope": "layout, readability, coordinate uniqueness, declared packed structure and present-variable units"}


def write_product(product, destination, spec, *, dry_run=False):
    """Stage 4: check conformance and write without overwriting existing data."""
    return create_zarr(product, destination, spec, dry_run=dry_run)


def validate_product(destination, spec, creation_report):
    """Stage 5: check the written product, or retain dry-run/failure diagnostics.

    Quick mode samples main-data blocks; declared external auxiliaries are also
    checked. It cannot authorize release. Failed creation is never re-opened.
    """
    if not creation_report.coverage["written"]:
        return creation_report
    report = validate_zarr(destination, spec, mode="quick")
    report.coverage["written"] = True
    report.coverage["conformance_passed"] = creation_report.coverage["conformance_passed"]
    return report


def build_dataset(
    source_config,
    target_spec,
    source_path,
    destination,
    *,
    process: Callable,
    dry_run: bool = False,
):
    """按源契约检查输入，调用处理函数构建目标数据，写出后校验。

    process(raw, spec) 返回完整的目标 Dataset；统计量、掩膜等专用参数
    由调用方通过 functools.partial 绑定。输出目录名仍须符合目标契约。
    dry_run 或写出失败时返回创建检查报告，成功写出后返回 quick 校验报告。
    """
    # 路径准备：隔离输入与输出，保护已有数据。
    source = Path(source_path).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Source and destination paths must not overlap")
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)

    # 输入检查：加载源契约，确认实际文件符合其中已声明的要求。
    description = load_source_contract(source_config)
    validate_source(source, description)

    # 数据构建：按目标契约处理源数据，检查内存产物并写出，随后释放输入资源。
    # dry_run 只检查产物，不写出。
    spec = load_target_contract(target_spec)
    with read_source(source, description) as raw:
        product = process(raw, spec)
        creation_report = write_product(product, target, spec, dry_run=dry_run)

    # 结果校验：检查已写出的产物；未写出时返回创建检查报告。
    return validate_product(target, spec, creation_report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("source-config", "target-spec", "source", "destination", "mean-nc", "std-nc", "mask-nc"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--processor", choices=("era5", "oras5"), required=True)
    parser.add_argument("--accumulation-hours", type=float, default=6.0, help="ERA5 radiation window")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    processor = process_era5 if args.processor == "era5" else process_oras5
    options = {"accumulation_hours": args.accumulation_hours} if args.processor == "era5" else {}
    process = partial(processor, mean_path=args.mean_nc, std_path=args.std_nc,
                      mask_path=args.mask_nc, **options)
    report = build_dataset(
        args.source_config, args.target_spec, args.source, args.destination,
        process=process, dry_run=args.dry_run,
    )
    print(report.to_dict())
    success = (report.coverage["conformance_passed"] if args.dry_run
               else report.coverage["written"] and report.quick_passed)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
