"""Non-mutating Xarray conformance checks followed by schema-bound writing."""

import json
from pathlib import Path

import numpy as np
import xarray as xr

from datakit.schema import DatasetSchema, load_schema, verify_schema_unchanged
from datakit.validation.structure import (
    STANDARD,
    check_channel_semantics,
    check_dataset_format,
    check_mask_labels,
    check_region_layout,
)
from datakit.validation.values import (
    StandardQualityAccumulator,
    check_mask_values,
    check_statistics,
)
from datakit.validation.zarr_inspection import file_sha256

from .configuration import DatasetValidationSpec, validation_input_files
from .evidence import (
    check_dataset_identity,
    check_product_confirmation,
)
from .reports import ValidationReport, ensure_report_destination


def _check_input(dataset, spec, report):
    channel_ok, checks = check_dataset_format(dataset, spec)
    report.checks.extend(checks)
    if channel_ok is None:
        return
    mean, std = dataset["mean"].values, dataset["std"].values
    stats_ok, checks = check_statistics(mean, std)
    report.checks.extend(checks)
    report.checks.extend(check_mask_labels(dataset, spec))
    mask = dataset[spec.product.mask.array_name].values
    mask_ok, checks = check_mask_values(mask, spec)
    report.checks.extend(checks)
    report.checks.extend(
        check_channel_semantics(dataset, spec, channel_ok=channel_ok, stats_ok=stats_ok)
    )
    check_product_confirmation(spec, report)
    if not (channel_ok and stats_ok and mask_ok):
        return
    regions = {}
    for name, reference in spec.valid_regions.items():
        if not report.require(
            f"quality.region.{name}.source",
            file_sha256(reference.path) == reference.sha256,
            "data_quality",
            reference.sha256,
        ):
            return
        region = np.load(reference.path, allow_pickle=False)
        region_ok, checks = check_region_layout(region, name, spec)
        report.checks.extend(checks)
        if not region_ok:
            return
        regions[name] = region
    quality = StandardQualityAccumulator(spec, mean, std, mask, regions)
    # The schema supports complete spatial/channel frames as logical blocks.
    for index in range(dataset.sizes["time"]):
        quality.update(
            np.asarray(dataset["data"].isel(time=slice(index, index + 1)).values)
        )
        report.coverage["frames_checked"] += 1
    checks, metrics = quality.finish()
    report.checks.extend(checks)
    report.quality.update(metrics)


def create_zarr(
    dataset, destination, spec, *, dry_run=False, output=None, write_options=None
):
    """Check a prepared Dataset against a required spec, then create Zarr.

    Return a ValidationReport with coverage.conformance_passed and written.
    dry_run performs checks only. Input values, coordinates, dtypes and business
    attributes are never adapted. Chunks/version follow the spec; no overwrite.
    External auxiliaries and publication evidence need separate validate_zarr.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("create_zarr requires an xr.Dataset prepared by the caller")
    if isinstance(spec, (str, Path)):
        spec, _ = load_schema(spec)
    if not isinstance(spec, DatasetSchema):
        raise TypeError("spec must be a DatasetSchema or a schema YAML path")
    if not isinstance(spec, DatasetValidationSpec):
        spec = DatasetValidationSpec.from_schema(spec)
    spec.validate_configuration()
    target = Path(destination).expanduser().resolve()
    options = dict(write_options or {})
    # Storage options must not replace schema decisions or CF-encode values.
    reserved = {
        "mode",
        "append_dim",
        "region",
        "encoding",
        "chunks",
        "zarr_format",
        "zarr_version",
        "format",
        "compute",
        "chunk_store",
        "store",
        "group",
        "write_empty_chunks",
    }
    if reserved & options.keys():
        raise ValueError(
            f"Schema-bound creation cannot override {sorted(reserved & options.keys())}"
        )
    options.setdefault("consolidated", False)
    files = validation_input_files(spec)
    if any(p is not None and p.resolve().is_relative_to(target) for p in files):
        raise ValueError("Destination must not contain schema or evidence inputs")
    report_path = None
    if output is not None:
        report_path = Path(output).expanduser().absolute()
        sources = [target]
        source = dataset.encoding.get("source")
        if source:
            files += (Path(source),)
            if Path(source).is_dir():
                sources.append(Path(source))
        for array in dataset.variables.values():
            source = array.encoding.get("source")
            if source:
                files += (Path(source),)
                if Path(source).is_dir():
                    sources.append(Path(source))
        ensure_report_destination(report_path, directories=sources, files=files)
        if report_path.exists() or report_path.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite an existing report: {report_path}"
            )

    report = ValidationReport(
        spec.dataset_id,
        "create",
        {
            "destination": str(target),
            "validation_scope": spec.validation_scope,
            "checked_scope": "input Dataset; external auxiliaries/publication/storage audit excluded",
        },
        coverage={
            "dry_run": dry_run,
            "conformance_passed": False,
            "written": False,
            "frames_checked": 0,
        },
    )
    check_dataset_identity(target.name, spec, report)
    try:
        _check_input(dataset, spec, report)
        if spec.schema_path is not None:
            verify_schema_unchanged(spec)
    except Exception as exc:
        report.add(
            "create.input_check",
            "fail",
            STANDARD,
            evidence={"type": type(exc).__name__, "message": str(exc)},
        )
    passed = bool(report.checks) and all(
        check.status in ("pass", "warning", "not-applicable") for check in report.checks
    )
    report.coverage["conformance_passed"] = passed
    report.require(
        "create.destination_available",
        not target.exists(),
        "data_integrity",
        "new destination",
        str(target),
    )
    if passed and not target.exists() and not dry_run:
        # Work on a shallow copy, discarding inherited storage/CF encodings.
        # In particular, an inherited dtype or scale_factor must not alter values.
        payload = dataset.copy(deep=False)
        for name in payload.variables:
            payload[name].encoding = {}
        encoding = {}
        for name, definition in spec.arrays.items():
            settings = {"_FillValue": None}
            if definition.chunks is not None:
                settings["chunks"] = tuple(
                    dataset.sizes[dim] if chunk == "full" else chunk
                    for dim, chunk in zip(
                        definition.dimensions, definition.chunks, strict=True
                    )
                )
            encoding[name] = settings
        try:
            payload.to_zarr(
                str(target),
                mode="w-",
                zarr_format=spec.zarr_format,
                encoding=encoding,
                write_empty_chunks=True,
                **options,
            )
            report.coverage["written"] = True
            report.add("create.write", "pass", "data_integrity")
        except Exception as exc:
            report.add(
                "create.write",
                "fail",
                "data_integrity",
                evidence={"type": type(exc).__name__, "message": str(exc)},
            )
    if report_path is not None:
        report.identity["report_path"] = str(report_path)
        content = json.dumps(
            report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as stream:
            stream.write(content + "\n")
    return report
