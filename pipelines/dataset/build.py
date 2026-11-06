"""Contract guard and resource-safe dataset construction; no data adaptation."""

import gc
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

from yaml import YAMLError

from datakit import io
from datakit.schema import DatasetSchema, load_mapping, load_schema
from .create import create_zarr
from .validate import validate_zarr
from .reports import BuildReport
from .preparation import assemble_standard_dataset
from .auxiliaries import copy_auxiliaries, prepare_auxiliaries


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


def resolve_build_contract(spec, destination, overrides=None):
    """Bind task coverage/identity to a standard without changing product rules."""
    values = dict(overrides or {})
    allowed = {"first_time", "last_time", "dataset_id", "content_version", "data_revision"}
    if set(values) - allowed:
        raise ValueError(f"Unsupported build overrides: {sorted(set(values) - allowed)}")
    if ("first_time" in values) != ("last_time" in values):
        raise ValueError("Build coverage requires both first_time and last_time")
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        raise ValueError("Build overrides must be nonempty strings")
    if not values:
        return spec
    if "first_time" in values and "dataset_id" not in values:
        raise ValueError("Build coverage overrides require an explicit dataset_id")
    identity_changed = (values.get("dataset_id", spec.dataset_id) != spec.dataset_id
                        or Path(destination).name != spec.directory_name)
    result = replace(spec, **values, directory_name=Path(destination).name,
                     identity_mapping_evidence=None if identity_changed else spec.identity_mapping_evidence)
    result.validate_configuration()
    return result


def read_source(source_path, description):
    """Stage 2: use the declared IO layout; caller must close the Dataset.

    File discovery/format checks belong to IO. A complete semantic source-data
    check is not yet available from the current source-layout declarations.
    """
    return io.open(source_path, description).read(all_files=True)


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
    access = io.open(path, description)
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


def build_zarr(
    source_config,
    target_spec,
    source_path,
    destination,
    *,
    process: Callable,
    read: Callable | None = None,
    dry_run: bool = False,
    build_overrides=None,
    auxiliaries=None,
):
    """显式编排各阶段，返回统一 BuildReport。

    process(raw, description, spec) 返回已加工的 data/mean/std/mask Dataset；标准元数据
    由本管线统一组装。统计量、掩膜等专用参数
    由调用方通过 functools.partial 绑定。输出目录名仍须符合目标契约。
    已知检查失败记录到报告；process 及意外异常继续向外抛出。
    read(source_path, description, *, spec) 可替换默认读取，用于需要先处理
    分区再合并的源；须返回可关闭的 Dataset，资源由本管线管理。
    build_overrides 仅允许本次时间范围、数据集标识、内容版本及修订号；
    产品通道、网格、步长、数值与附件规则仍由标准契约确定，不写回 YAML。
    auxiliaries 按契约文件名映射附件源路径，写出后复制、再进行 Quick 校验。
    正式写出须提供必需附件；dry-run 可省略，不预先校验附件内容。
    """
    # 路径准备：隔离输入与输出，保护已有数据。
    source = Path(source_path).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Source and destination paths must not overlap")
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)

    report = BuildReport(str(target), dry_run)

    # 契约和源检查：只捕获这些阶段已知的输入错误。
    try:
        description = load_source_contract(source_config)
    except (ValueError, OSError, YAMLError) as exc:
        report.fail("source_contract", str(exc), exc)
        return report
    report.stages["source_contract"] = "passed"

    try:
        validate_source(source, description)
    except (ValueError, OSError) as exc:
        report.fail("source_check", str(exc), exc)
        return report
    report.stages["source_check"] = "passed"

    try:
        template = load_target_contract(target_spec)
        spec = resolve_build_contract(template, target, build_overrides)
    except (ValueError, OSError, YAMLError) as exc:
        report.fail("target_contract", str(exc), exc)
        return report
    report.stages["target_contract"] = "passed"
    report.coverage["target_contract"] = {
        "template_path": str(template.schema_path) if template.schema_path else None,
        "template_sha256": template.schema_sha256,
        "overrides": dict(build_overrides or {}),
        "effective": {"dataset_id": spec.dataset_id, "directory_name": spec.directory_name,
                      "first_time": spec.first_time, "last_time": spec.last_time,
                      "time_count": len(spec.expected_times()), "content_version": spec.content_version,
                      "data_revision": spec.data_revision},
    }
    try:
        assets = prepare_auxiliaries(spec, auxiliaries, target, require_all=not dry_run)
    except (ValueError, OSError) as exc:
        report.fail("auxiliary_inputs", str(exc), exc)
        return report
    report.stages["auxiliary_inputs"] = "passed"
    report.coverage["auxiliary_inputs"] = {
        asset.filename: {"source": str(asset.source), "sha256": asset.sha256} for asset in assets
    }
    report.coverage["missing_auxiliary_inputs"] = [
        item.filename for item in spec.variant.auxiliaries
        if item.required and item.filename not in (auxiliaries or {})
    ]

    # 惰性源必须保持打开直到写出完成；process 的异常不在捕获范围内。
    with ExitStack() as resources:
        try:
            raw = resources.enter_context(
                read_source(source, description) if read is None
                else read(source, description, spec=spec)
            )
        except (ValueError, OSError) as exc:
            report.fail("read", str(exc), exc)
            return report
        report.stages["read"] = "passed"
        product = process(raw, description, spec)
        if product is not raw and callable(getattr(product, "close", None)):
            resources.callback(product.close)
        report.stages["process"] = "passed"

        # 只做标准结构与元数据组装，不转换或修复处理函数返回的数据。
        try:
            product = assemble_standard_dataset(product, spec)
        except (ValueError, TypeError, KeyError) as exc:
            report.fail("assemble", str(exc), exc)
            return report
        if product is not raw:
            resources.callback(product.close)
        report.stages["assemble"] = "passed"

        # 底层将内存检查与写出合并；只调用一次，避免重复扫描。
        try:
            creation = create_zarr(product, target, spec, dry_run=dry_run)
        except (ValueError, OSError) as exc:
            report.fail("create", str(exc), exc)
            return report
        report.stages["create"] = "completed"
        report.creation_report = creation
        report.coverage.update(creation.coverage)
        if not creation.coverage["conformance_passed"]:
            report.fail_checks("input_check", creation)
            return report
        report.stages["input_check"] = "passed"

        if any(check.status not in ("pass", "warning", "not-applicable")
               for check in getattr(creation, "checks", ())):
            report.fail_checks("create" if dry_run else "write", creation)
            return report

        if dry_run:
            report.stages["write"] = "skipped"
            report.stages["auxiliary_copy"] = "skipped"
            report.stages["validate"] = "skipped"
            return report

        if not creation.coverage["written"]:
            report.fail_checks("write", creation)
            return report
        report.stages["write"] = "passed"

    # Closing files does not drop these local references to arrays/Dask graphs.
    # Release them, including cycles, before the read-back memory-budget check.
    del raw, product
    gc.collect()

    try:
        copy_auxiliaries(assets, target)
    except (ValueError, OSError) as exc:
        report.fail("auxiliary_copy", str(exc), exc)
        return report
    report.stages["auxiliary_copy"] = "passed"

    # 源已关闭，此阶段只校验落盘产物，失败时保留目录。
    try:
        validation = validate_zarr(target, spec, mode="quick")
    except (ValueError, OSError) as exc:
        report.fail("validate", str(exc), exc)
        return report
    report.validation_report = validation
    if not validation.quick_passed:
        report.fail_checks("validate", validation)
        return report
    report.stages["validate"] = "passed"
    return report
