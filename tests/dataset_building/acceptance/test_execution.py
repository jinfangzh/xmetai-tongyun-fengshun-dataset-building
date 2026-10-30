"""Real file bindings keep target YAML immutable when execution receipts change."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import yaml
import zarr

from datakit.schema import DatasetSchema, load_schema
from pipelines.dataset import validate_zarr
from pipelines.dataset.configuration import (
    load_schema_reference,
    load_validation_run,
    load_validation_schema,
    read_frozen_json,
)
from pipelines.dataset.configuration import DatasetValidationSpec
from pipelines.dataset.old.conversion_validation import run_conversion_validation

from ._helpers import _checks, _receipt, validate_fixture


def _reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _run_file(tmp_path, schema, spec):
    document = {
        "schema": schema.name,
        "execution": {
            "snapshot_id": spec.snapshot_id,
            "production_evidence": _reference(spec.production_evidence.path),
        },
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def _validate_run(run, output, **options):
    spec, locations = load_validation_run(run)
    return validate_zarr(
        locations["dataset_path"],
        spec,
        output=output,
        auxiliary_directory=locations["auxiliary_directory"],
        **options,
    )


@pytest.mark.parametrize("mode", ["quick", "full"])
def test_caller_can_validate_independent_datasets_with_disjoint_times(
    published_schema, tmp_path, mode
):
    path, spec, schema, document = published_schema
    first_run, _ = _run_file(tmp_path, schema, spec)
    other_root = tmp_path / "later"
    other = other_root / "data.zarr"
    shutil.copytree(path, other)
    group = zarr.open_group(str(other), mode="r+")
    times = np.asarray(group["time"][:], dtype="datetime64[ns]") + np.timedelta64(10, "D")
    group["time"][:] = times
    zarr.consolidate_metadata(str(other))
    later_document = deepcopy(document)
    meta = later_document["meta_info"]
    meta.update(dataset_path=str(other), auxiliary_directory=str(other))
    meta["time_coverage"].update(first=str(times[0]), last=str(times[-1]))
    later_schema = other_root / "later.yaml"
    later_schema.write_text(yaml.safe_dump(later_document, sort_keys=False), encoding="utf-8")
    later_spec = _receipt(other, load_validation_schema(later_schema)[0])
    later_run, _ = _run_file(other_root, later_schema, later_spec)
    output = tmp_path / "independent-reports"
    reports = [
        _validate_run(run, output / f"{index}.json", mode=mode).to_dict()
        for index, run in enumerate([first_run, later_run])
    ]
    assert len(reports) == 2
    assert not np.intersect1d(spec.expected_times(), later_spec.expected_times()).size
    for report in reports:
        assert report["quick_passed"] is (mode == "quick")
        assert report["release_passed"] is (mode == "full")
        assert "cross_dataset_compatibility" not in report
        assert all(check["check_id"] != "combination.scope" for check in report["checks"])
    saved = list(output.glob("*.json"))
    assert len(saved) == 2
    assert not list(output.glob("*.jsonl"))


@pytest.mark.parametrize(
    "field",
    ["snapshot_id", "production_evidence", "conversion_evidence"],
)
def test_runtime_receipts_cannot_be_embedded_in_target_schema(
    published_schema,
    field,
):
    _, _, schema, document = published_schema
    document["meta_info"][field] = "runtime-value"
    schema.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_schema(schema)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        (("arrays", "data"), "dtype"),
        (("arrays", "data"), "chunks"),
        (("mask",), "valid_value"),
        (("metadata", "channel_metadata"), "version"),
        (("metadata", "data_attributes"), "inverse_transform"),
    ],
)
def test_missing_standard_fields_are_not_filled_from_python(
    published_schema,
    section,
    field,
):
    _, _, schema, document = published_schema
    target = document["meta_info"]
    for name in section:
        target = target[name]
    del target[field]
    schema.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields|complete spatial/channel frame"):
        load_schema(schema)


@pytest.mark.parametrize("declaration", ["data_dtype", "mask_values", "channel_metadata"])
def test_explicit_yaml_declarations_control_actual_validation(published_schema, declaration):
    path, _, schema, document = published_schema
    meta = document["meta_info"]
    if declaration == "data_dtype":
        meta["arrays"]["data"]["dtype"] = "float32"
    elif declaration == "mask_values":
        meta["mask"].update(allowed_values=[2, 5], valid_value=5, complement_sum=7)
    else:
        meta["metadata"]["channel_metadata"].update(
            attribute="per_channel",
            version_attribute="per_channel_version",
            version="2.7",
        )
        attributes = meta["metadata"]["data_attributes"]
        attributes["scale_factor_location"] = "channel.per_channel[*].scale_factor"
        attributes["add_offset_location"] = "channel.per_channel[*].add_offset"
    schema.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    spec, _ = load_schema(schema)
    assert not validate_fixture(path, spec, mode="quick").to_dict()["quick_passed"]

    group = zarr.open_group(str(path), mode="r+")
    if declaration == "data_dtype":
        values = group["data"][:].astype("float32")
        attrs = dict(group["data"].attrs)
        del group["data"]
        group.create_array(
            "data",
            data=values,
            chunks=(1, 3, 3, 4),
            dimension_names=("time", "channel", "lat", "lon"),
            attributes=attrs,
            config={"write_empty_chunks": True},
        )
    elif declaration == "mask_values":
        group["mask"][:] = group["mask"][:] * 3 + 2
    else:
        attrs = group["channel"].attrs
        attrs["per_channel"] = attrs.pop("channel_info")
        attrs["per_channel_version"] = "2.7"
        del attrs["channel_info_schema_version"]
        group["data"].attrs.update(meta["metadata"]["data_attributes"])
    zarr.consolidate_metadata(str(path))
    report = validate_fixture(path, spec, mode="quick")
    assert report.to_dict()["quick_passed"], report.to_dict()


@pytest.mark.parametrize("fault", ["dtype", "chunks", "formula"])
def test_yaml_declarations_still_respect_execution_support_limits(published_schema, fault):
    _, _, schema, document = published_schema
    meta = document["meta_info"]
    if fault == "dtype":
        meta["arrays"]["data"]["dtype"] = "float64"
    elif fault == "chunks":
        meta["arrays"]["data"]["chunks"]["time"] = 2
    else:
        meta["metadata"]["channel_metadata"]["scale_offset_formula"] = "arbitrary_formula"
    schema.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported|complete spatial/channel frame|expected"):
        load_schema(schema)


def test_expected_chunks_are_resolved_from_the_explicit_declaration(published_schema):
    _, spec, _, _ = published_schema
    arrays = {
        **spec.arrays,
        "data": replace(spec.arrays["data"], chunks=(2, "full", 1, "full")),
    }
    declared = replace(spec, arrays=arrays)
    assert declared.expected_chunks == (2, 3, 1, 4)
    with pytest.raises(ValueError, match="complete spatial/channel frame"):
        declared.validate_configuration()


def test_real_conversion_receipt_attaches_without_changing_yaml(published_schema, tmp_path):
    path, spec, schema, _ = published_schema
    frozen_yaml = schema.read_bytes()
    source = tmp_path / "independent-source.zarr"
    shutil.copytree(path, source)
    conversion_config = tmp_path / "conversion.json"
    conversion_config.write_text(
        json.dumps(
            {
                "target_schema": schema.name,
                "source_path": source.name,
                "execution": {"snapshot_id": spec.snapshot_id},
                "conversion": {
                    "source_identity": "independent-source",
                    "source_snapshot_id": "independent-source-frozen",
                    "rules_version": "independent-identity-1",
                    "source_first_time": spec.first_time,
                    "source_last_time": spec.last_time,
                    "source_cadence_hours": 6,
                    "source_channels": list(spec.product.channel_names),
                    "rules": [
                        {"target_channel": name, "source_channel": name, "atol": 0, "rtol": 0}
                        for name in spec.product.channel_names
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "conversion-receipt.json"
    receipt = run_conversion_validation(conversion_config, receipt_path)
    assert receipt["comparison_complete"]
    assert receipt["schema_sha256"] == hashlib.sha256(frozen_yaml).hexdigest()
    run, document = _run_file(tmp_path, schema, spec)
    document["execution"].update(
        conversion_required=True,
        conversion_evidence=_reference(receipt_path),
    )
    run.write_text(json.dumps(document), encoding="utf-8")
    report = _validate_run(run, tmp_path / "report.json", mode="full").to_dict()
    assert report["release_passed"]
    assert schema.read_bytes() == frozen_yaml
    assert report["identity"]["schema_sha256"] == receipt["schema_sha256"]
    assert report["identity"]["execution_config_sha256"] == _reference(run)["sha256"]
    assert report["identity"]["execution_config_sha256"] != receipt["config_sha256"]
    assert report["identity"]["snapshot_id"] == spec.snapshot_id
    receipt["config_id"] = "different-target"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    document["execution"]["conversion_evidence"] = _reference(receipt_path)
    run.write_text(json.dumps(document), encoding="utf-8")
    mismatched, _ = load_validation_run(run)
    assert (
        _checks(validate_fixture(path, mismatched, mode="full"))["conversion_fidelity.identity"]
        == "fail"
    )
    assert schema.read_bytes() == frozen_yaml

    def change_conversion_config():
        with conversion_config.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        return False

    changed_output = tmp_path / "changed-conversion-receipt.json"
    with pytest.raises(ValueError, match="Execution configuration changed"):
        run_conversion_validation(
            conversion_config,
            changed_output,
            stop_requested=change_conversion_config,
        )
    assert not changed_output.exists()


def test_execution_changes_during_scan_cannot_pass(published_schema, tmp_path):
    path, spec, schema, _ = published_schema
    run, _ = _run_file(tmp_path, schema, spec)
    loaded, _ = load_validation_run(run)

    def change_run(_entry):
        with run.open("a", encoding="utf-8") as stream:
            stream.write("\n")

    report = validate_fixture(path, loaded, mode="full", on_block=change_run)
    assert _checks(report)["publication.execution_unchanged"] == "fail"
    assert not report.release_passed


def test_parsed_execution_bytes_must_match_the_frozen_digest(published_schema, tmp_path):
    _, spec, schema, _ = published_schema
    run, _ = _run_file(tmp_path, schema, spec)
    document, digest = read_frozen_json(run)
    with run.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Execution configuration changed"):
        load_schema_reference(run, document, config_sha256=digest)


@pytest.mark.parametrize(
    "execution",
    [
        {"typo": True},
        {"snapshot_id": False},
        {"conversion_required": "yes"},
        {"production_evidence": {"path": "receipt.json"}},
    ],
)
def test_invalid_execution_fields_fail_before_dataset_io(
    published_schema,
    tmp_path,
    execution,
):
    _, spec, schema, _ = published_schema
    run, document = _run_file(tmp_path, schema, spec)
    document["execution"] = execution
    run.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        load_validation_run(run)


def test_execution_cannot_relax_schema_evidence_requirements(
    published_schema,
    tmp_path,
):
    _, spec, schema, document = published_schema
    document["meta_info"]["conversion_required"] = True
    schema.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    run, configuration = _run_file(tmp_path, schema, spec)
    configuration["execution"]["conversion_required"] = False
    run.write_text(json.dumps(configuration), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot relax"):
        load_validation_run(run)


@pytest.mark.parametrize(
    "location",
    ["yaml", "execution", "run", "conversion"],
)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("consumer_required", False),
        ("consumer_required", True),
        ("consumer_evidence", {"path": "retired.json", "sha256": "0" * 64}),
        ("model_evidence", {"path": "retired.json", "sha256": "0" * 64}),
    ],
)
def test_retired_model_fields_are_rejected(
    published_schema,
    tmp_path,
    location,
    field,
    value,
):
    _, spec, schema, document = published_schema
    if location == "yaml":
        document["meta_info"][field] = value
        schema.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown fields"):
            load_schema(schema)
    elif location in ("execution", "run"):
        run, configuration = _run_file(tmp_path, schema, spec)
        target = configuration["execution"] if location == "execution" else configuration
        target[field] = value
        run.write_text(json.dumps(configuration), encoding="utf-8")
        message = "unknown fields" if location == "execution" else "only supported"
        with pytest.raises(ValueError, match=message):
            load_validation_run(run)
    else:
        configuration = {
            "target_schema": schema.name,
            "source_path": "not-opened.zarr",
            "conversion": {},
            field: value,
        }
        config = tmp_path / "specialist.json"
        config.write_text(json.dumps(configuration), encoding="utf-8")
        with pytest.raises(ValueError, match="no longer supported"):
            run_conversion_validation(config, tmp_path / "rejected.json")
        assert not (tmp_path / "rejected.json").exists()


def test_run_configuration_only_references_rules_and_resolves_its_own_paths(
    published_schema,
    tmp_path,
):
    path, spec, schema, _ = published_schema
    run, document = _run_file(tmp_path, schema, spec)
    document["dataset_path"] = path.name
    document["execution"]["production_evidence"]["path"] = spec.production_evidence.path.name
    run.write_text(json.dumps(document), encoding="utf-8")
    loaded, locations = load_validation_run(run)
    assert Path(locations["dataset_path"]) == path
    assert locations["auxiliary_directory"] == locations["dataset_path"]
    assert loaded.production_evidence.path == spec.production_evidence.path
    assert not any(
        name in asdict(load_schema(schema)[0])
        for name in ("snapshot_id", "production_evidence", "conversion_evidence")
    )
    document["channels"] = ["override"]
    run.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="only supported"):
        load_validation_run(run)


def test_pure_schema_can_be_used_by_the_workflow_without_attached_evidence(published_schema):
    path, _, schema, _ = published_schema
    pure, _ = load_schema(schema)
    report = validate_fixture(path, pure)
    assert report.quick_passed
    assert not report.release_passed
    assert report.identity["snapshot_id"] is None


def test_input_loaders_keep_standard_and_execution_models_separate(published_schema):
    _, _, schema, _ = published_schema
    standard, locations = load_schema(schema)
    execution, run_locations = load_validation_schema(schema)
    assert type(standard) is DatasetSchema
    assert type(execution) is DatasetValidationSpec
    assert not hasattr(standard, "snapshot_id")
    assert execution.snapshot_id is None
    assert execution == DatasetValidationSpec.from_schema(standard)
    assert run_locations == locations


@pytest.mark.parametrize("source_name", ["schema", "run", "production"])
def test_report_cannot_overwrite_configurations_or_evidence(
    published_schema,
    tmp_path,
    source_name,
):
    _, spec, schema, _ = published_schema
    run, document = _run_file(tmp_path, schema, spec)
    production = spec.production_evidence.path
    if source_name != "schema":
        isolated = tmp_path / f"{source_name}-input"
        isolated.mkdir()
        if source_name == "run":
            run = isolated / "run.json"
            document["schema"] = str(schema)
        else:
            production = isolated / "production.json"
            production.write_bytes(spec.production_evidence.path.read_bytes())
            document["execution"]["production_evidence"] = _reference(production)
        run.write_text(json.dumps(document), encoding="utf-8")
    sources = {"schema": schema, "run": run, "production": production}
    source = sources[source_name]
    before = source.read_bytes()
    with pytest.raises(ValueError, match="overwrite input"):
        _validate_run(run, source)
    assert source.read_bytes() == before
