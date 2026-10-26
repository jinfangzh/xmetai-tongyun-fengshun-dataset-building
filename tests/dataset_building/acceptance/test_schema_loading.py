"""Strict configuration errors and YAML-driven, read-only end-to-end checks."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import yaml
import zarr

from datakit.schema.datasets.inputs import verify_schema_unchanged
from pipelines.dataset_building import validate_standard_dataset
from pipelines.dataset_building.support.inputs import load_validation_schema

from ._helpers import SCHEMA_DIRECTORY, _checks


def _write(path, document):
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return load_validation_schema(path)[0]


@pytest.mark.parametrize(
    "keys,value,reason",
    [
        (("unexpected",), {}, "unknown fields"),
        (("meta_info", "typo"), 1, "unknown fields"),
        (("meta_info", "grid", "latitude", "count"), True, "integer"),
        (("meta_info", "grid", "space_resolution"), float("nan"), "finite"),
        (("meta_info", "grid", "latitude", "last"), 86.0, "disagree"),
        (("meta_info", "grid", "longitude", "direction"), "east-to-west", "disagree"),
        (("meta_info", "time_coverage", "time_count"), 13, "time_count"),
        (("meta_info", "time_coverage", "timezone"), "local", "unsupported value"),
        (("meta_info", "time_coverage", "time_resolution", "value"), True, "integer"),
        (("meta_info", "channels"), ["t2m", "t2m", "tp"], "unique"),
        (("meta_info", "channels"), ["t2m", "q850", "tp"], "undefined channel"),
        (("meta_info", "arrays", "data", "dtype"), "float64", "Unsupported.*layout"),
        (("meta_info", "arrays", "data", "chunks", "time"), True, "integer"),
        (("meta_info", "arrays", "data", "chunks", "channel"), 1, "complete.*frame"),
        (("meta_info", "mask", "valid_value"), True, "integer"),
        (("meta_info", "normalization", "statistics", "std_positive"), False, "unsupported"),
        (("meta_info", "normalization", "effective_statistics", "unused"),
         {"mean": 0, "std": 1}, "unknown variables"),
        (("variable_metadata", "q", "units"), None, "nonempty string"),
        (("variable_metadata", "q", "preprocessing", 0, "factor"), True, "finite"),
        (("variable_metadata", "q", "preprocessing", 0, "factor"), float("inf"), "finite"),
        (("variable_metadata", "q", "preprocessing", 0, "operation"), "eval", "unsupported"),
        (("variable_metadata", "q", "preprocessing", 0, "unknown"), 1, "unknown fields"),
        (("meta_info", "storage_format", "allow_nested_groups"), True, "unsupported"),
        (("meta_info", "schema_version"), 2.0, "nonempty string"),
    ],
)
def test_invalid_standards_fail_before_data_io(published_schema, keys, value, reason):
    _, _, schema, original = published_schema
    document = deepcopy(original)
    target = document
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(ValueError, match=reason) as error:
        _write(schema, document)
    assert str(schema) in str(error.value)


@pytest.mark.parametrize(
    "text,reason",
    [
        ("meta_info: {name: one, name: two}\nvariable_metadata: {}", "Duplicate YAML key"),
        ("meta_info: &common {}\nvariable_metadata: *common", "aliases"),
        ("!!python/object/apply:os.system ['must-not-execute']", "constructor"),
    ],
)
def test_unsafe_or_ambiguous_yaml_is_rejected(tmp_path, text, reason):
    schema = tmp_path / "invalid.yaml"
    schema.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=reason):
        load_validation_schema(schema)


def test_missing_fields_and_unused_variable_declarations_are_rejected(published_schema):
    _, _, schema, original = published_schema
    document = deepcopy(original)
    del document["variable_metadata"]["q"]["source_name"]
    with pytest.raises(ValueError, match="missing fields"):
        _write(schema, document)
    document = deepcopy(original)
    document["variable_metadata"]["unused"] = deepcopy(document["variable_metadata"]["t2m"])
    with pytest.raises(ValueError, match="unused declarations"):
        _write(schema, document)


def test_yaml_drives_preprocessing_and_channel_metadata_checks(published_schema):
    path, _, schema, document = published_schema
    document["variable_metadata"]["q"]["preprocessing"][0]["factor"] = 100.0
    document["meta_info"]["metadata"]["channel_metadata"]["version"] = "1.1"
    spec = _write(schema, document)
    report = validate_standard_dataset(path, spec)
    assert _checks(report)["semantics.q1000"] == "fail"
    assert _checks(report)["semantics.info_version"] == "fail"
    assert not report.quick_passed


def test_radiation_window_is_not_reconstructed_from_product_name(tmp_path):
    document = yaml.safe_load((SCHEMA_DIRECTORY / "era5_c78_p25_h6.yaml").read_text(encoding="utf-8"))
    step = document["variable_metadata"]["ssr"]["preprocessing"][0]
    step.update(divisor=3600.0, accumulation_window_seconds=3600)
    spec = _write(tmp_path / "radiation.yaml", document)
    channel = next(channel for channel in spec.product.channels if channel.name == "ssr")
    assert channel.semantic_fields()["preprocessing"][0]["divisor"] == 3600.0
    step["accumulation_window_seconds"] = 21600
    with pytest.raises(ValueError, match="window must match"):
        _write(tmp_path / "radiation.yaml", document)


def test_declared_channel_metadata_location_is_used(published_schema):
    path, _, schema, document = published_schema
    metadata = document["meta_info"]["metadata"]
    metadata["channel_metadata"]["attribute"] = "declared_info"
    metadata["data_attributes"]["scale_factor_location"] = "channel.declared_info[*].scale_factor"
    metadata["data_attributes"]["add_offset_location"] = "channel.declared_info[*].add_offset"
    report = validate_standard_dataset(path, _write(schema, document))
    assert _checks(report)["semantics.channel_info"] == "fail"
    assert not report.quick_passed


def test_declared_main_dtype_drives_layout_and_raw_scan(published_schema):
    path, original, schema, document = published_schema
    document["meta_info"]["arrays"]["data"]["dtype"] = "float32"
    spec = replace(
        _write(schema, document), snapshot_id=original.snapshot_id,
        production_evidence=original.production_evidence,
    )
    assert _checks(validate_standard_dataset(path, spec))["layout.data"] == "fail"
    group = zarr.open_group(str(path), mode="a", use_consolidated=False)
    old = group["data"]
    values, attrs, chunks = old[:].astype("f4"), dict(old.attrs), old.chunks
    del group["data"]
    group.create_array(
        "data", data=values, chunks=chunks, attributes=attrs,
        dimension_names=("time", "channel", "lat", "lon"),
        config={"write_empty_chunks": True},
    )
    zarr.consolidate_metadata(str(path))
    report = validate_standard_dataset(path, spec, mode="full")
    assert report.release_passed
    assert report.coverage["value_checked"] == 12
    assert report.coverage["raw_block_bytes"] == 1 * 3 * 3 * 4 * 4


def test_yaml_full_and_quick_preserve_completion_and_production_gates(published_schema):
    path, spec, schema, document = published_schema
    full = validate_standard_dataset(path, spec, mode="full")
    assert full.release_passed and full.coverage["scan_complete"]
    quick = validate_standard_dataset(path, spec)
    assert quick.quick_passed and not quick.release_passed
    assert quick.coverage["selection"] == [0, 5, 11]
    limited = validate_standard_dataset(path, spec, mode="full", max_blocks=1)
    assert not limited.release_passed and limited.coverage["interrupted"]
    missing = validate_standard_dataset(path, _write(schema, document), mode="full")
    assert _checks(missing)["publication.frozen_snapshot"] == "needs-review"
    assert not missing.release_passed


def test_loaded_schema_hash_and_approval_are_enforced(published_schema):
    path, spec, schema, document = published_schema
    assert spec.schema_sha256 == hashlib.sha256(schema.read_bytes()).hexdigest()
    schema.write_text(schema.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        verify_schema_unchanged(spec)
    report = validate_standard_dataset(path, spec)
    assert _checks(report)["publication.schema_unchanged"] == "fail"
    assert not report.quick_passed
    document["meta_info"]["status"] = "draft"
    report = validate_standard_dataset(path, _write(schema, document))
    assert _checks(report)["publication.schema_approval"] == "needs-review"
    assert not report.quick_passed


def test_npy_without_grid_evidence_is_not_silently_accepted(published_schema):
    path, _, schema, document = published_schema
    document["meta_info"]["auxiliaries"]["files"] = {
        "weight.npy": {
            "format": "npy", "role": "weight", "required": True,
            "dimensions": ["singleton", "lat", "singleton"], "dtype": "float32",
            "grid": "main_data", "finite": True, "nonnegative_atol": 1e-7,
            "require_positive_value": True,
        },
    }
    np.save(path / "weight.npy", np.ones((1, 3, 1), dtype="f4"))
    report = validate_standard_dataset(path, _write(schema, document), mode="full")
    assert _checks(report)["auxiliary.weight.npy.grid_association"] == "needs-review"
    assert not report.release_passed


def test_batch_cli_reads_standalone_schemas_without_inventory(published_schema, tmp_path):
    from examples.dataset_building.validate_standard_dataset import main

    path, _, schema, document = published_schema
    other = tmp_path / "other.yaml"
    document["meta_info"]["dataset_id"] = "fixture.alternate"
    _write(other, document)
    output = tmp_path / "reports"
    assert main(["--schema-directory", str(tmp_path), "--output", str(output)]) == 0
    batch = json.loads((output / "batch.json").read_text())
    assert batch["requested_schemas"] == batch["completed_schemas"] == 2
    assert len(list(output.glob("*-blocks.jsonl"))) == 2
    forbidden = tmp_path / "invalid-output"
    assert main([
        "--schema", str(schema), str(other), "--dataset-path", str(path),
        "--output", str(forbidden),
    ]) == 3
    assert not forbidden.exists()
