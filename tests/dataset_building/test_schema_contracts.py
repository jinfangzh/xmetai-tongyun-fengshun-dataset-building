"""Schema declarations retain metadata and reject ambiguous/unsafe mappings."""

import numpy as np
import pytest
import xarray as xr
import yaml

from datakit.schema import build_standard_channel_info, load_schema
from datakit.schema.mapping import DatasetMapping, VarSpec, apply_dataset_mapping, load_mapping
from datakit.schema.specifications import GridDefinition


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "2"])
@pytest.mark.parametrize("field", ["scale", "offset"])
def test_variable_mapping_rejects_invalid_coefficients(value, field):
    with pytest.raises(ValueError, match="finite numbers"):
        VarSpec(**{field: value})


@pytest.mark.parametrize("declaration", [
    {"packed_variable": "data"},
    {"packed_dimension": "channel"},
    {"variables": {"a": VarSpec(source="x"), "b": VarSpec(source="x")}},
    {"variables": {"a": VarSpec(source="X"), "b": VarSpec(source="x")}, "case_insensitive": True},
])
def test_mapping_object_and_loader_enforce_same_constraints(declaration):
    for factory in (DatasetMapping, lambda **values: load_mapping(values)):
        with pytest.raises(ValueError):
            factory(**declaration)


def test_single_variable_requires_unambiguous_target():
    dataset = xr.Dataset({"unknown": ("row", [1.0])})
    mapping = DatasetMapping(variables={"a": VarSpec(), "b": VarSpec()})
    with pytest.raises(ValueError, match="multiple unnamed"):
        apply_dataset_mapping(dataset, mapping, ["a", "b"])


def test_unit_mapping_preserves_metadata_and_input():
    dataset = xr.Dataset({"raw": ("row", [1.0], {"units": "m", "long_name": "rainfall"})})
    before = dataset.copy(deep=True)
    mapped = apply_dataset_mapping(
        dataset,
        DatasetMapping(variables={"tp": VarSpec(source="raw", scale=1000, unit="mm", source_unit="m")}),
        ["tp"],
    )
    np.testing.assert_array_equal(mapped.tp.values, [1000.0])
    assert mapped.tp.attrs == {"units": "mm", "original_units": "m", "long_name": "rainfall"}
    xr.testing.assert_identical(dataset, before)


def test_schema_keeps_declared_description_for_metadata_generation(published_schema):
    _, _, path, document = published_schema
    document["variable_metadata"]["t2m"]["long_name"] = "Temperature at two metres"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    schema, _ = load_schema(path)
    assert schema.product.channels[0].long_name == "Temperature at two metres"
    mean = np.zeros(len(schema.product.channels), dtype="f4")
    std = np.ones_like(mean)
    metadata = build_standard_channel_info(
        schema.product, mean=mean, std=std, channel_metadata=schema.channel_metadata,
    )
    assert metadata["t2m"]["long_name"] == "Temperature at two metres"


@pytest.mark.parametrize("count", [True, 3.0, 3.5])
def test_grid_requires_integer_sizes(count):
    with pytest.raises(ValueError):
        GridDefinition("invalid", count, 4, 1.5, 90.0, 0.0, "north-to-south")
