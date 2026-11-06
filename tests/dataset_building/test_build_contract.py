"""A standard's product rules stay fixed while task coverage and identity vary."""

from pathlib import Path

import pytest

from datakit.schema import load_schema
from pipelines.dataset.build import resolve_build_contract

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("filename,start,end,count", [
    ("era5_c78_p25_h6.yaml", "2025-01-01T06", "2025-01-31T18", 123),
    ("oras5_c88_p25_h24.yaml", "2025-01-01T00", "2025-01-31T00", 31),
])
def test_build_contract_reuses_product_without_mutating_standard(tmp_path, filename, start, end, count):
    path = ROOT / "config/dataset_specs" / filename
    before = path.read_bytes()
    base, _ = load_schema(path)
    resolved = resolve_build_contract(base, tmp_path / "new-product", {
        "first_time": start, "last_time": end, "dataset_id": "new-product",
        "content_version": "v2", "data_revision": "20261104",
    })
    assert resolved.product is base.product
    assert resolved.arrays is base.arrays
    assert resolved.schema_sha256 == base.schema_sha256
    assert resolved.directory_name == resolved.dataset_id == "new-product"
    assert resolved.content_version == "v2"
    assert resolved.identity_mapping_evidence is None
    assert len(resolved.expected_times()) == count
    assert base.first_time != resolved.first_time
    assert path.read_bytes() == before
    assert resolve_build_contract(base, tmp_path / "unused") is base


@pytest.mark.parametrize("overrides", [
    {"first_time": "2025-01-01T00"},
    {"first_time": "2025-01-01T00", "last_time": "2025-01-31T00"},
    {"first_time": "2025-01-02T00", "last_time": "2025-01-01T00", "dataset_id": "test"},
    {"first_time": "2025-01-01T06", "last_time": "2025-01-02T06", "dataset_id": "test"},
    {"dataset_id": ""}, {"product": "different"}, {"schema_status": "approved"},
])
def test_build_contract_rejects_incomplete_coverage_and_product_overrides(tmp_path, overrides):
    base, _ = load_schema(ROOT / "config/dataset_specs/oras5_c88_p25_h24.yaml")
    with pytest.raises(ValueError):
        resolve_build_contract(base, tmp_path / "test", overrides)
