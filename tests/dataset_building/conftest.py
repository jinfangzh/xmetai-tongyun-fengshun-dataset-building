"""Shared small samples; every test gets its own writable temporary inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from .samples.legacy import profile, write_auxiliaries, write_legacy_store
from .samples.standard import write_published, write_published_schema

EXAMPLE_PARENT = Path(__file__).resolve().parents[2] / "examples" / "dataset_building"
if str(EXAMPLE_PARENT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_PARENT))


@pytest.fixture
def published(tmp_path):
    return write_published(tmp_path)


@pytest.fixture
def published_schema(published):
    return write_published_schema(published)


@pytest.fixture(params=["reordered", "aligned"])
def legacy_sample(tmp_path, request):
    source = tmp_path / "legacy.zarr"
    write_legacy_store(source, aligned=request.param == "aligned")
    auxiliary = tmp_path / "auxiliary"
    write_auxiliaries(auxiliary)
    expected = json.loads(
        (Path(__file__).parent / "samples" / "legacy_expected.json").read_text(encoding="utf-8")
    )
    return profile(source), auxiliary, expected


@pytest.fixture
def validation_samples(published_schema, tmp_path):
    later = write_published_schema(
        write_published(
            tmp_path / "later",
            dataset_id="later",
            channel_names=("q1000", "t2m"),
            time_shift_days=10,
        )
    )
    return [published_schema, later]
