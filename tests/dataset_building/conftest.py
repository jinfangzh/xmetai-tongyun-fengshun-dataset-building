"""Shared small samples; every test gets its own writable temporary inputs."""

from __future__ import annotations

import pytest

from .samples.standard import write_published, write_published_schema


@pytest.fixture
def published(tmp_path):
    return write_published(tmp_path)


@pytest.fixture
def published_schema(published):
    return write_published_schema(published)


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
