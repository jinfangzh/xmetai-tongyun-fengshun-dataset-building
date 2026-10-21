"""Resolve external fixture locations and explain missing data at test setup."""

import os
from pathlib import Path

import pytest


def fixture_directory(name: str) -> Path:
    """Locate a fixture group beneath XMETAI_FIXTURE_DIR."""
    return Path(os.environ.get("XMETAI_FIXTURE_DIR") or "/workspace/data/shenzw") / name


def require_path(value: str | Path | None, label: str) -> Path:
    """Skip the requesting test if its external input is absent."""
    if not value:
        pytest.skip(f"未配置资料路径 {label}")
    path = Path(value)
    if not path.exists():
        pytest.skip(f"{label}={path} 不存在")
    return path
