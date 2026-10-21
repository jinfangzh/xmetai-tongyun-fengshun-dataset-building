"""Self-contained example for refreshing legacy meteorological Zarr stores.

This package intentionally lives under ``examples`` rather than ``datakit``.
It is dataset-building workflow code under review, not part of the supported
public library API.
"""

from .auxiliary_files import migrate_auxiliary_files
from .config import RefreshProfile, load_profiles, select_profiles
from .pipeline import build_refresh_plan, refresh_legacy_zarr
from .source import LegacyZarrSource
from .validation import read_manifest, validate_refreshed_store

__all__ = [
    "LegacyZarrSource",
    "RefreshProfile",
    "build_refresh_plan",
    "load_profiles",
    "migrate_auxiliary_files",
    "read_manifest",
    "refresh_legacy_zarr",
    "select_profiles",
    "validate_refreshed_store",
]