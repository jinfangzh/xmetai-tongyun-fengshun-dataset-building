"""Shared synthetic factories and external-data prerequisites for integration tests."""

from types import SimpleNamespace

import pytest

from tests.evaluation.helpers import weather_field
from tests.evaluation.helpers.paths import fixture_directory, require_path


@pytest.fixture
def weather_field_data(weather_inputs):
    return SimpleNamespace(
        write_synthetic_fuxi_phys_multi=weather_field.write_synthetic_fuxi_phys_multi,
        write_synthetic_era5_zarr_multi=weather_field.write_synthetic_era5_zarr_multi,
        write_synthetic_daily_climatology=weather_field.write_synthetic_daily_climatology,
    )


@pytest.fixture(scope="session")
def fdp_field_data(netcdf_inputs):
    """Check Fengqing, CRA40, and climatology inputs after test collection."""
    pytest.importorskip("cfgrib")
    root = fixture_directory("fdp_test_data")
    for name in ("fengqing", "cra_root", "cli_fake"):
        require_path(root / name, f"FDP {name}")
    return root


@pytest.fixture(scope="session")
def fdp_precip_data(netcdf_inputs):
    """Check Fengqing and station rainfall inputs after test collection."""
    root = fixture_directory("fdp_test_data")
    for name in ("fengqing", "rain"):
        require_path(root / name, f"FDP {name}")
    return root


@pytest.fixture(scope="session")
def netcdf_inputs():
    """Require the NetCDF backend when running integration tests."""
    pytest.importorskip("netCDF4")


@pytest.fixture(scope="session")
def weather_inputs(netcdf_inputs):
    """Require the Zarr backend for weather-field integration tests."""
    pytest.importorskip("zarr")
