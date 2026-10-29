"""Evaluation adapters; raw dataset declarations are loaded from config/data_sources YAML."""

from .climatology import ClimatologyCatalog, ClimatologyReader
from .gridded import GriddedCatalog, GriddedReader
from .station import DiamondStationCatalog, DiamondStationReader, load_station_whitelist

__all__ = [
    "ClimatologyCatalog",
    "ClimatologyReader",
    "GriddedCatalog",
    "GriddedReader",
    "DiamondStationCatalog",
    "DiamondStationReader",
    "load_station_whitelist",
]
