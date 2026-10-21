"""XMetAI's framework-independent meteorological data toolkit."""

from . import (
    io,
    datasets,
    schema,
    transforms,
    statistics,
    validation,
    radar,
    visualization,
)

__all__ = [
    "io",
    "datasets",
    "schema",
    "transforms",
    "statistics",
    "validation",
    "radar",
    "visualization",
]

__version__ = "0.1.0"
