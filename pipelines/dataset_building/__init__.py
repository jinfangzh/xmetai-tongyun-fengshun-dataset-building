"""Production definitions and lazily loaded dataset validation workflows."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .workflows.conversion import run_conversion_validation, validate_standard_conversion
    from .workflows.dataset import run_dataset_validation, validate_standard_dataset

__all__ = [
    "validate_standard_dataset",
    "validate_standard_conversion",
    "run_dataset_validation",
    "run_conversion_validation",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    workflow = "conversion" if "conversion" in name else "dataset"
    value = getattr(import_module(f".workflows.{workflow}", __name__), name)
    globals()[name] = value
    return value
