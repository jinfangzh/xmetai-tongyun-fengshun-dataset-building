"""Evaluation metadata rules for climatology reference fields."""

from datakit.evaluation.adapters.rules import AdaptationRules
from datakit.schema.semantics import DataKind

DEFAULT_UNITS = {
    "z500": "m",
    "gh": "m",
    "t2m": "K",
    "msl": "Pa",
    "u10": "m/s",
    "v10": "m/s",
}


def climatology_rules(units=None) -> AdaptationRules:
    selected = DEFAULT_UNITS if units is None else units
    return AdaptationRules(
        kind=DataKind.REFERENCE,
        grid_type="regular_latlon",
        variable_names={name: name for name in selected},
        coordinate_names={"latitude": "lat", "longitude": "lon"},
        units=selected,
    )
