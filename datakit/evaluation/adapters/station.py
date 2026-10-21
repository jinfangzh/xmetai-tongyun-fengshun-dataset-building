"""Evaluation metadata rules for Diamond station observations."""

from datakit.evaluation.adapters.rules import AdaptationRules
from datakit.schema.semantics import DataKind, TemporalKind

DIAMOND_STATION_RULES = AdaptationRules(
    kind=DataKind.STATION_OBSERVATION,
    grid_type="stations",
    variable_names={"precipitation": "tp"},
    units={"precipitation": "mm"},
    default_temporal_kind=TemporalKind.INTERVAL_ACCUMULATION,
)
