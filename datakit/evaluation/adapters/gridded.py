"""Evaluation metadata rules for decoded gridded sources."""

from datakit.evaluation.adapters.rules import AdaptationRules
from datakit.io.readers.layouts import GriddedLayout


def rules_for_layout(layout: GriddedLayout) -> AdaptationRules:
    return AdaptationRules(
        kind=layout.kind,
        grid_type="regular_latlon",
        variable_names={name: name for name in layout.variables},
        units={name: spec.unit for name, spec in layout.variables.items()},
        prefer_dataset_units=True,
        temporal_kind={
            name: spec.temporal_kind for name, spec in layout.variables.items()
        },
        temporal_per_variable=True,
        member_dim=layout.member_dim,
    )
