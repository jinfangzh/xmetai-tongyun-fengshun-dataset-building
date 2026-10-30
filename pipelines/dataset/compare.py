"""Read and compare complete datasets with optional source-name mapping."""

from collections.abc import Mapping
from pathlib import Path

from datakit.validation.comparison import compare_datasets, validate_comparison_options

from ._sources import _source


def _mapped(dataset, mapping):
    if isinstance(mapping, (str, Path)):
        import yaml

        mapping = yaml.safe_load(Path(mapping).read_text(encoding="utf-8"))
    if mapping is None:
        return dataset
    if not isinstance(mapping, Mapping) or set(mapping) - {
        "dimensions",
        "variables",
        "channels",
    }:
        raise ValueError(
            "mapping accepts only dimensions, variables and channels (source -> target)"
        )
    for section in ("dimensions", "variables", "channels"):
        if section in mapping and not isinstance(mapping[section], Mapping):
            raise TypeError(f"mapping.{section} must be a mapping")
    dataset = dataset.rename(dict(mapping.get("dimensions", {})))
    dataset = dataset.rename(dict(mapping.get("variables", {})))
    if mapping.get("channels"):
        if "channel" not in dataset.coords:
            raise ValueError(
                "Channel aliases require a channel coordinate after dimension mapping"
            )
        aliases = mapping["channels"]
        labels = [aliases.get(str(v), str(v)) for v in dataset.channel.values]
        if len(set(labels)) != len(labels):
            raise ValueError("Channel aliases produce duplicate labels")
        dataset = dataset.assign_coords(channel=labels)
    return dataset


def compare_zarr(
    source, target, *, mapping=None, atol=0.0, rtol=0.0, memory_budget_mib=64
):
    """Align full label sets and compare values; mappings rename source labels only."""
    validate_comparison_options(atol, rtol, memory_budget_mib)
    with _source(source) as left, _source(target) as right:
        return compare_datasets(
            _mapped(left, mapping),
            right,
            atol=atol,
            rtol=rtol,
            memory_budget_mib=memory_budget_mib,
        )
