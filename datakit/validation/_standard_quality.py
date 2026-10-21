"""Bounded channel-at-a-time aggregation of unmodified standard data blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from datakit.schema.standard_dataset import DatasetValidationSpec, QualityRule
from .reports import ValidationReport


@dataclass
class _State:
    count: int = 0
    nan: int = 0
    inf: int = 0
    finite: int = 0
    minimum: float | None = None
    maximum: float | None = None
    constant_blocks: int = 0
    outside: int = 0
    inverse_nonfinite: int = 0
    invalid_processing_values: int = 0
    maximum_block_nan_fraction: float = 0.0
    excessive_missing_blocks: int = 0


class StandardQualityAccumulator:
    """Hold O(channels) state and O(one channel's pixels) temporary arrays.

    Constant-field policy is per block, not over the global aggregate. Bounds
    use the declared domain; Inf always fails. NaN limits apply across checked
    valid pixels and reports identify the checked scope separately.
    """

    def __init__(
        self,
        spec: DatasetValidationSpec,
        mean: np.ndarray,
        std: np.ndarray,
        mask: np.ndarray,
        valid_regions: dict[str, np.ndarray] | None = None,
    ):
        self.spec, self.mean, self.std, self.mask = spec, mean, std, mask
        self.valid_regions = valid_regions or {}
        self.rules = {rule.channel: rule for rule in spec.quality}
        self.states = {name: _State() for name in spec.product.channel_names}

    def update(self, block: np.ndarray) -> dict[str, Any]:
        anomalies = {}
        for index, definition in enumerate(self.spec.product.channels):
            name, state = definition.name, self.states[definition.name]
            rule = self.rules.get(name)
            raw = block[:, index, :, :]
            region = rule.valid_region if rule else "all"
            if region == "all":
                values = raw.reshape(-1)
            else:
                region_mask = (
                    self.valid_regions[region.removeprefix("custom:")]
                    if region.startswith("custom:")
                    else self.mask[self.spec.product.mask.region_index(region)]
                )
                selected = (
                    region_mask.astype(bool, copy=False)
                    if region.startswith("custom:")
                    else region_mask == self.spec.product.mask.valid_value
                )
                values = raw[:, selected].reshape(-1)
            nan, inf = int(np.isnan(values).sum()), int(np.isinf(values).sum())
            state.count += values.size
            state.nan += nan
            state.inf += inf
            fraction = nan / values.size if values.size else 0.0
            state.maximum_block_nan_fraction = max(state.maximum_block_nan_fraction, fraction)
            block_limit = (
                rule.max_block_nan_fraction
                if rule and rule.max_block_nan_fraction is not None
                else rule.max_nan_fraction if rule else None
            )
            excessive_missing = block_limit is not None and fraction > block_limit
            state.excessive_missing_blocks += int(excessive_missing)
            finite = np.isfinite(values)
            data = values[finite].astype("f4")
            processing_bad = int((data < 0).sum()) if definition.variable == "tp" else 0
            state.invalid_processing_values += processing_bad
            state.finite += data.size
            if not data.size:
                anomalies[name] = {"nan": nan, "inf": inf, "valid_count": values.size}
                continue
            if rule and rule.domain != "stored":
                data *= self.std[index]
                data += self.mean[index]
                if rule.domain == "physical":
                    with np.errstate(over="ignore", invalid="ignore"):
                        if definition.variable == "tp":
                            np.expm1(data, out=data)
                            data /= 1000.0  # source metres; never clip errors away
                        elif definition.variable in ("q", "q2m"):
                            data /= 1000.0
                        elif definition.radiation_seconds and definition.variable in (
                            "ssr",
                            "ssrd",
                            "fdir",
                            "ttr",
                        ):
                            data *= definition.radiation_seconds
            inverse_bad = int((~np.isfinite(data)).sum())
            state.inverse_nonfinite += inverse_bad
            lo, hi = float(data.min()), float(data.max())
            state.minimum = lo if state.minimum is None else min(state.minimum, lo)
            state.maximum = hi if state.maximum is None else max(state.maximum, hi)
            constant = bool(lo == hi)
            state.constant_blocks += int(constant)
            outside = (
                int(((data < rule.bounds[0]) | (data > rule.bounds[1])).sum())
                if rule and rule.bounds
                else 0
            )
            state.outside += outside
            if (
                nan
                or inf
                or inverse_bad
                or outside
                or processing_bad
                or excessive_missing
                or (constant and rule and not rule.allow_constant)
            ):
                anomalies[name] = {
                    "nan": nan,
                    "inf": inf,
                    "outside": outside,
                    "constant": constant,
                    "inverse_nonfinite": inverse_bad,
                    "invalid_processing_values": processing_bad,
                    "nan_fraction": fraction,
                }
        return anomalies

    def finish(self, report: ValidationReport) -> None:
        from dataclasses import asdict

        for definition in self.spec.product.channels:
            name = definition.name
            state = self.states[name]
            rule: QualityRule | None = self.rules.get(name)
            metrics = asdict(state)
            metrics["nan_fraction"] = state.nan / state.count if state.count else None
            metrics["finite_fraction"] = state.finite / state.count if state.count else None
            metrics["domain"] = rule.domain if rule else "stored"
            metrics["valid_region"] = (
                rule.valid_region if rule else "all pixels (stored-domain diagnostics)"
            )
            report.quality[name] = metrics
            if definition.variable == "tp":
                if rule is None and self.spec.validation_scope != "confirmed-format":
                    status = "needs-review"
                elif not state.count:
                    status = "not-run"
                else:
                    status = "fail" if state.invalid_processing_values else "pass"
                report.add(
                    f"quality.{name}.processing_domain",
                    status,
                    "data_quality",
                    "clip_min(0) -> log1p implies finite stored tp >= 0",
                    state.invalid_processing_values,
                )
            if rule is None:
                if self.spec.validation_scope == "confirmed-format":
                    if not state.count:
                        status = "not-run"
                    else:
                        status = "fail" if state.inf or state.invalid_processing_values else "pass"
                    report.add(
                        f"quality.{name}",
                        status,
                        "data_quality",
                        "No Inf; finite stored tp >= 0 when applicable",
                        metrics,
                        "NaN fractions, finite ranges and constant fields are diagnostics without acceptance thresholds",
                    )
                    continue
                report.add(
                    f"quality.{name}",
                    "needs-review",
                    "data_quality",
                    evidence="Missing independently approved per-channel quality policy",
                )
            elif not state.count:
                report.add(
                    f"quality.{name}", "not-run", "data_quality", evidence="No checked valid pixels"
                )
            else:
                passed = (
                    not state.inf
                    and not state.inverse_nonfinite
                    and not state.outside
                    and not state.invalid_processing_values
                    and not state.excessive_missing_blocks
                    and state.nan / state.count <= rule.max_nan_fraction
                    and (rule.allow_constant or not state.constant_blocks)
                )
                report.require(
                    f"quality.{name}", passed, "data_quality", asdict(rule), metrics, rule.evidence
                )
