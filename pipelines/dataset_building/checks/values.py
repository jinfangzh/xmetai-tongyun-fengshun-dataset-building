"""Statistics, masks, auxiliary values and bounded main-data quality checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from datakit.schema.datasets.specifications import AuxiliaryDefinition, DatasetSchema, QualityRule

from ..support.reports import ValidationReport

STANDARD = "standard_conformance"


def check_statistics(
    mean: np.ndarray, std: np.ndarray, report: ValidationReport,
) -> bool:
    return report.require(
        "statistics.finite_positive",
        bool(np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()),
        STANDARD, "finite mean/std and std > 0", {"mean": mean, "std": std},
    )


def check_mask_values(
    mask: np.ndarray, spec: DatasetSchema, report: ValidationReport,
) -> bool:
    mask_schema = spec.product.mask
    return report.require(
        "mask.binary_complement",
        bool(
            np.isin(mask, mask_schema.allowed_values).all()
            and np.all(
                np.add(
                    mask[mask_schema.region_index("land")],
                    mask[mask_schema.region_index("sea")],
                    dtype=np.int64,
                ) == mask_schema.complement_sum
            )
        ),
        STANDARD,
        {"allowed_values": mask_schema.allowed_values, "complement_sum": mask_schema.complement_sum},
    )


def check_auxiliary_finite(
    values: np.ndarray, definition: AuxiliaryDefinition, report: ValidationReport,
) -> None:
    prefix = f"auxiliary.{definition.filename}"
    if definition.finite:
        report.require(f"{prefix}.finite", bool(np.isfinite(values).all()), "data_quality")
    else:
        report.add(
            f"{prefix}.finite", "not-applicable", "data_quality",
            evidence="Finite-value checks are not required by this standard",
        )


def check_auxiliary_values(
    values: np.ndarray,
    definition: AuxiliaryDefinition,
    spec: DatasetSchema,
    mask: np.ndarray | None,
    report: ValidationReport,
) -> None:
    """Check values only after the auxiliary header and layout have passed."""
    prefix = f"auxiliary.{definition.filename}"
    if definition.role == "weight":
        tolerance = spec.auxiliary_tolerances.get(
            definition.filename, definition.weight_nonnegative_atol,
        )
        nonnegative = report.require(
            f"{prefix}.weight_nonnegative",
            bool(
                (values >= -tolerance).all()
                and (not definition.require_positive_value or (values > 0).any())
            ),
            "data_quality",
            {"minimum": -tolerance, "positive_pixels_required": definition.require_positive_value},
            {
                "minimum": float(values.min()), "maximum": float(values.max()),
                "negative_count": int((values < 0).sum()),
                "below_tolerance_count": int((values < -tolerance).sum()),
            },
            spec.auxiliary_provenance.get(definition.filename),
        )
        if nonnegative and bool((values < 0).any()):
            report.add(
                f"{prefix}.weight_roundoff", "warning", "data_quality",
                {"absolute_tolerance": tolerance},
                {"minimum": float(values.min()), "negative_count": int((values < 0).sum())},
                "Within the approved nonnegative tolerance; raw weights were not clipped or changed",
            )
    if definition.equals_mask:
        if mask is None:
            report.add(
                f"{prefix}.mask_equal", "blocked", STANDARD, evidence="Main mask layout invalid",
            )
        else:
            report.require(
                f"{prefix}.mask_equal",
                np.array_equal(
                    values, mask[spec.product.mask.channels.index(definition.equals_mask)],
                ),
                STANDARD,
            )
    if (
        definition.role == "const"
        and spec.product.mask.require_provenance
        and spec.mask_derivation == "lsm>0.5"
    ):
        if "lsm" not in definition.channels or mask is None:
            report.add(
                f"{prefix}.lsm_derivation", "blocked", STANDARD,
                evidence="Approved derivation requires declared lsm channel",
            )
        else:
            report.require(
                f"{prefix}.lsm_derivation",
                np.array_equal(
                    values[definition.channels.index("lsm")] > 0.5,
                    mask[spec.product.mask.region_index("land")] == spec.product.mask.valid_value,
                ),
                STANDARD, "Approved lsm>0.5 relation to unchanged binary mask",
            )


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
        spec: DatasetSchema,
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
            processing_bad = (
                int((data * self.std[index] + self.mean[index] < 0).sum())
                if definition.nonnegative_preprocessed else 0
            )
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
                        for step in reversed(definition.preprocessing):
                            if step["operation"] == "log1p":
                                np.expm1(data, out=data)
                            elif step["operation"] == "multiply":
                                data /= step["factor"]
                            elif step["operation"] == "divide":
                                data *= step["divisor"]
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
            if definition.nonnegative_preprocessed:
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
                    "Nonnegative clip_min -> log1p requires finite preprocessed values >= 0",
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
