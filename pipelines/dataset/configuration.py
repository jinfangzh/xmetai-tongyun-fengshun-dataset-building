"""Runtime validation specifications, execution config loading and evidence binding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from datakit.schema.parser import load_schema, resolve_config_path
from datakit.schema.specifications import DatasetSchema, EvidenceReference

__all__ = [
    "DatasetValidationSpec",
    "apply_execution",
    "load_schema_reference",
    "load_validation_run",
    "load_validation_schema",
    "read_frozen_json",
    "validation_input_files",
    "verify_execution_unchanged",
]


@dataclass(frozen=True)
class DatasetValidationSpec(DatasetSchema):
    """One run's evidence references; never serialized back into its schema."""

    production_evidence: EvidenceReference | None = None
    snapshot_id: str | None = None
    conversion_evidence: EvidenceReference | None = None
    execution_config_path: Path | None = None
    execution_config_sha256: str | None = None

    @classmethod
    def from_schema(cls, schema: DatasetSchema) -> DatasetValidationSpec:
        return cls(
            **{item.name: getattr(schema, item.name) for item in fields(DatasetSchema)}
        )

    def validate_configuration(self) -> None:
        super().validate_configuration()
        if self.snapshot_id is not None and (
            not isinstance(self.snapshot_id, str) or not self.snapshot_id.strip()
        ):
            raise ValueError("Snapshot identity must be a nonempty string")
        if (self.execution_config_path is None) != (
            self.execution_config_sha256 is None
        ):
            raise ValueError(
                "Execution configuration requires both source path and SHA256"
            )
        if (
            self.execution_config_path is not None
            and self.execution_config_sha256 is not None
        ):
            EvidenceReference(self.execution_config_path, self.execution_config_sha256)


def _reject_retired_model_fields(document: dict[str, Any]) -> None:
    retired = set(document) & {
        "consumer_required",
        "consumer_evidence",
        "model_evidence",
    }
    if retired:
        raise ValueError(
            "Consumer/model checks are no longer supported: "
            + ", ".join(sorted(retired))
        )


def read_frozen_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    result = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    _reject_retired_model_fields(result)
    return result, hashlib.sha256(raw).hexdigest()


def load_validation_schema(path: Path) -> tuple[DatasetValidationSpec, dict[str, Any]]:
    """Preserve the workflow entry point without duplicating the YAML parser."""
    schema, locations = load_schema(path)
    return DatasetValidationSpec.from_schema(schema), locations


def apply_execution(
    spec: DatasetValidationSpec,
    execution: dict[str, Any],
    *,
    config_path: Path,
    config_sha256: str,
) -> DatasetValidationSpec:
    allowed = {
        "snapshot_id",
        "production_evidence",
        "conversion_evidence",
        "conversion_required",
    }
    if not isinstance(execution, dict) or set(execution) - allowed:
        raise ValueError("Execution inputs contain unknown fields or are not a mapping")
    values: dict[str, Any] = {}
    if "snapshot_id" in execution:
        snapshot = execution["snapshot_id"]
        if not isinstance(snapshot, str) or not snapshot.strip():
            raise ValueError("execution.snapshot_id must be a nonempty string")
        values["snapshot_id"] = snapshot
    for name in ("production_evidence", "conversion_evidence"):
        if name not in execution:
            continue
        reference = execution[name]
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ValueError(f"execution.{name} requires exactly path and sha256")
        values[name] = EvidenceReference(
            resolve_config_path(config_path, reference["path"]),
            reference["sha256"],
        )
    if "conversion_required" in execution:
        required = execution["conversion_required"]
        if type(required) is not bool:
            raise ValueError("execution.conversion_required must be boolean")
        if spec.conversion_required and not required:
            raise ValueError(
                "Execution inputs cannot relax schema requirement conversion_required"
            )
        values["conversion_required"] = required
    result = replace(
        spec,
        **values,
        execution_config_path=config_path.resolve(),
        execution_config_sha256=config_sha256,
    )
    result.validate_configuration()
    return result


def load_schema_reference(
    config_path: Path,
    document: dict[str, Any],
    *,
    config_sha256: str,
    schema_key: str = "schema",
) -> tuple[DatasetValidationSpec, dict[str, Any]]:
    _reject_retired_model_fields(document)
    spec, locations = load_validation_schema(
        resolve_config_path(config_path, document[schema_key])
    )
    spec = apply_execution(
        spec,
        document.get("execution", {}),
        config_path=config_path,
        config_sha256=config_sha256,
    )
    verify_execution_unchanged(spec)
    return spec, locations


def load_validation_run(path: Path) -> tuple[DatasetValidationSpec, dict[str, Any]]:
    path = path.expanduser().resolve()
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8-sig"))
    allowed = {"schema", "dataset_path", "auxiliary_directory", "execution"}
    if (
        not isinstance(document, dict)
        or "schema" not in document
        or set(document) - allowed
    ):
        raise ValueError(
            f"{path}: a run requires schema and only supported execution fields"
        )
    spec, locations = load_validation_schema(
        resolve_config_path(path, document["schema"])
    )
    spec = apply_execution(
        spec,
        document.get("execution", {}),
        config_path=path,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )
    for name in ("dataset_path", "auxiliary_directory"):
        if name in document:
            locations[name] = str(resolve_config_path(path, document[name]))
    if "dataset_path" in document and "auxiliary_directory" not in document:
        locations["auxiliary_directory"] = locations["dataset_path"]
    return spec, locations


def verify_execution_unchanged(spec: DatasetValidationSpec) -> None:
    if spec.execution_config_path is not None:
        digest = hashlib.sha256(spec.execution_config_path.read_bytes()).hexdigest()
        if digest != spec.execution_config_sha256:
            raise ValueError(
                f"Execution configuration changed: {spec.execution_config_path}"
            )


def validation_input_files(spec: DatasetValidationSpec) -> tuple[Path | None, ...]:
    references = [
        spec.production_evidence,
        spec.conversion_evidence,
        *spec.valid_regions.values(),
    ]
    return (
        spec.schema_path,
        spec.execution_config_path,
        *(reference.path for reference in references if reference is not None),
    )
