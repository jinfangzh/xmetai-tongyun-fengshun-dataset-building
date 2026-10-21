"""Reduce ensemble members while preserving evaluation's DataBundle contract."""

from dataclasses import replace

from datakit.evaluation.contracts import DataBundle


def compute_ensemble_mean(
    bundle: DataBundle, member_dim: str = "member", skipna: bool = False
) -> DataBundle:
    """Average members and record the reduction; missing members propagate by default."""
    data = bundle.payload
    if member_dim not in data.dims:
        return bundle

    # Keep the existing Dataset result for both Dataset and DataArray inputs.
    ds = data if hasattr(data, "data_vars") else data.to_dataset()
    mean_ds = ds.mean(dim=member_dim, skipna=skipna)

    semantic = replace(bundle.semantic, member_count=None)
    provenance = replace(
        bundle.provenance,
        transform_chain=bundle.provenance.transform_chain + [f"ensemble_mean_over_{member_dim}"],
    )
    return DataBundle(
        payload=mean_ds,
        kind=bundle.kind,
        source_id=bundle.source_id,
        standard_vars=bundle.standard_vars,
        semantic=semantic,
        provenance=provenance,
        quality=bundle.quality,
    )


class EnsembleMeanTransform:
    """Evaluation transform from a member DataBundle to its mean."""

    def __init__(self, member_dim: str = "member"):
        self.member_dim = member_dim

    def transform(self, bundle: DataBundle) -> DataBundle:
        return compute_ensemble_mean(bundle, member_dim=self.member_dim)
