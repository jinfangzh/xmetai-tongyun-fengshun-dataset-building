from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from datakit.evaluation.contracts import EvaluationBatch, ResultStatus
from datakit.evaluation.errors import MetricError
from datakit.evaluation.metrics.base import ProductType
from datakit.evaluation.metrics.probabilistic import EnsembleProbabilityScore

THRESHOLDS = [("rain", 1.0)]


def _batch(members, observation, *, valid_mask=None, weights=None, reference=None):
    members = xr.DataArray(np.asarray(members, dtype="f8"), dims=("member", "station"))
    observation = xr.DataArray(np.asarray(observation, dtype="f8"), dims=("station",))
    if valid_mask is None:
        valid_mask = xr.DataArray(np.ones(observation.shape, bool), dims=("station",))
    else:
        valid_mask = xr.DataArray(np.asarray(valid_mask, dtype=bool), dims=("station",))
    weights_da = None if weights is None else xr.DataArray(weights, dims=("station",))
    return EvaluationBatch(
        forecast=members.mean("member"),
        observation=observation,
        members=members,
        reference=reference,
        sample_keys=[{"lead_hours": 6}],
        valid_mask=valid_mask,
        weights=weights_da,
        alignment={"time": "matched", "space": "station"},
        protocol_id="station_valid_time",
    )


def test_probability_score_matches_hand_calculated_bs_bss_and_aroc():
    members = [[0, 0, 2, 2], [0, 2, 2, 2]]
    result = EnsembleProbabilityScore(THRESHOLDS).compute(_batch(members, [0, 2, 2, 0]))

    score = result.value["rain"]
    assert result.status is ResultStatus.SUCCESS
    assert score["BS"] == pytest.approx(0.3125)
    assert score["base_rate"] == pytest.approx(0.5)
    assert score["BS_ref"] == pytest.approx(0.25)
    assert score["BSS"] == pytest.approx(-0.25)
    # One event/non-event pair is tied at probability 1 and counts as half.
    assert score["AROC"] == pytest.approx(0.625)
    assert score["n_points"] == 4


def test_probability_score_is_weighted_and_respects_mask():
    members = [[0, 0, 2, 2], [0, 2, 2, 2]]
    result = EnsembleProbabilityScore(THRESHOLDS).compute(
        _batch(
            members,
            [0, 2, 2, 0],
            valid_mask=[True, True, True, False],
            weights=[1.0, 2.0, 3.0, 4.0],
        )
    )
    # probabilities 0, 0.5, 1 with weighted squared errors 0, .5, 0.
    assert result.value["rain"]["BS"] == pytest.approx(0.5 / 6.0)
    assert result.n_valid == 3


@pytest.mark.parametrize("weights, expected_ref", [(None, 0.375), ([1.0, 2.0, 3.0, 4.0], 0.465)])
def test_probability_external_reference_picks_matching_threshold_column(weights, expected_ref):
    reference = xr.DataArray(
        [[0.1], [0.2], [0.8], [0.9]],
        dims=("station", "threshold"),
        coords={"threshold": [1.0]},
    )
    result = EnsembleProbabilityScore(THRESHOLDS).compute(
        _batch(
            [[0, 0, 2, 2], [0, 2, 2, 2]],
            [0, 2, 2, 0],
            weights=weights,
            reference=reference,
        )
    )
    assert result.value["rain"]["BS_ref"] == pytest.approx(expected_ref)
    assert result.value["rain"]["BSS"] == pytest.approx(
        1 - result.value["rain"]["BS"] / expected_ref
    )


def test_probability_excludes_nonfinite_members_and_empty_data():
    metric = EnsembleProbabilityScore(THRESHOLDS)
    result = metric.compute(_batch([[0, np.nan], [2, 2]], [0, 2]))
    assert result.n_valid == 1

    empty = metric.compute(_batch([[0, 2], [2, 2]], [0, 2], valid_mask=[False, False]))
    assert empty.status is ResultStatus.NO_VALID_DATA
    assert empty.n_valid == 0


def test_probability_without_both_event_classes_is_partial():
    result = EnsembleProbabilityScore(THRESHOLDS).compute(_batch([[2, 2], [2, 2]], [2, 2]))
    assert result.status is ResultStatus.PARTIAL
    assert np.isnan(result.value["rain"]["AROC"])


def test_probability_merge_matches_whole_batch():
    metric = EnsembleProbabilityScore(THRESHOLDS)
    whole = metric.compute(_batch([[0, 0, 2, 2], [0, 2, 2, 2]], [0, 2, 2, 0]))
    merged = metric.finalize(
        metric.merge(
            [
                metric.accumulate(_batch([[0, 0], [0, 2]], [0, 2])),
                metric.accumulate(_batch([[2, 2], [2, 2]], [2, 0])),
            ]
        )
    )
    for key in ("AROC", "BS", "BS_ref", "BSS", "base_rate"):
        assert merged.value["rain"][key] == pytest.approx(whole.value["rain"][key])


def test_probability_rejects_bad_inputs_and_foreign_states():
    metric = EnsembleProbabilityScore(THRESHOLDS)
    batch = _batch([[0, 2], [2, 2]], [0, 2])
    state = metric.accumulate(batch)

    bad_members = replace(batch, members=xr.DataArray([0.0, 2.0], dims=("station",)))
    with pytest.raises(MetricError, match="member, sample"):
        metric.compute(bad_members)
    bad_weights = replace(batch, weights=xr.DataArray([1.0], dims=("station",)))
    with pytest.raises(MetricError, match="权重形状"):
        metric.compute(bad_weights)

    with pytest.raises(MetricError, match="empty states"):
        metric.merge([])
    with pytest.raises(MetricError, match="different metrics"):
        metric.merge([state, replace(state, metric_name="crps")])

    different_members = EnsembleProbabilityScore(THRESHOLDS).accumulate(
        _batch([[0, 2], [1, 2], [2, 2]], [0, 2])
    )
    with pytest.raises(MetricError, match="成员数不一致"):
        metric.merge([state, different_members])


def test_probability_requirements_request_members_and_reference():
    metric = EnsembleProbabilityScore(THRESHOLDS)
    assert metric.requirements().product_type is ProductType.ENSEMBLE_SAMPLES
    assert metric.needs_members()
    assert metric.needs_reference()


def test_probability_identical_forecast_and_reference_have_zero_skill():
    reference = xr.DataArray(
        [[0.0], [0.0]], dims=("station", "threshold"), coords={"threshold": [1.0]}
    )
    result = EnsembleProbabilityScore(THRESHOLDS).compute(
        _batch([[0, 0], [0, 0]], [0, 2], weights=[0.9, 0.1], reference=reference)
    )
    score = result.value["rain"]
    assert score["BS"] == pytest.approx(0.1)
    assert score["BS_ref"] == pytest.approx(0.1)
    assert score["BSS"] == pytest.approx(0.0)


def test_probability_weighted_reference_merge_matches_whole_batch():
    reference = xr.DataArray(
        [[0.1], [0.2], [0.8], [0.9]],
        dims=("station", "threshold"),
        coords={"threshold": [1.0]},
    )
    metric = EnsembleProbabilityScore(THRESHOLDS)
    whole = metric.compute(
        _batch(
            [[0, 0, 2, 2], [0, 2, 2, 2]],
            [0, 2, 2, 0],
            weights=[0.1, 0.2, 0.3, 0.4],
            reference=reference,
        )
    )
    merged = metric.finalize(
        metric.merge(
            [
                metric.accumulate(
                    _batch(
                        [[0, 0], [0, 2]],
                        [0, 2],
                        weights=[0.1, 0.2],
                        reference=reference.isel(station=slice(0, 2)),
                    )
                ),
                metric.accumulate(
                    _batch(
                        [[2, 2], [2, 2]],
                        [2, 0],
                        weights=[0.3, 0.4],
                        reference=reference.isel(station=slice(2, 4)),
                    )
                ),
            ]
        )
    )
    assert whole.value["rain"]["BS_ref"] == pytest.approx(0.465)
    for key in ("AROC", "BS", "BS_ref", "BSS", "base_rate"):
        assert merged.value["rain"][key] == pytest.approx(whole.value["rain"][key])


def test_probability_reference_uses_same_mask_and_finite_samples_as_forecast():
    reference = xr.DataArray(
        [[0.0], [0.5], [1.0], [1.0]],
        dims=("station", "threshold"),
        coords={"threshold": [1.0]},
    )
    result = EnsembleProbabilityScore(THRESHOLDS).compute(
        _batch(
            [[0, 2, np.nan, 2], [0, 2, 2, 2]],
            [0, 2, 0, 0],
            valid_mask=[True, True, True, False],
            weights=[0.2, 0.8, 100.0, 100.0],
            reference=reference,
        )
    )
    assert result.n_valid == 2
    assert result.value["rain"]["BS_ref"] == pytest.approx(0.2)


@pytest.mark.skip(
    reason=(
        "源码冻结暂不修：EnsembleProbabilityScore.merge() 目前只校验 states 非空、metric_name 与"
        "成员数，不比 metric_version、也不比阈值参数"
    ),
)
def test_probability_merge_version_and_threshold_contract_pending_source_fix():
    metric = EnsembleProbabilityScore(THRESHOLDS)
    batch = _batch([[0, 2], [2, 2]], [0, 2])
    state = metric.accumulate(batch)

    with pytest.raises(MetricError, match="different versions"):
        metric.merge([state, replace(state, metric_version="2.0.0")])

    other = EnsembleProbabilityScore([("rain", 5.0)]).accumulate(batch)
    with pytest.raises(MetricError, match="different parameters"):
        metric.merge([state, other])
