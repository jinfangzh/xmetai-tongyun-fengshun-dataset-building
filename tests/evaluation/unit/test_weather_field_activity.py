"""Unit tests for the deterministic activity-ratio metric."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from datakit.evaluation.contracts import ResultStatus
from datakit.evaluation.errors import MetricError
from datakit.evaluation.metrics.specialized import ActivityRatio
from tests.evaluation.helpers.batches import make_grid_batch, make_grid_field


def test_activity_identical_anomalies_returns_one():
    """Identical non-constant anomalies have activity ratio one."""
    climatology = [[10.0, 10.0], [10.0, 10.0]]
    field = [[11.0, 12.0], [13.0, 14.0]]

    result = ActivityRatio().compute(make_grid_batch(field, field, climatology))

    assert result.status is ResultStatus.SUCCESS
    assert result.value["ACTIVITY_RATIO"] == pytest.approx(1.0)
    assert result.value["FC_ACTIVITY"] == pytest.approx(np.sqrt(1.25))
    assert result.value["OBS_ACTIVITY"] == pytest.approx(np.sqrt(1.25))
    assert result.value["FC_OBS_BIAS"] == pytest.approx(0.0)


def test_activity_scaled_forecast_returns_expected_ratio():
    """Doubling every forecast anomaly doubles its weighted standard deviation."""
    climatology = [[10.0, 10.0], [10.0, 10.0]]
    observation = [[11.0, 12.0], [13.0, 14.0]]
    forecast = [[12.0, 14.0], [16.0, 18.0]]

    result = ActivityRatio().compute(make_grid_batch(forecast, observation, climatology))

    assert result.value["ACTIVITY_RATIO"] == pytest.approx(2.0)
    assert result.value["FC_ACTIVITY"] == pytest.approx(2.0 * np.sqrt(1.25))
    assert result.value["OBS_ACTIVITY"] == pytest.approx(np.sqrt(1.25))


def test_activity_is_invariant_to_uniform_anomaly_shift():
    """A uniform anomaly offset is removed by the area-weighted mean."""
    observation = [[1.0, 2.0], [3.0, 4.0]]
    forecast = [[6.0, 7.0], [8.0, 9.0]]
    climatology = [[0.0, 0.0], [0.0, 0.0]]

    result = ActivityRatio().compute(make_grid_batch(forecast, observation, climatology))

    assert result.value["ACTIVITY_RATIO"] == pytest.approx(1.0)
    assert result.value["FC_OBS_BIAS"] == pytest.approx(0.0)


def test_activity_uses_cosine_latitude_weights():
    """Errors at 60 degrees latitude receive half the equatorial weight."""
    field = [[0.0, 0.0], [2.0, 2.0]]
    result = ActivityRatio().compute(
        make_grid_batch(
            field,
            field,
            [[0.0, 0.0], [0.0, 0.0]],
            weights=[[1.0, 1.0], [0.5, 0.5]],
        )
    )

    expected_std = np.sqrt(8.0 / 9.0)
    assert result.value["ACTIVITY_RATIO"] == pytest.approx(1.0)
    assert result.value["FC_ACTIVITY"] == pytest.approx(expected_std)
    assert result.value["OBS_ACTIVITY"] == pytest.approx(expected_std)


def test_activity_requires_climatology():
    """Activity is an anomaly statistic and must reject a missing reference."""
    with pytest.raises(MetricError, match="需要气候态参考"):
        ActivityRatio().compute(
            make_grid_batch([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], None)
        )


def test_activity_zero_observation_variance_is_undefined():
    """A constant observation anomaly makes the activity ratio undefined."""
    result = ActivityRatio().compute(
        make_grid_batch(
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 2.0], [2.0, 2.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        )
    )

    assert result.status is ResultStatus.UNDEFINED
    assert np.isnan(result.value["ACTIVITY_RATIO"])
    assert result.value["OBS_ACTIVITY"] == pytest.approx(0.0)


def test_activity_all_points_invalid_returns_no_valid_data():
    """An empty valid domain must be reported explicitly."""
    result = ActivityRatio().compute(
        make_grid_batch(
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            valid_mask=[[False, False], [False, False]],
        )
    )

    assert result.status is ResultStatus.NO_VALID_DATA
    assert np.isnan(result.value["ACTIVITY_RATIO"])
    assert result.n_valid == 0


def test_activity_automatically_excludes_nan_and_inf():
    """The metric itself intersects its mask with finite forecast/reference data."""
    forecast = make_grid_field([[1.0, np.nan], [np.inf, 4.0]])
    observation = make_grid_field([[1.0, 2.0], [3.0, 4.0]])
    climatology = xr.zeros_like(observation)

    result = ActivityRatio().compute(make_grid_batch(forecast, observation, climatology))

    assert result.status is ResultStatus.SUCCESS
    assert result.value["ACTIVITY_RATIO"] == pytest.approx(1.0)
    assert result.n_valid == 2


def test_activity_merged_chunks_equal_whole_batch():
    """Mergeable weighted moments must reproduce whole-field activity."""
    metric = ActivityRatio()
    whole = make_grid_batch(
        [[1.0, 3.0], [6.0, 10.0]],
        [[1.0, 2.0], [4.0, 8.0]],
        [[0.0, 0.0], [0.0, 0.0]],
    )
    top = make_grid_batch([[1.0, 3.0]], [[1.0, 2.0]], [[0.0, 0.0]])
    bottom = make_grid_batch([[6.0, 10.0]], [[4.0, 8.0]], [[0.0, 0.0]])

    whole_result = metric.compute(whole)
    merged_result = metric.finalize(
        metric.merge([metric.accumulate(top), metric.accumulate(bottom)])
    )

    assert merged_result.status is ResultStatus.SUCCESS
    assert merged_result.value["ACTIVITY_RATIO"] == pytest.approx(
        whole_result.value["ACTIVITY_RATIO"]
    )
    assert merged_result.value["FC_ACTIVITY"] == pytest.approx(whole_result.value["FC_ACTIVITY"])
    assert merged_result.value["OBS_ACTIVITY"] == pytest.approx(whole_result.value["OBS_ACTIVITY"])


def test_activity_rejects_states_from_another_metric():
    """Sufficient statistics from another metric cannot be merged."""
    metric = ActivityRatio()
    state = metric.accumulate(make_grid_batch([[1.0]], [[1.0]], [[0.0]]))

    with pytest.raises(MetricError, match="different metrics"):
        metric.merge([state, replace(state, metric_name="rmse")])
