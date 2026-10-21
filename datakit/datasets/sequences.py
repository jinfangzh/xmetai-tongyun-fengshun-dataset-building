"""Indices for fixed-frequency dataset sample windows."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

import numpy as np
import pandas as pd


def _times(values: Sequence[object]) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(pd.to_datetime(values))
    if times.hasnans:
        raise ValueError("Time coordinates cannot contain missing values")
    if not times.is_monotonic_increasing:
        raise ValueError("Time coordinates must be monotonically increasing")
    if not times.is_unique:
        raise ValueError("Time coordinates must be unique")
    return times


def make_sequence_indices(
    times: Sequence[object],
    total_frames: int,
    *,
    freq: int = 1,
    frequency: str | timedelta | pd.Timedelta | None = None,
    interval: int = 1,
) -> np.ndarray:
    """Return start indices for every complete fixed-frequency time window.

    ``freq`` retains the original integer-hours interface. Use ``frequency``
    for minute-level or other fixed intervals, for example ``"6min"``.
    """

    if total_frames < 1 or freq < 1 or interval < 1:
        raise ValueError("total_frames, freq and interval must be positive")
    if frequency is not None and freq != 1:
        raise ValueError("Provide either a non-default freq or frequency, not both")
    index = _times(times)
    if total_frames > len(index):
        return np.empty(0, dtype=np.int64)
    if frequency is None:
        expected = pd.Timedelta(hours=freq)
    else:
        try:
            expected = pd.Timedelta(frequency)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("frequency must be a fixed positive interval") from exc
        if pd.isna(expected) or expected <= pd.Timedelta(0):
            raise ValueError("frequency must be a fixed positive interval")
    valid = []
    for start in range(0, len(index) - total_frames + 1, interval):
        window = index[start : start + total_frames]
        if total_frames == 1 or np.all(
            np.diff(window.values) == expected.to_timedelta64()
        ):
            valid.append(start)
    return np.asarray(valid, dtype=np.int64)


