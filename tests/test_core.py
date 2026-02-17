"""
Tests for the core logic of the head_direction package.
Covers angle calculation, tuning curve generation, and statistical scoring.
"""

from typing import Any, cast

import numpy as np

from head_direction.core import (
    head_direction,
    head_direction_rate,
    head_direction_score,
)


class MockQuantity:
    """Mock class to simulate a Neo/Quantity object for duck-typing tests."""

    def __init__(self, magnitude, units="s"):
        self.magnitude = np.array(magnitude)
        self.units = units

    def rescale(self, target_unit):
        # specific mock for testing conversion logic
        if self.units == "ms" and target_unit == "s":
            return MockQuantity(self.magnitude / 1000.0, "s")
        return self


def test_head_direction_45():
    """
    Test that a vector pointing (+1, +1) yields 45 degrees (pi/4 radians).
    Target: North-East.
    Logic: Vector = Front(1) - Back(2).
    Need dx > 0 (x1 > x2) and dy > 0 (y1 > y2).
    """
    x1 = np.linspace(0.01, 1, 10)
    y1 = x1

    # Back LED (2) is 'behind' Front LED (1)
    x2 = x1 - 0.01
    y2 = y1 - 0.01

    t = x1
    a, _ = head_direction(x1, y1, x2, y2, t)
    assert np.allclose(a, np.pi / 4)


def test_head_direction_135():
    """
    Test that a vector pointing (-1, +1) yields 135 degrees (3pi/4 radians).
    Target: North-West.
    Logic: Need dx < 0 (x1 < x2) and dy > 0 (y1 > y2).
    """
    x1 = np.linspace(0.01, 1, 10)[::-1]
    y1 = x1[::-1]

    # x1 < x2
    x2 = x1 + 0.01
    # y1 > y2
    y2 = y1 - 0.01

    t = x1
    a, _ = head_direction(x1, y1, x2, y2, t)
    assert np.allclose(a, np.pi - np.pi / 4)


def test_head_direction_225():
    """
    Test that a vector pointing (-1, -1) yields 225 degrees (5pi/4 radians).
    Target: South-West.
    Logic: Need dx < 0 (x1 < x2) and dy < 0 (y1 < y2).
    """
    x1 = np.linspace(0.01, 1, 10)[::-1]
    y1 = x1

    # x1 < x2
    x2 = x1 + 0.01
    # y1 < y2
    y2 = y1 + 0.01

    t = x1
    a, _ = head_direction(x1, y1, x2, y2, t)
    assert np.allclose(a, np.pi + np.pi / 4)


def test_head_direction_reverse_315():
    """
    Test that a vector pointing (+1, -1) yields 315 degrees (7pi/4 radians).
    Target: South-East.
    Logic: Need dx > 0 (x1 > x2) and dy < 0 (y1 < y2).
    """
    x1 = np.linspace(0.01, 1, 10)
    y1 = x1[::-1]

    # x1 > x2
    x2 = x1 - 0.01
    # y1 < y2
    y2 = y1 + 0.01

    t = x1
    a, _ = head_direction(x1, y1, x2, y2, t)
    assert np.allclose(a, 2 * np.pi - np.pi / 4)


def test_head_rate_basic():
    """
    Test standard rate calculation.
    """
    # Create constant head direction at 45 degrees
    x1 = np.linspace(0.01, 1, 10)
    y1 = x1

    # Ensure this generates 45 degrees (x1 > x2, y1 > y2)
    x2 = x1 - 0.01
    y2 = y1 - 0.01

    t = np.linspace(0, 1, 10)

    angles, t_clean = head_direction(x1, y1, x2, y2, t)

    # 100 Hz firing rate
    sptr = np.linspace(0, 1, 100)

    # Use 9 bins to avoid 45 degrees falling exactly on a bin edge.
    bins, rate = head_direction_rate(sptr, angles, t_clean, num_bins=9, smoothing_window=0)

    target_idx = np.argmin(np.abs(bins - np.pi / 4))

    # Use nanargmax/nanmax to handle empty bins (which are NaN)
    peak_idx = np.nanargmax(rate)

    assert abs(peak_idx - target_idx) <= 1
    assert np.nanmax(rate) > 90


def test_head_rate_mock_neo():
    """
    Test that functions accept 'Neo-like' objects (duck typing).
    """
    spikes = MockQuantity(np.linspace(0, 1000, 10), units="ms")
    angles = np.zeros(10)
    times = np.linspace(0, 1, 10)

    # Use cast(Any, spikes) to bypass the strict type check for this mock object
    _, rate = head_direction_rate(cast("Any", spikes), angles, times, num_bins=4, smoothing_window=0)
    assert rate[0] > 0


def test_head_score():
    """
    Test vector length and mean angle calculation for a perfect tuning curve.
    """
    bins = np.linspace(0, 2 * np.pi, 360, endpoint=False) + (np.pi / 360)
    rate = np.zeros_like(bins)
    idx_90 = np.argmin(np.abs(bins - np.pi / 2))
    rate[idx_90] = 10.0

    ang, score = head_direction_score(bins, rate)

    assert abs(score - 1.0) < 0.001
    assert abs(ang - np.pi / 2) < 0.01


def test_head_score_uniform():
    """
    Test that a uniform firing rate across all angles results in a score of 0.
    """
    # endpoint=False ensures we don't double count 0 and 360 degrees.
    bins = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    rate = np.ones_like(bins)

    _, score = head_direction_score(bins, rate)
    assert score < 0.001
