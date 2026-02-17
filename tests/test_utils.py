"""
Tests for utility functions in head_direction.utils.
"""

import numpy as np
import pytest

from head_direction.utils import get_alignment_offset, moving_average, to_unitless


class MockObject:
    """Helper class to simulate Neo/Quantity objects for testing utilities."""

    def __init__(self, val, unit=None):
        self.magnitude = val
        self.unit = unit

    def rescale(self, target):
        if self.unit == "ms" and target == "s":
            return MockObject(self.magnitude / 1000.0, "s")
        return self


def test_to_unitless_array():
    """Test that a plain numpy array is returned unchanged."""
    data = np.array([1, 2, 3])
    out = to_unitless(data)
    assert np.array_equal(out, data)


def test_to_unitless_list():
    """Test that a list is correctly converted to a numpy array."""
    data = [1.0, 2.0, 3.0]
    out = to_unitless(data)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, np.array(data))


def test_to_unitless_mock_rescale():
    """
    Test that an object with .rescale() is correctly converted
    to the target unit before stripping units.
    """
    # Simulate a Quantity in ms
    data = MockObject(np.array([1000, 2000]), unit="ms")

    # Ask for seconds
    out = to_unitless(data, target_unit="s")

    # Should divide by 1000
    assert np.array_equal(out, np.array([1.0, 2.0]))


def test_to_unitless_mock_magnitude():
    """
    Test that an object with .magnitude but NO rescale capability
    (or no target unit requested) returns raw magnitude.
    """
    # Simulate an object that has magnitude (e.g. plain Quantity without target)
    data = MockObject(np.array([5, 5]))
    out = to_unitless(data)
    assert np.array_equal(out, np.array([5, 5]))


def test_moving_average_simple():
    """
    Test basic moving average functionality.
    Verifies that the average is calculated correctly for a simple window.
    """
    data = np.array([0, 0, 10, 0, 0])
    # Window 3.
    # Center (10): (0+10+0)/3 = 3.33
    # Neighbors: (0+0+10)/3 = 3.33
    res = moving_average(data, window_size=3)

    assert len(res) == len(data)
    assert np.allclose(res[2], 10 / 3)


def test_moving_average_circular():
    """
    Test circular (wrapping) behavior of the moving average.
    Verifies that the start and end of the array affect each other.
    """
    data = np.array([10, 0, 0, 0, 10])
    # Window 3.
    # Index 0 (10): Neighbors are [10 (end), 10 (self), 0]. Sum=20. Avg=6.66
    res = moving_average(data, window_size=3)

    assert np.allclose(res[0], 20 / 3)
    assert np.allclose(res[-1], 20 / 3)


def test_moving_average_window_error():
    """
    Test that a ValueError is raised if the window size is larger
    than the data length.
    """
    data = np.ones(5)
    with pytest.raises(ValueError):
        # Window size 6 is > 5. Should fail.
        moving_average(data, window_size=6)


def test_get_alignment_offset_90deg():
    """
    Test that the function correctly identifies a 90-degree offset.
    Scenario: Animal moves East (0 rad), but head points North (pi/2 rad).
    Result should be pi/2.
    """
    # 1. Simulate motion East (x increases, y constant)
    t = np.linspace(0, 10, 100)
    pos_x = t * 10.0  # Speed = 10 cm/s
    pos_y = np.zeros_like(t)

    # 2. Simulate head pointing North (pi/2)
    # The head angles are consistently pi/2
    angle_samples = np.full_like(t, np.pi / 2)

    offset = get_alignment_offset(angle_samples, pos_x, pos_y, t)

    # Expected: Head (pi/2) - Move (0) = pi/2
    assert np.allclose(offset, np.pi / 2)


def test_get_alignment_offset_stationary():
    """
    Test that the function safely returns 0.0 if the animal isn't moving.
    """
    t = np.linspace(0, 10, 100)
    pos_x = np.zeros_like(t)  # Not moving
    pos_y = np.zeros_like(t)
    angle_samples = np.zeros_like(t)

    # Should detect low movement and return 0.0 without crashing
    offset = get_alignment_offset(angle_samples, pos_x, pos_y, t, min_speed=5.0)
    assert offset == 0.0
