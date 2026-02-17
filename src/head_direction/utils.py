from typing import Any, cast

import numpy as np

from .type_alias import ArrayLike


def to_unitless(data: Any, target_unit: str | None = None) -> np.ndarray:
    """
    Safely converts a Neo object, Quantity, or numpy array into a
    plain, unitless numpy array.

    Parameters
    ----------
    data : Any
        Input data (Quantity, SpikeTrain, or array).
    target_unit : str, optional
        If provided (e.g., 's'), attempts to rescale the data
        before stripping units.

    Returns
    -------
    np.ndarray
        Unitless magnitude of the data.
    """
    # Try rescaling (if unit provided)
    if target_unit and hasattr(data, "rescale"):
        # Treat data as Any for a moment.
        # This allows us to access .rescale and .magnitude without errors.
        return cast(Any, data).rescale(target_unit).magnitude

    # Try getting magnitude (raw units)
    if hasattr(data, "magnitude"):
        return cast(Any, data).magnitude

    # Fallback (assume it's already a list/array)
    return np.asarray(data)


def moving_average(data: ArrayLike, window_size: int) -> np.ndarray:
    """
    Circular moving average (boxcar filter).

    The input is treated as circular (periodic), so the start connects
    to the end for smoothing.

    Parameters
    ----------
    data : ArrayLike
        Vector to smooth.
    window_size : int
        Length of the boxcar window.

    Returns
    -------
    np.ndarray
        Smoothed vector of the same shape as input.

    Raises
    ------
    ValueError
        If the window size is too large (> half the data length).
    """
    # Create a float copy to avoid mutating the original input
    arr = np.array(data, dtype=float)

    # Handle NaNs by treating them as 0
    arr[np.isnan(arr)] = 0.0

    # Window needs to be smaller than or equal to data length
    if window_size > len(arr):
        raise ValueError(f"Window size ({window_size}) cannot be larger than data length ({len(arr)}).")

    # Circular padding: Pad the end with the start, and the start with the end
    padded_arr = np.concatenate((arr[-window_size:], arr, arr[:window_size]))

    # Convolve with normalized boxcar kernel
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded_arr, kernel, mode="same")

    # Slice strictly the middle valid part corresponding to original data
    return smoothed[window_size:-window_size]


def get_alignment_offset(
    angle_samples: np.ndarray, pos_x: ArrayLike, pos_y: ArrayLike, time_stamps: ArrayLike, min_speed: float = 2.0
) -> float:
    """
    Calculates the angular offset between the calculated head direction and
    the actual movement direction (velocity vector).

    This is useful for detecting if LEDs are mounted Front/Back (offset~0),
    Left/Right (offset~90), or inverted (offset~180).

    Parameters
    ----------
    angle_samples : np.ndarray
        Head direction angles in radians (from head_direction function).
    pos_x, pos_y : ArrayLike
        Position of the animal (usually centroid or LED1).
    time_stamps : ArrayLike
        Timestamps.
    min_speed : float
        Minimum speed (cm/s) required to include a sample.
        Stationary animals don't define a movement vector well.

    Returns
    -------
    offset : float
        The circular mean difference between Head Angle and Movement Angle.
        Subtract this value from your head direction to align it with movement.
    """
    x = np.asarray(pos_x)
    y = np.asarray(pos_y)
    t = np.asarray(time_stamps)

    # Calculate Velocity Vector
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    speed = np.hypot(vx, vy)
    movement_angle = np.arctan2(vy, vx)
    movement_angle[movement_angle < 0] += 2 * np.pi

    # Filter for frames where the animal is actually moving
    # (head direction is independent of motion, but motion direction is undefined at rest)
    moving_mask = speed > min_speed

    if np.sum(moving_mask) < 10:
        # Not enough moving data to calibrate
        return 0.0

    valid_head = angle_samples[moving_mask]
    valid_move = movement_angle[moving_mask]

    # Calculate difference using circular statistics
    # diff = head - move
    # We use complex numbers for robust circular mean
    diffs = valid_head - valid_move
    mean_vector = np.mean(np.exp(1j * diffs))

    # Return offset in radians
    return float(np.angle(mean_vector))
