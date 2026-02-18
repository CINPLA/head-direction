import logging

import numpy as np

from .type_alias import ArrayLike, SpikeInput
from .utils import moving_average, to_unitless

logger = logging.getLogger(__name__)


def head_direction_rate(
    spike_times: SpikeInput,
    angle_samples: np.ndarray,
    time_stamps: np.ndarray,
    num_bins: int = 36,
    smoothing_window: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate firing rate as a function of head direction (tuning curve).

    The function bins the head direction samples and computes the firing rate
    within each angular bin, normalized by the time spent in that direction.

    Parameters
    ----------
    spike_times : SpikeInput
        The timestamps of spikes. Can be a neo.SpikeTrain, Quantity array,
        plain numpy array, or a list of these. Units are converted to seconds.
    angle_samples : np.ndarray
        Head direction samples corresponding to `time_stamps`.
        Must be in **radians** (0 to 2pi).
    time_stamps : np.ndarray
        1D vector of timestamps corresponding to the `angle_samples`.
        These define the temporal bins for occupancy.
    num_bins : int, optional
        Number of angular bins to divide 0-2pi into. Default is 36.
    smoothing_window : int, optional
        Size of the moving average window (in bins) for smoothing. Default is 4.

    Returns
    -------
    angle_centers : np.ndarray
        The centers of the angular bins (radians).
    tuning_curve : np.ndarray
        The smoothed firing rate (Hz) for each angular bin.
    """
    if len(angle_samples) != len(time_stamps):
        raise ValueError(
            f"Shape mismatch: angle_samples ({len(angle_samples)}) and "
            f"time_stamps ({len(time_stamps)}) must have the same length."
        )

    # Flatten and standardize spike times
    flat_spikes = _to_unitless_seconds(spike_times)

    # Ensure time_stamps is unitless seconds
    time_stamps = to_unitless(time_stamps, target_unit="s")

    # Temporal binning (occupancy)
    # We use time_stamps as bin edges.
    # NOTE: histogram(bins=time_stamps) produces N-1 counts.
    # The recorded angle_samples usually correspond to these moments.
    spikes_per_frame, _ = np.histogram(flat_spikes, bins=time_stamps)

    # Pad to match length of angle_samples (N)
    # Appending 0 implies no spikes occurred after the last timestamp
    spikes_per_frame = np.append(spikes_per_frame, 0)

    # Calculate duration of each frame (dt)
    dt_per_frame = np.diff(time_stamps)
    dt_per_frame = np.append(dt_per_frame, 0)

    # Angular binning
    angle_edges = np.linspace(0, 2.0 * np.pi, num_bins + 1)

    # Weighted histogram: Sum of 'spikes per frame' for each angle bin
    spikes_per_angle, _ = np.histogram(angle_samples, weights=spikes_per_frame, bins=angle_edges)

    # Weighted histogram: Sum of 'duration per frame' for each angle bin
    time_per_angle, _ = np.histogram(angle_samples, weights=dt_per_frame, bins=angle_edges)

    # Rate calculation
    with np.errstate(divide="ignore", invalid="ignore"):
        tuning_curve = np.divide(spikes_per_angle, time_per_angle)

    # Smoothing
    if smoothing_window > 0:
        tuning_curve = moving_average(tuning_curve, smoothing_window)

    # Calculate bin centers
    angle_centers = angle_edges[:-1] + np.diff(angle_edges) / 2

    return angle_centers, tuning_curve


def head_direction_score(angle_bins: np.ndarray, tuning_curve: np.ndarray) -> tuple[float, float]:
    """
    Calculate statistical metrics for the head direction tuning curve.

    Computes the mean vector length (Rayleigh vector length) and the
    mean angle using polar coordinate statistics.

    Parameters
    ----------
    angle_bins : np.ndarray
        Centers of the angular bins (radians).
    tuning_curve : np.ndarray
        Firing rate (Hz) corresponding to each angle bin (used as weights).

    Returns
    -------
    mean_angle : float
        The preferred direction in radians [0, 2pi].
    mean_vector_length : float
        Directionality score between 0 (uniform) and 1 (perfectly directional).
    """
    # Filter out NaN rates
    valid_mask = ~np.isnan(tuning_curve)
    angles = angle_bins[valid_mask]
    rates = tuning_curve[valid_mask]

    if np.sum(rates) == 0:
        return np.nan, np.nan

    # Polar center of mass
    rx = np.sum(rates * np.cos(angles))
    ry = np.sum(rates * np.sin(angles))
    total_rate = np.sum(rates)

    mean_vector_length = np.sqrt(rx**2 + ry**2) / total_rate
    mean_angle = np.arctan2(ry, rx)

    if mean_angle < 0:
        mean_angle += 2 * np.pi

    return mean_angle, mean_vector_length


def head_direction(
    led1_x: ArrayLike,
    led1_y: ArrayLike,
    led2_x: ArrayLike,
    led2_y: ArrayLike,
    time_stamps: ArrayLike,
    std_filter_threshold: float = 2.0,
    offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate head direction from two LED position tracking signals.

    Filters outliers based on the distance between LEDs and computes
    the angle of the vector from LED2 (Back) to LED1 (Front).

    The `offset` parameter allows users to correct for LED placement.
    It represents the angular rotation of the LED vector relative to the
    nose. For example, if LEDs are mounted Left/Right (pointing +90 deg),
    an offset of +pi/2 is subtracted to align the result with the nose.

    Table of common offsets (assuming LED2->LED1 vector):

    ============ ============ =====================
    Offset (rad) LED 1        LED 2
    ============ ============ =====================
    0.0          Front        Back (Assumed Default)
    pi/2         Left         Right
    pi           Back         Front (Inverted)
    -pi/2        Right        Left
    ============ ============ =====================

    Use `head_direction.get_alignment_offset` to find this value automatically.

    Parameters
    ----------
    led1_x, led1_y : ArrayLike
        Positions of the front/primary LED.
    led2_x, led2_y : ArrayLike
        Positions of the back/secondary LED.
    time_stamps : ArrayLike
        Timestamps corresponding to the position samples.
    std_filter_threshold : float, optional
        Threshold for artifact rejection (default=2.0).
    offset : float, optional
        Angular offset in radians to subtract from the calculated angle.
        Default is 0.0 (assumes LED1=Front, LED2=Back).

    Returns
    -------
    angles_rad : np.ndarray
        Head direction angles in radians [0, 2pi].
    clean_time_stamps : np.ndarray
        Timestamps corresponding to valid samples.
    """

    # Log assumption if using default offset
    if offset == 0.0:
        logger.info("head_direction called with offset=0.0. Assuming System: LED1=Front (Nose), LED2=Back.")

    x1 = np.asarray(led1_x)
    y1 = np.asarray(led1_y)
    x2 = np.asarray(led2_x)
    y2 = np.asarray(led2_y)
    t = np.asarray(time_stamps)

    # Validate LED distance consistency
    dx = x2 - x1
    dy = y1 - y2
    distances = np.hypot(dx, dy)

    r_mean = np.mean(distances)
    r_std = np.std(distances)

    # Filter bad tracking
    valid_mask = distances > (r_mean - std_filter_threshold * r_std)

    x1 = x1[valid_mask]
    y1 = y1[valid_mask]
    x2 = x2[valid_mask]
    y2 = y2[valid_mask]
    clean_time_stamps = t[valid_mask]

    # Angle calculation
    dx = x1 - x2
    dy = y1 - y2
    angles_rad = np.arctan2(dy, dx)

    # Apply user-supplied calibration offset
    angles_rad -= offset  # Subtract the offset to align 'true' North

    # Normalize to [0, 2pi]
    angles_rad = angles_rad % (2 * np.pi)

    return angles_rad, clean_time_stamps


def _to_unitless_seconds(data: SpikeInput) -> np.ndarray:
    """
    Helper to convert various spike input formats (Neo, Quantity, List)
    into a flat, unitless numpy array of seconds.
    """
    # Handle recursion for lists of SpikeTrains
    if isinstance(data, (list | tuple)):
        processed_items = [_to_unitless_seconds(item) for item in data]
        return np.concatenate(processed_items)
    return to_unitless(data, target_unit="s")
