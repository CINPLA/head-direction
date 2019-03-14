import numpy as np
from .tools import moving_average


def head_direction_rate(spike_train, head_angles, t,
                        n_bins=36, avg_window=4):
    """
    Calculeate firing rate at head direction in binned head angles for time t.
    Moving average filter is applied on firing rate

    Parameters
    ----------
    spike_train : array
    head_angles : array in degrees
        all recorded head directions
    t : array
        1d vector of times at x, y positions
    avg_window : int
        number of bins to average over

    Returns
    -------
    out : np.ndarray, np.ndarray
        binned angles, avg rate in corresponding bins
    """
    assert len(head_angles) == len(t)
    # make bins around angle measurements
    spikes_in_bin, _ = np.histogram(spike_train, t)
    spikes_in_bin = np.append(spikes_in_bin, 0)
    # take out the first and every other bin
    time_in_bin = np.diff(t)
    time_in_bin = np.append(time_in_bin, 0)

    # bin head_angles
    ang_bins = np.linspace(0, 2.0 * np.pi, n_bins + 1)

    spikes_in_ang, _ = np.histogram(head_angles, weights=spikes_in_bin, bins=ang_bins)
    time_in_ang, _ = np.histogram(head_angles, weights=time_in_bin, bins=ang_bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        rate_in_ang = np.divide(spikes_in_ang, time_in_ang)

    rate_in_ang = moving_average(rate_in_ang, avg_window)
    return ang_bins[:-1], rate_in_ang


def head_direction_score(head_angle_bins, rate):
    """
    Calculeate firing rate at head direction in head angles for time t

    Parameters
    ----------
    head_angle_bins : array in radians
        binned head directions
    rate : array
        firing rate magnitude coresponding to angles

    Returns
    -------
    out : float, float
        mean angle, mean vector length
    """
    import math
    import pycircstat as pc
    # if any(np.isnan(rate)):
        # raise ValueError('Nan not supported')
    nanIndices = np.where(np.isnan(rate))
    head_angle_bins = np.delete(head_angle_bins, nanIndices)
    mean_ang = pc.mean(head_angle_bins, w=rate)
    mean_vec_len = pc.resultant_vector_length(head_angle_bins, w=rate)
    # ci_lim = pc.mean_ci_limits(head_angle_bins, w=rate)
    return mean_ang, mean_vec_len


def head_direction(x1, y1, x2, y2, t, filt=2.):
    """
    Calculeate head direction in angles or radians for time t

    Parameters
    ----------
    x1 : quantities.Quantity array in m
        1d vector of x positions from LED 1
    y1 : quantities.Quantity array in m
        1d vector of y positions from LED 1
    x2 : quantities.Quantity array in m
        1d vector of x positions from LED 2
    y2 : quantities.Quantity array in m
        1d vector of x positions from LED 2
    t : quantities.Quantity array in s
        1d vector of times from LED 1 or 2 at x, y positions
    filt : float
        threshold filter all LED distances less than filt*std(dist)

    Returns
    -------
    out : angles, resized t
    """
    dx = x2 - x1
    dy = y1 - y2
    dr = np.array([dx, dy])
    r = np.linalg.norm(dr, axis=0)
    r_mean = np.mean(r)
    r_std = np.std(r)
    mask = (r > r_mean - filt*r_std)
    x1 = x1[mask]
    y1 = y1[mask]
    x2 = x2[mask]
    y2 = y2[mask]

    # Calculate angles in range [0, 2pi]:
    dx = x2 - x1
    dy = y1 - y2

    angles_rad = np.arctan2(dy, dx)
    tmpIndices = np.where(angles_rad < 0)
    angles_rad[tmpIndices] += 2 * np.pi

    return angles_rad, t[mask]
