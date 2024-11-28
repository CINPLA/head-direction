# -*- coding: utf-8 -*-
import numpy as np


def unit_vector(v):
    """Return unit vector of v
    modified from David Wolever,
    https://stackoverflow.com/questions/2827393/angles
    -between-two-n-dimensional-vectors-in-python
    """
    return v / np.linalg.norm(v)


def moving_average(vector, N):
    """
    Circular moving average (box car) using convolution with square function of
    unity. By circular we connect the start with the end before averaging.
    Parameters
    ----------
    vector : np.array
        Vector of desired averaging.
    N : int
        Length of square function (box car).
    Returns
    -------
    np.array
    Note
    ----
    Equal, but faster than:
    vector = np.concatenate((vector[-N:], vector, vector[:N]))
    return [sum(vector[i:i + N]) / N for i in range(N - 2, len(vector) - N - 2)]
    Examples
    -------
    >>> a = np.ones((10, ))
    >>> moving_average(a, 5)
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    >>> a = np.concatenate((np.arange(5), np.arange(5)[::-1]))
    >>> print(a)
    [0 1 2 3 4 4 3 2 1 0]
    >>> moving_average(a, 5)
    array([0.8, 1.2, 2. , 2.8, 3.2, 3.2, 2.8, 2. , 1.2, 0.8])
    >>> a = np.arange(10)
    >>> moving_average(a, 5)
    array([4., 3., 2., 3., 4., 5., 6., 7., 6., 5.])
    """
    vector[np.isnan(vector)] = 0
    if N * 2 > len(vector):
        raise ValueError('Window must be at least half of "len(vector)"')
    vector = np.concatenate((vector[-N:], vector, vector[:N]))
    return np.convolve(vector, np.ones((N,)) / N, mode="same")[N:-N]


def angle_between_vectors(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'
    modified from David Wolever,
    https://stackoverflow.com/questions/2827393/angles
    -between-two-n-dimensional-vectors-in-python
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
