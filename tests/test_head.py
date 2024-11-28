# -*- coding: utf-8 -*-
import numpy as np

from head_direction.head import (
    head_direction,
    head_direction_rate,
    head_direction_score,
)


def test_head_direction_45():
    x1 = np.linspace(0.01, 1, 10)
    y1 = x1
    x2 = x1 + 0.01  # 1cm between
    y2 = x1 - 0.01
    t = x1
    a, t = head_direction(x1, y1, x2, y2, t)
    assert np.allclose(a, np.pi / 4)


def test_head_direction_135():
    x1 = np.linspace(0.01, 1, 10)[::-1]
    y1 = x1[::-1]
    x2 = x1 - 0.01  # 1cm between
    y2 = y1 - 0.01
    t = x1
    a, t = head_direction(x1, y1, x2, y2, t)
    assert np.allclose(a, np.pi - np.pi / 4)


def test_head_direction_225():
    x1 = np.linspace(0.01, 1, 10)[::-1]
    y1 = x1
    x2 = x1 - 0.01  # 1cm between
    y2 = y1 + 0.01
    t = x1
    a, t = head_direction(x1, y1, x2, y2, t)
    assert np.allclose(a, np.pi + np.pi / 4)


def test_head_direction_reverse_315():
    x1 = np.linspace(0.01, 1, 10)
    y1 = x1[::-1]
    x2 = x1 + 0.01  # 1cm between
    y2 = y1 + 0.01
    t = x1
    a, t = head_direction(x1, y1, x2, y2, t)
    assert np.allclose(a, 2 * np.pi - np.pi / 4)


def test_head_rate():
    x1 = np.linspace(0.01, 1, 10)
    y1 = x1
    x2 = x1 + 0.01  # 1cm between
    y2 = x1 - 0.01
    t = np.linspace(0, 1, 10)
    a, t = head_direction(x1, y1, x2, y2, t)
    sptr = np.linspace(0, 1, 100)
    bins, rate = head_direction_rate(sptr, a, t, n_bins=8, avg_window=1)
    assert bins[1] == np.pi / 4
    assert abs(rate[1] - 100) < 0.5


def test_head_score():
    x1 = np.linspace(0.01, 1, 10)
    y1 = x1
    x2 = x1 + 0.01  # 1cm between
    y2 = x1 - 0.01
    t = np.linspace(0, 1, 10)
    a, t = head_direction(x1, y1, x2, y2, t)
    sptr = np.linspace(0, 1, 100)
    bins, rate = head_direction_rate(sptr, a, t, n_bins=100, avg_window=2)
    ang, score = head_direction_score(bins, rate)
    assert abs(score - 1) < 0.001
    assert abs(ang - np.pi / 4) < 0.00001
