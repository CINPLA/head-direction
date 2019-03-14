import pytest
import numpy as np
import math

def test_head_direction_45():
    from head_direction.head import head_direction
    x1 = np.linspace(.01,1,10)
    y1 = x1
    x2 = x1 + .01 # 1cm between
    y2 = x1 - .01
    t = x1
    a, t = head_direction(x1, y1, x2, y2, t)
    assert np.allclose(a, np.pi / 4)


def test_head_direction_135():
    from head_direction.head import head_direction
    x1 = np.linspace(.01,1,10)[::-1]
    y1 = x1[::-1]
    x2 = x1 - .01 # 1cm between
    y2 = y1 - .01
    t = x1
    a, t = head_direction(x1, y1, x2, y2, t)
    assert np.allclose(a, np.pi - np.pi / 4)


def test_head_direction_225():
    from head_direction.head import head_direction
    x1 = np.linspace(.01,1,10)[::-1]
    y1 = x1
    x2 = x1 - .01 # 1cm between
    y2 = y1 + .01
    t = x1
    a, t = head_direction(x1, y1, x2, y2, t)
    assert np.allclose(a, np.pi + np.pi / 4)


def test_head_direction_reverse_315():
    from head_direction.head import head_direction
    x1 = np.linspace(.01,1,10)
    y1 = x1[::-1]
    x2 = x1 + .01 # 1cm between
    y2 = y1 + .01
    t = x1
    a, t = head_direction(x1, y1, x2, y2, t)
    assert np.allclose(a, 2 * np.pi - np.pi / 4)
