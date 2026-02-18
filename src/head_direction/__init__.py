#!/usr/bin/env python
#
# Copyright (C) 2024
#
# This file is part of head_direction
# SPDX-License-Identifier:    GPLv3

import importlib.metadata

from .core import head_direction, head_direction_rate, head_direction_score
from .type_alias import ArrayLike, SpikeInput
from .utils import get_alignment_offset

__version__ = importlib.metadata.version("head_direction")

__all__ = [
    "ArrayLike",
    "SpikeInput",
    "get_alignment_offset",
    "head_direction",
    "head_direction_rate",
    "head_direction_score",
]
