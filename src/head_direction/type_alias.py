from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import neo
    import quantities as pq

# ArrayLike can be a numpy array, a list of numbers, or a tuple of numbers
type ArrayLike = np.ndarray | list[float] | tuple[float, ...]

# SpikeInput can be:
# 1. A plain numpy array
# 2. A Neo SpikeTrain (if neo is installed)
# 3. A Quantities array (if quantities is installed)
# 4. A list/iterable of Neo SpikeTrains
type SpikeInput = np.ndarray | "neo.SpikeTrain" | "pq.Quantity" | list["neo.SpikeTrain"] | list["pq.Quantity"]
