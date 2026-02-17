# Head Direction Analysis

A Python package for the analysis of head direction cells. It provides robust tools for reconstructing head angles from tracking data, computing firing rate tuning curves, and quantifying directionality using Rayleigh vector analysis.

## Key Features

* **Robust angle reconstruction**: Calculates head direction from two tracked LEDs with artifact rejection.
* **Auto-calibration**: Automatically detects and corrects for unknown LED orientations (e.g., side-mounted ears vs. front/back) using velocity vector alignment.
* **Duck typing support**: Works seamlessly with NumPy arrays, lists, [Neo](https://neuralensemble.org/neo/) `SpikeTrain` objects, and [Quantities](https://python-quantities.readthedocs.io/).
* **Statistical scoring**: Computes Mean Vector Length and Mean Angle to quantify tuning strength.

## Installation

### For Users
You can install the package directly using `pip`:

```bash
pip install head_direction
```


## Quick Start

### 1. Calculate head direction
Reconstruct the head angle from tracking data (Front and Back LEDs).

```python
import numpy as np
from head_direction import head_direction

# Load your tracking data (x, y positions)
# t, led1_x, led1_y, led2_x, led2_y = load_data(...)

# Calculate angles (assumes LED1=Front, LED2=Back)
angles, clean_times = head_direction(
    led1_x, led1_y,
    led2_x, led2_y,
    time_stamps=t,
    std_filter_threshold=3.0
)
```

### 2. Compute tuning curves
Calculate the firing rate as a function of head direction.

```python
from head_direction import head_direction_rate

# Spike times can be a list, numpy array, or Neo SpikeTrain
spike_times = [0.1, 0.5, 0.9, ...]

bin_centers, firing_rate = head_direction_rate(
    spike_times,
    angles,
    clean_times,
    num_bins=36,       # 10-degree bins
    smoothing_window=4 # Boxcar smoothing
)
```

### 3. Score directionality
Quantify how strongly the cell is tuned to a specific direction.

```python
from head_direction import head_direction_score

mean_angle, vector_length = head_direction_score(bin_centers, firing_rate)

print(f"Preferred Direction: {np.degrees(mean_angle):.1f}°")
print(f"Vector Length: {vector_length:.4f}") # 0.0 (uniform) to 1.0 (perfectly directional)
```

## Handling Unknown LED Orientations

If your LEDs are not mounted in the standard Front-Back configuration (e.g., they are on the ears), you can use the auto-calibration tool to find the correct offset.

```python
from head_direction.utils import get_alignment_offset

# Calculate raw angles (will be rotated by some unknown offset)
raw_angles, raw_times = head_direction(led1_x, led1_y, led2_x, led2_y, t)

# Detect the offset by comparing Head Angle vs. Movement Direction
offset = get_alignment_offset(
    raw_angles,
    pos_x=led1_x,
    pos_y=led1_y,
    time_stamps=t
)

# Recalculate with the correction
final_angles, _ = head_direction(
    led1_x, led1_y,
    led2_x, led2_y,
    t,
    offset=offset
)
```

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating full analysis pipelines:

* `demo_analysis.ipynb`: A complete walkthrough generating synthetic data, calculating tuning curves, and visualizing the results.

## Contributing

Please see [DEV.md](DEV.md) for guidelines on setting up the development environment, running tests, and using `pre-commit`.

## Authors

* Mikkel Lepperød
* Nicolai Haug
* Alessio Buccino

## License

GNU General Public License v3 (GPLv3)
