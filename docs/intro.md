# Head Direction Analysis

**`head_direction`** is a Python package for the analysis of head direction cells. It provides robust tools for reconstructing head angles from tracking data, computing firing rate tuning curves, and quantifying directionality using Rayleigh vector analysis.

## Key Features

* **Robust angle reconstruction**: Calculates head direction from two tracked LEDs with artifact rejection.
* **Auto-calibration**: Automatically detects and corrects for unknown LED orientations (e.g., side-mounted ears vs. front/back) using velocity vector alignment.
* **Duck typing support**: Works seamlessly with NumPy arrays, lists, [Neo](https://neuralensemble.org/neo/) `SpikeTrain` objects, and [Quantities](https://python-quantities.readthedocs.io/).
* **Statistical scoring**: Computes Mean Vector Length and Mean Angle to quantify tuning strength.

## Installation

```bash
pip install head_direction
```

## Quick Start

Check out the [Getting Started](tutorials/getting_started.ipynb) tutorial for a complete walkthrough.
