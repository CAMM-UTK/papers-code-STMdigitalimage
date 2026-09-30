# stm_noise

Simulate realistic noise on clean STM (Scanning Tunneling Microscopy) images.

## Overview

![Pipeline overview](https://raw.githubusercontent.com/HuanhuanZhao08/stm_noise/main/docs/images/fig4.png)


*The toolkit generates physics-informed synthetic STM noise that can be combined with any simulated STM image 
to form clean/noisy training pairs for denoising models. Models trained this way effectively denoise experimental
 STM images while preserving true physical features*

For full methodology and quantitative validation, see our paper: **A Digital Simulation Toolkit for Physics-Based Generation of Realistic
  Experimental Scanning Tunneling Microscopy Images**/[Link](http://arxiv.org/abs/2609.36639)



The pipeline combines five artifact types commonly seen in experimental STM data:

- **Stripe noise** — uneven, frequency-modulated scan-line stripes
- **1/f flicker noise** — temporal 1/f^alpha noise simulating forward-scan drift
- **Gaussian noise** — white noise
- **Paraboloid background** — smooth large-scale bump/tilt artifact
- **Row drift** — per-row pixel shift simulating scanner drift

All randomization ranges are controlled from a single YAML config file, so you
can tune noise strength/statistics without touching any code.

## Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/HuanhuanZhao08/stm_noise
cd stm_noise
pip install -e .
```

Or create the conda environment first:

```bash
conda env create -f environment.yml
conda activate stm-noise
pip install -e .
```

## Usage

### 1. Command line (batch processing)

Point the config at your clean/noisy image folders (see `configs/default_config.yaml`),
then run:

```bash
stm-noise --config configs/default_config.yaml
```

This reads every `.npy` clean image in `io.clean_dir`, applies the full noise
pipeline, and saves the result (same filename) into `io.noisy_dir`.

### 2. Python API

```python
import numpy as np
from stm_noise.config import load_config
from stm_noise.simulate import simulate_noise, add_noise, batch_add_noise

config = load_config("configs/default_config.yaml")

# Apply noise to an in-memory clean image (numpy array)
clean_image = np.load("data/clean_images/example.npy")
noisy_image = simulate_noise(clean_image, config)

# Or apply noise directly from a file path
noisy_image = add_noise("data/clean_images/example.npy", config)

# Or process an entire folder (same as the CLI)
saved_paths = batch_add_noise(config)
```

See `examples/run_example.py` and `examples/run_example.ipynb` for a full runnable example with plotting.

### 3. Individual noise functions

Every noise component is also available standalone if you want finer control:

```python
from stm_noise.noise_functions import (
    add_row_drift_to_image,
    gaussian_noise,
    add_paraboloid,
    generate_stm_flicker_image,
    generate_stripes,
    scale_noise,
    dynamic_norm,
)
```

## Configuration

All tunable parameters live in `configs/default_config.yaml`:

```yaml
noise:
  stripes:
    num_stripes_range: [1, 5]
    freq_range: [2, 10]
    amp_range: [0.1, 0.5]
    sigma_choices: [0, 0.05, 0.1, 0.15, 0.2]
  flicker:
    alpha_range: [2, 4]
    sigma_choices: [0, 0.01, ..., 0.1]
  gaussian:
    amplitude_range: [0.01, 0.03]
    sigma_choices: [0, 0.01, ..., 0.1]
  paraboloid:
    l_stm_range: [1, 4]
    a_b_ratio_max: 3
    amp_max: 2
    sigma_choices: [0, 0.01, ..., 0.05]
  row_drift:
    max_shift_px: 3
    mode: "wrap"        # 'wrap' | 'zero' | 'edge'
    smooth: false
    drift_step_std: 0.25

io:
  clean_dir: "Denoise_graphene/train/clean"
  noisy_dir: "Denoise_graphene/train/noisy"
  file_ext: ".npy"

seed: null   # set an integer for reproducible batches
```

Copy `configs/default_config.yaml` and create your own variant (e.g.
`configs/my_experiment.yaml`) rather than editing the default in place.

## Project layout

```
stm_noise/
├── pyproject.toml
├── environment.yml
├── README.md
├── configs/
│   └── default_config.yaml
├── examples/
│   └── run_example.py
├── src/
│   └── stm_noise/
│       ├── __init__.py
│       ├── __about__.py
│       ├── config.py
│       ├── noise_functions.py
│       ├── simulate.py
│       └── cli.py
└── tests/
    ├── test_noise_functions.py
    └── test_config.py
```

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT (see `LICENSE`).
