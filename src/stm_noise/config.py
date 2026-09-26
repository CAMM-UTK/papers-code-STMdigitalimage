"""Load and validate the YAML configuration file used to parameterize
the noise simulation pipeline."""

from pathlib import Path
import yaml

REQUIRED_TOP_LEVEL_KEYS = ["noise", "io"]
REQUIRED_NOISE_KEYS = ["stripes", "flicker", "gaussian", "paraboloid", "row_drift"]


def load_config(path: str) -> dict:
    """Load and validate a YAML config file.

    Parameters
    ----------
    path : str
        Path to a YAML config file.

    Returns
    -------
    dict
        Parsed and validated configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in config:
            raise KeyError(f"Config must contain a '{key}' section")

    for key in REQUIRED_NOISE_KEYS:
        if key not in config["noise"]:
            raise KeyError(f"Config['noise'] must contain a '{key}' section")

    for key in ["clean_dir", "noisy_dir"]:
        if key not in config["io"]:
            raise KeyError(f"Config['io'] must contain a '{key}' key")
