import pytest
from pathlib import Path

from stm_noise.config import load_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default_config.yaml"


def test_load_default_config():
    config = load_config(str(DEFAULT_CONFIG_PATH))
    assert "noise" in config
    assert "io" in config
    for key in ["stripes", "flicker", "gaussian", "paraboloid", "row_drift"]:
        assert key in config["noise"]


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")


def test_load_config_missing_noise_section(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("io:\n  clean_dir: a\n  noisy_dir: b\n")
    with pytest.raises(KeyError):
        load_config(str(bad_config))
