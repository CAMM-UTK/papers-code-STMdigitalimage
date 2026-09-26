# Changelog

## [0.1.0] - 2026-09-22
### Added
- Initial package structure converted from prototype notebook.
- Noise functions: row drift, Gaussian noise, paraboloid background,
  1/f flicker noise, stripe noise, scale_noise, dynamic_norm.
- `simulate_noise`, `add_noise`, `batch_add_noise` pipeline functions.
- YAML-based configuration with validation (`config.py`).
- CLI entry point (`stm-noise --config configs/default_config.yaml`).
- Unit tests for noise functions and config loading.
- Example script (`examples/run_example.py`).
