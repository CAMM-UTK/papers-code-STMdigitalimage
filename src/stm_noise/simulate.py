"""
Main noise-simulation pipeline: combines stripes, 1/f flicker noise,
Gaussian noise, a paraboloid background artifact, and row drift into a
single realistic noisy STM image, driven entirely by a config dict.
"""

import random
import numpy as np

from .noise_functions import (
    add_row_drift_to_image,
    gaussian_noise,
    add_paraboloid,
    generate_stm_flicker_image,
    generate_stripes,
    scale_noise,
    dynamic_norm,
)


def simulate_noise(image: np.ndarray, config: dict) -> np.ndarray:
    """
    Apply the full noise pipeline (stripes + flicker + gaussian + paraboloid
    background + row drift) to a clean 2D image, using parameter ranges
    defined in `config['noise']`.

    Parameters
    ----------
    image : 2D numpy array
        Clean input image.
    config : dict
        Parsed configuration (see configs/default_config.yaml).

    Returns
    -------
    final_image : 2D numpy array
        Noisy image with row drift applied.
    """
    cfg = config["noise"]
    size = image.shape[0]

    # ---- stripes (sum of multiple frequency components) ----
    s_cfg = cfg["stripes"]
    num_stripes = np.random.randint(s_cfg["num_stripes_range"][0], s_cfg["num_stripes_range"][1])
    stripes = np.zeros_like(image, dtype=np.float32)
    for _ in range(num_stripes):
        f0 = np.random.uniform(*s_cfg["freq_range"])
        stripe = generate_stripes(image, f0)
        amp = np.random.uniform(*s_cfg["amp_range"])
        stripes += amp * stripe

    # ---- flicker noise ----
    f_cfg = cfg["flicker"]
    alpha = np.random.uniform(*f_cfg["alpha_range"])
    flicker = generate_stm_flicker_image(size, alpha)

    # ---- gaussian noise ----
    g_cfg = cfg["gaussian"]
    noise_amplitude = np.random.uniform(*g_cfg["amplitude_range"])
    gaussian = gaussian_noise(image, noise_amplitude)

    # ---- paraboloid background ----
    p_cfg = cfg["paraboloid"]

    x_c = np.random.uniform(-size / 2, size / 2)
    y_c = np.random.uniform(-size / 2, size / 2)
    hyper = np.random.randint(0, 2)
    inv = np.random.randint(0, 2)
    a_b_ratio_max = p_cfg["a_b_ratio_max"]
    a_b_ratio = np.random.uniform(1 / a_b_ratio_max, a_b_ratio_max)
    amp_max = p_cfg["amp_max"]
    amp = np.random.uniform(0, amp_max)
    background = add_paraboloid(image, x_c, y_c, hyper, inv, a_b_ratio, amp)


    # ---- standardize all noise types to zero-mean/unit-std ----
    stripes = scale_noise(stripes)
    flicker = scale_noise(flicker)
    gaussian = scale_noise(gaussian)
    background = scale_noise(background)

    # ---- randomly sample noise strength (sigma) per noise type ----
    r1 = random.choice(s_cfg["sigma_choices"])
    r2 = random.choice(f_cfg["sigma_choices"])
    r3 = random.choice(g_cfg["sigma_choices"])
    r4 = random.choice(p_cfg["sigma_choices"])

    stripes_norm = dynamic_norm(image, stripes, sigma=r1)
    flicker_norm = dynamic_norm(image, flicker, sigma=r2)
    gaussian_norm = dynamic_norm(image, gaussian, sigma=r3)
    background_norm = dynamic_norm(image, background, sigma=r4)

    noisy_image = image + stripes_norm + flicker_norm + gaussian_norm + background_norm

    # ---- row drift ----
    rd_cfg = cfg["row_drift"]
    final_image = add_row_drift_to_image(
        noisy_image,
        max_shift_px=rd_cfg["max_shift_px"],
        mode=rd_cfg["mode"],
        smooth=rd_cfg["smooth"],
        drift_step_std=rd_cfg["drift_step_std"],
    )

    return final_image


def add_noise(image_path, config: dict) -> np.ndarray:
    """
    Load a clean image (.npy) from disk and apply the full noise pipeline.

    Parameters
    ----------
    image_path : str or Path
        Path to a clean .npy image.
    config : dict
        Parsed configuration.

    Returns
    -------
    final_image : 2D numpy array
    """
    image = np.load(image_path)
    return simulate_noise(image, config)


def batch_add_noise(config: dict) -> list:
    """
    Process every clean image in `config['io']['clean_dir']`, apply noise,
    and save the result into `config['io']['noisy_dir']` with the same
    filename.

    Returns
    -------
    list of Path
        Paths of the saved noisy images.
    """
    from pathlib import Path

    io_cfg = config["io"]
    clean_folder = Path(io_cfg["clean_dir"])
    noisy_folder = Path(io_cfg["noisy_dir"])
    noisy_folder.mkdir(parents=True, exist_ok=True)

    ext = io_cfg.get("file_ext", ".npy")

    if config.get("seed") is not None:
        np.random.seed(config["seed"])
        random.seed(config["seed"])

    all_clean = sorted(
        p for p in clean_folder.iterdir() if p.is_file() and p.suffix.lower() == ext
    )

    saved_paths = []
    for image_path in all_clean:
        filename = image_path.name
        noisy = add_noise(image_path, config)
        out_path = noisy_folder / filename
        np.save(out_path, noisy)
        saved_paths.append(out_path)

    return saved_paths
