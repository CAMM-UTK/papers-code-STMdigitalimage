import numpy as np
import pytest

from stm_noise.noise_functions import (
    add_row_drift_to_image,
    gaussian_noise,
    add_paraboloid,
    generate_1d_flicker,
    generate_stm_flicker_image,
    generate_stripes,
    scale_noise,
    dynamic_norm,
)


@pytest.fixture
def clean_image():
    rng = np.random.default_rng(0)
    return rng.normal(size=(256,256)).astype(np.float32)


def test_row_drift_shape_preserved(clean_image):
    out = add_row_drift_to_image(clean_image, max_shift_px=3, mode="wrap")
    assert out.shape == clean_image.shape


def test_row_drift_modes(clean_image):
    for mode in ["wrap", "zero", "edge"]:
        out = add_row_drift_to_image(clean_image, max_shift_px=2, mode=mode)
        assert out.shape == clean_image.shape
        assert np.isfinite(out).all()


def test_row_drift_invalid_mode(clean_image):
    with pytest.raises(ValueError):
        add_row_drift_to_image(clean_image, mode="bogus")


def test_gaussian_noise_shape(clean_image):
    noise = gaussian_noise(clean_image, 0.05)
    assert noise.shape == clean_image.shape


def test_add_paraboloid_shape(clean_image):
    bg = add_paraboloid(clean_image, x_c=0, y_c=0,
                         hyper=0, inv=0, a_b_ratio=1.0, amp=0.1)
    assert bg.shape == clean_image.shape


def test_generate_1d_flicker_normalized():
    noise = generate_1d_flicker(256, alpha=2.0)
    assert noise.shape == (256,)
    assert abs(noise.mean()) < 1e-6
    assert abs(noise.std() - 1.0) < 1e-6


def test_generate_stm_flicker_image_shape():
    img = generate_stm_flicker_image(N=32, alpha=2.04)
    assert img.shape == (32, 32)


def test_generate_stripes_range(clean_image):
    img = generate_stripes(clean_image, f0=4)
    assert img.shape == clean_image.shape
    assert img.min() >= 0.0 - 1e-6
    assert img.max() <= 1.0 + 1e-6


def test_scale_noise_zero_mean_unit_std(clean_image):
    scaled = scale_noise(clean_image)
    assert abs(scaled.mean()) < 1e-4
    assert abs(scaled.std() - 1.0) < 1e-4


def test_scale_noise_flat_input():
    flat = np.ones((10, 10))
    out = scale_noise(flat)
    assert np.all(out == 0)


def test_dynamic_norm_sigma_zero(clean_image):
    noise = scale_noise(np.random.normal(size=clean_image.shape))
    out = dynamic_norm(clean_image, noise, sigma=0)
    assert np.all(out == 0)


def test_dynamic_norm_scales_relative_to_image_range(clean_image):
    noise = scale_noise(np.random.normal(size=clean_image.shape))
    sigma = 0.1
    out = dynamic_norm(clean_image, noise, sigma=sigma)
    img_range = clean_image.max() - clean_image.min()
    assert abs(out.std() - sigma * img_range) < 1e-2
