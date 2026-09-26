"""
Individual noise-generation building blocks used to simulate realistic
STM image artifacts:

- add_row_drift_to_image : row-by-row scan drift
- gaussian_noise         : white Gaussian noise
- add_paraboloid         : smooth paraboloid/saddle background artifact
- generate_1d_flicker / generate_stm_flicker_image : 1/f (flicker) noise
- generate_stripes       : uneven, frequency-modulated stripe lines
- scale_noise            : zero-mean / unit-std normalization
- dynamic_norm           : scale noise relative to image dynamic range
"""

import numpy as np


def add_row_drift_to_image(
    STM_image, max_shift_px=3, mode="wrap", smooth=False, drift_step_std=0.25
):
    """
    Add row drift directly to an existing STM image by shifting each row.

    Parameters
    ----------
    STM_image : 2D numpy array
        Input STM image.
    max_shift_px : int
        Maximum absolute shift (in pixels) for any row.
    mode : str
        'zero' -> shifted-in pixels are filled with 0
        'edge' -> shifted-in pixels are filled with nearest edge value
        'wrap' -> circular shift (np.roll)
    smooth : bool
        If True, use smooth drift (random walk, more realistic STM drift).
        If False, use independent random shift per row.
    drift_step_std : float
        Step size for smooth drift random walk.

    Returns
    -------
    drifted_image : 2D numpy array
        STM image with row drift added.
    """
    STM_image = np.asarray(STM_image, dtype=float)
    pxls = STM_image.shape[0]
    drifted_image = np.zeros_like(STM_image)

    if smooth:
        offsets = np.zeros(pxls, dtype=float)
        for i in range(1, pxls):
            offsets[i] = offsets[i - 1] + np.random.normal(0, drift_step_std)
        offsets = np.clip(offsets, -max_shift_px, max_shift_px)
        offsets = np.round(offsets).astype(int)
    else:
        offsets = np.random.randint(-max_shift_px, max_shift_px + 1, size=pxls)

    for i in range(pxls):
        shift = offsets[i]
        row = STM_image[i]

        if mode == "wrap":
            drifted_image[i] = np.roll(row, shift)

        elif mode == "zero":
            new_row = np.zeros_like(row)
            if shift > 0:
                new_row[shift:] = row[:-shift]
            elif shift < 0:
                new_row[:shift] = row[-shift:]
            else:
                new_row[:] = row
            drifted_image[i] = new_row

        elif mode == "edge":
            new_row = np.empty_like(row)
            if shift > 0:
                new_row[:shift] = row[0]
                new_row[shift:] = row[:-shift]
            elif shift < 0:
                new_row[shift:] = row[-1]
                new_row[:shift] = row[-shift:]
            else:
                new_row[:] = row
            drifted_image[i] = new_row

        else:
            raise ValueError("mode must be 'zero', 'edge', or 'wrap'")

    return drifted_image


def gaussian_noise(STM_image, noise_amplitude):
    """Generate white Gaussian noise scaled to the normalized image range."""
    max_ = np.max(STM_image)
    min_ = np.min(STM_image)

    if max_ > min_:
        scaled_image = (STM_image - min_) / (max_ - min_)
    else:
        scaled_image = np.zeros_like(STM_image)

    noise = np.random.normal(0, noise_amplitude, scaled_image.shape)
    return noise


def add_paraboloid(STM_image, x_c_frac, y_c_frac, hyper, inv, a_b_ratio, amp):
    """
    x_c_frac, y_c_frac: center position as a FRACTION of image size (0 to 1),
                         instead of physical nm coordinates.
    """
    pixels = STM_image.shape[0]
    x = np.linspace(-1, 1, pixels)   # normalized coordinates, no physical units needed
    y = x
    xx, yy = np.meshgrid(x, y)

    x_c = 2 * x_c_frac - 1   # convert fraction [0,1] to normalized [-1,1]
    y_c = 2 * y_c_frac - 1

    a = 1
    b = a * a_b_ratio
    z = (-1) ** inv * (((xx - x_c) / a) ** 2 + (-1) ** hyper * ((yy - y_c) / b) ** 2)

    return  amp * z


def generate_1d_flicker(N, alpha=1):
    """Generate 1D flicker (1/f^alpha) noise using the FFT method."""
    freqs = np.fft.rfftfreq(N)
    freqs[0] = freqs[1]  # avoid division by zero

    random_complex = np.random.normal(size=len(freqs)) + 1j * np.random.normal(
        size=len(freqs)
    )

    scaling = 1 / (freqs ** (alpha * 0.5))
    spectrum = random_complex * scaling

    noise = np.fft.irfft(spectrum, n=N)
    noise = (noise - np.mean(noise)) / np.std(noise)

    return noise


def generate_stm_flicker_image(N=128, alpha=2.04):
    """
    Generate STM-like 2D flicker noise using temporal 1/f noise, simulating
    a forward scan with a discarded back-scan.

    Parameters
    ----------
    N : int
        Image size (e.g. 64, 128, 256, 512).
    alpha : float
        Exponent in 1/(f^alpha); alpha=2.04 was fit from prior experimental data.
    """
    total_points = N * N * 2  # include back-scan

    time_noise = generate_1d_flicker(total_points, alpha)

    image = np.zeros((N, N))

    idx = 0
    for row in range(N):
        image[row, :] = time_noise[idx : idx + N]
        idx += N
        idx += N  # discard back-scan

    image = (image - np.mean(image)) / np.std(image)

    return image


def generate_stripes(image, f0=4):
    """
    Generate unevenly distributed stripe lines.

    Parameters
    ----------
    image : 2D numpy array
        Used only for its shape.
    f0 : float
        Frequency around which the stripe lines oscillate (typical range 3-10).
    """
    width, height = image.shape

    x = np.linspace(0, 20, width)
    y = np.linspace(0, 20, height)

    X, Y = np.meshgrid(x, y)

    dt = y[1] - y[0]

    # frequency variation
    rv = np.random.randn()
    f = f0 + 1.5 * np.sin(rv * y)

    phase = 2 * np.pi * np.cumsum(f) * dt

    phase2D = phase[:, None] + X * 2

    # random amplitude
    noise_A = np.random.randn(width)
    # kernel size must not exceed the image width, otherwise
    # np.convolve(..., mode="same") returns an array longer than `width`
    kernel_size = max(1, min(100, width))
    smooth_A = np.convolve(noise_A, np.ones(kernel_size) / (kernel_size / 2), mode="same")
    A = 1 + 2 * smooth_A

    A2D = np.tile(A[:, None], (1, width))

    img = A2D * np.sin(phase2D)

    # normalize
    img = (img - img.min()) / (img.max() - img.min())

    return img


def scale_noise(noise):
    """
    Center noise to zero mean and normalize to unit std. This standardizes
    all noise types to the same scale before dynamic_norm sets the final
    amplitude.

    Parameters
    ----------
    noise : 2D numpy array

    Returns
    -------
    noise_scaled : zero-mean, unit-std noise array
    """
    noise = noise.astype(np.float32)

    mean = noise.mean()
    std = noise.std()

    if std < 1e-8:
        return np.zeros_like(noise)

    noise_scaled = (noise - mean) / std

    return noise_scaled


def dynamic_norm(image, noise, sigma=0.05):
    """
    Scale noise to a fraction of the clean image's dynamic range.

    The noise amplitude is set to: sigma * (image.max() - image.min())
    so noise is always proportional to the signal strength.

    Parameters
    ----------
    image : 2D numpy array
        The clean image.
    noise : 2D numpy array
        The noise (should be zero-mean, unit-std after scale_noise).
    sigma : float
        Noise strength as a fraction of image dynamic range.
        sigma=0    -> no noise added
        sigma=0.05 -> noise std = 5% of image range
        sigma=0.20 -> noise std = 20% of image range

    Returns
    -------
    noise_scaled : noise scaled to match image dynamic range
    """
    if sigma == 0:
        return np.zeros_like(noise, dtype=np.float32)

    image = image.astype(np.float32)
    noise = noise.astype(np.float32)

    img_min = image.min()
    img_max = image.max()
    img_range = img_max - img_min

    if img_range < 1e-8:
        return np.zeros_like(noise, dtype=np.float32)

    noise_std = noise.std() + 1e-8
    noise_scaled = noise * (sigma * img_range / noise_std)

    return noise_scaled.astype(np.float32)
