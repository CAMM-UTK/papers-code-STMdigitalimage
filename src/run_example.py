"""
Example: apply the noise pipeline to a single clean STM image and plot
before/after side by side.

Run from the repo root after installing the package:
    python examples/run_example.py
"""

import numpy as np
import matplotlib.pyplot as plt

from stm_noise.config import load_config
from stm_noise.simulate import simulate_noise

# 1. Load config (adjust path if running from elsewhere)
config = load_config("configs/default_config.yaml")

# 2. Load (or fake) a clean image
try:
    clean_image = np.load("data/clean_images/example.npy")
except FileNotFoundError:
    print("No example clean image found — generating a synthetic one instead.")
    x = np.linspace(-3, 3, 128)
    xx, yy = np.meshgrid(x, x)
    clean_image = np.sin(xx) * np.cos(yy)

# 3. Apply the noise pipeline
noisy_image = simulate_noise(clean_image, config)

# 4. Visualize
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(clean_image, cmap="gray")
axes[0].set_title("Clean")
axes[0].axis("off")

axes[1].imshow(noisy_image, cmap="gray")
axes[1].set_title("Noisy (simulated)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("example_output.png", dpi=150)
print("Saved example_output.png")
