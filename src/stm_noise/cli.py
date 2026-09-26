import argparse

from .config import load_config
from .simulate import batch_add_noise


def main():
    parser = argparse.ArgumentParser(
        description="Simulate realistic noise on clean STM images"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    saved_paths = batch_add_noise(config)
    print(f"Saved {len(saved_paths)} noisy images to '{config['io']['noisy_dir']}'")


if __name__ == "__main__":
    main()
