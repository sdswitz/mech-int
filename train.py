"""Backward-compatible wrapper around the refactored SAE training script."""

import sys

from scripts.train_sae import main


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--legacy-layout" not in argv:
        argv = ["--legacy-layout", *argv]
    main(argv)
