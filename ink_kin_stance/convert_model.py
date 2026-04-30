"""
Convert a trained diffusion model checkpoint to be compatible with Isaac Sim's numpy.

Converts numpy arrays in the checkpoint to lists, which are portable across
different numpy versions. Also saves with pickle protocol 4 for broader compat.

Usage:
    python -m ink_kin_stance.convert_model <input_model.pt> <output_model_compat.pt>

Example:
    python -m ink_kin_stance.convert_model still/diffusion_model.pt still/diffusion_model_compat.pt
"""

import sys
import numpy as np

# Numpy compatibility fix for models saved with newer numpy
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core

import torch


KEYS_TO_CONVERT = ["traj_mean", "traj_std", "delta_mean", "delta_std", "pos_mean", "pos_std"]


def convert_checkpoint(input_path: str, output_path: str):
    """Load checkpoint and re-save with numpy arrays converted to lists."""
    print(f"Loading {input_path}...")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    print("Converting numpy arrays to lists for compatibility...")
    for key in KEYS_TO_CONVERT:
        if key in checkpoint:
            value = checkpoint[key]
            if isinstance(value, np.ndarray):
                checkpoint[key] = value.tolist()
                print(f"  Converted {key}: {type(value).__name__} -> list")

    print(f"Saving to {output_path}...")
    torch.save(checkpoint, output_path, pickle_protocol=4)

    print("Done!")
    print(f"\nCheckpoint contents:")
    for key, value in checkpoint.items():
        if key == "model_state_dict":
            print(f"  {key}: <state_dict with {len(value)} keys>")
        elif isinstance(value, list):
            print(f"  {key}: list({len(value)} elements)")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    convert_checkpoint(sys.argv[1], sys.argv[2])
