#!/usr/bin/env python3
"""Train the step (leg) diffusion model."""

import argparse
from ink_kin_stance.diffusion.training import train


def main():
    parser = argparse.ArgumentParser(description="Train step diffusion model")
    parser.add_argument("--zarr-path", default="step/trajectory_log.zarr")
    parser.add_argument("--save-path", default="step/diffusion_model.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-timesteps", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-dir", default="step/runs/diffusion")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--time-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=6)
    args = parser.parse_args()

    train(
        zarr_path=args.zarr_path,
        save_path=args.save_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_timesteps=args.num_timesteps,
        device=args.device,
        log_dir=args.log_dir,
        hidden_dim=args.hidden_dim,
        time_dim=args.time_dim,
        num_blocks=args.num_blocks,
        eta_min_factor=100,
    )


if __name__ == "__main__":
    main()
