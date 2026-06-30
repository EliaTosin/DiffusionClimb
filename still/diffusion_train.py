import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import zarr
import math
from ink_kin_stance.diffusion.model import ConditionalDiffusionModel, ConditionalDropOutDiffusionModel
from ink_kin_stance.diffusion.gaussian_diffusion import GaussianDiffusion
# Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, start_pos, goal_pos, trajectories, start_rpy=None, goal_rpy=None, norm_stats=None):
        """
        Initialize dataset with pre-split data.
        Condition: delta (pos+rpy, 6D) + current joint values (first timestep)
        norm_stats: dict with traj_mean, traj_std, delta_mean, delta_std (from training set)
        """
        self.trajectories = trajectories
        delta_pos = goal_pos - start_pos  # (N, 3)
        if start_rpy is not None and goal_rpy is not None:
            delta_rpy = goal_rpy - start_rpy  # (N, 3)
            self.delta = np.concatenate([delta_pos, delta_rpy], axis=1)  # (N, 6)
        else:
            self.delta = delta_pos  # (N, 3) fallback
        self.current_joints = trajectories[:, 0, :]  # (N, num_joints) - first timestep

        if norm_stats is not None:
            # Use provided normalization stats (for val/test)
            self.traj_mean = norm_stats["traj_mean"]
            self.traj_std = norm_stats["traj_std"]
            self.delta_mean = norm_stats["delta_mean"]
            self.delta_std = norm_stats["delta_std"]
        else:
            # Compute per-joint normalization stats (for train)
            self.traj_mean = self.trajectories.mean(axis=(0, 1))  # shape: (num_joints,)
            self.traj_std = self.trajectories.std(axis=(0, 1))    # shape: (num_joints,)
            self.traj_std = np.where(self.traj_std < 1e-6, 1.0, self.traj_std)
            # Delta normalization (pos + rpy)
            self.delta_mean = self.delta.mean(axis=0)
            self.delta_std = self.delta.std(axis=0)
            self.delta_std = np.where(self.delta_std < 1e-6, 1.0, self.delta_std)

        # Apply normalization
        self.trajectories = (self.trajectories - self.traj_mean) / self.traj_std
        self.delta = (self.delta - self.delta_mean) / self.delta_std
        # Current joints use same normalization as trajectories
        self.current_joints = (self.current_joints - self.traj_mean) / self.traj_std

    def get_norm_stats(self):
        return {
            "traj_mean": self.traj_mean,
            "traj_std": self.traj_std,
            "delta_mean": self.delta_mean,
            "delta_std": self.delta_std,
        }

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return {
            "delta": torch.tensor(self.delta[idx]),
            "current_joints": torch.tensor(self.current_joints[idx]),
            "trajectory": torch.tensor(self.trajectories[idx]),
        }


def load_and_split_data(zarr_path, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Load zarr and split into train/val/test datasets.
    Returns: train_dataset, val_dataset, test_dataset, num_steps, num_joints
    """
    root = zarr.open_group(zarr_path, mode='r')
    print(f"Loading data from {zarr_path}")

    # Load data from zarr arrays
    start_pos = np.array(root["start_positions"][:], dtype=np.float32)
    goal_pos = np.array(root["goal_positions"][:], dtype=np.float32)
    trajectories = np.array(root["joint_angles"][:], dtype=np.float32)
    start_rpy = np.array(root["start_rpys"][:], dtype=np.float32) if "start_rpys" in root else None
    goal_rpy = np.array(root["goal_rpys"][:], dtype=np.float32) if "goal_rpys" in root else None

    # Infer trajectory shape from data
    num_steps = trajectories.shape[1]
    num_joints = trajectories.shape[2]
    print(f"Trajectory shape: {trajectories.shape} (num_steps={num_steps}, num_joints={num_joints})")

    print(f"Loaded {len(trajectories)} trajectories")

    # Check for NaN/Inf
    if np.any(np.isnan(trajectories)) or np.any(np.isinf(trajectories)):
        print("WARNING: NaN or Inf found in trajectories, replacing with 0")
        trajectories = np.nan_to_num(trajectories, nan=0.0, posinf=0.0, neginf=0.0)

    # Shuffle indices
    n = len(trajectories)
    np.random.seed(seed)
    indices = np.random.permutation(n)

    # Split indices
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Create train dataset (computes normalization stats)
    s_rpy = lambda idx: start_rpy[idx] if start_rpy is not None else None
    g_rpy = lambda idx: goal_rpy[idx] if goal_rpy is not None else None
    train_dataset = TrajectoryDataset(
        start_pos[train_idx], goal_pos[train_idx], trajectories[train_idx],
        s_rpy(train_idx), g_rpy(train_idx)
    )
    norm_stats = train_dataset.get_norm_stats()
    print(f"Trajectory stats (per-joint): mean shape={norm_stats['traj_mean'].shape}, std shape={norm_stats['traj_std'].shape}")
    print(f"  traj_mean: {norm_stats['traj_mean']}")
    print(f"  traj_std: {norm_stats['traj_std']}")
    print(f"Delta stats ({norm_stats['delta_mean'].shape[0]}D): mean={norm_stats['delta_mean']}, std={norm_stats['delta_std']}")

    # Create val/test datasets with train's normalization stats
    val_dataset = TrajectoryDataset(
        start_pos[val_idx], goal_pos[val_idx], trajectories[val_idx],
        s_rpy(val_idx), g_rpy(val_idx), norm_stats
    )
    test_dataset = TrajectoryDataset(
        start_pos[test_idx], goal_pos[test_idx], trajectories[test_idx],
        s_rpy(test_idx), g_rpy(test_idx), norm_stats
    )

    return train_dataset, val_dataset, test_dataset, num_steps, num_joints


# =============================================================================
# Diffusion Utilities
# =============================================================================

def get_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear beta schedule."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def extract(a, t, x_shape):
    """Extract values from a at indices t, reshape for broadcasting."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# =============================================================================
# Training
# =============================================================================

def train(
    zarr_path="trajectory_log.zarr",
    num_epochs=150,
    batch_size=1024,
    lr=1e-4,
    num_timesteps=500,
    device="cuda",
    save_path="diffusion_model.pt",
    train_ratio=0.8,
    val_ratio=0.1,
    log_dir="runs/diffusion",
    hidden_dim=512,
    time_dim=256,
    num_blocks=6,
    trial=None,
):
    print(f"Training on device: {device}")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # Load and split dataset
    train_dataset, val_dataset, test_dataset, num_steps, num_joints = load_and_split_data(
        zarr_path, train_ratio=train_ratio, val_ratio=val_ratio
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True
    )

    # Model and diffusion
    delta_dim = train_dataset.delta.shape[1]  # 3 (pos only) or 6 (pos + rpy)
    condition_dim = delta_dim + num_joints
    model = ConditionalDiffusionModel(
        num_steps=num_steps,
        num_joints=num_joints,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        time_dim=time_dim,
        num_blocks=num_blocks,
    ).to(device)

    diffusion = GaussianDiffusion(num_timesteps=num_timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/10)
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")

    total_batches = len(train_loader)

    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            delta = batch["delta"].to(device, non_blocking=True)
            current_joints = batch["current_joints"].to(device, non_blocking=True)
            trajectory = batch["trajectory"].to(device, non_blocking=True)

            condition = torch.cat([delta, current_joints], dim=-1)
            t = torch.randint(0, num_timesteps, (trajectory.shape[0],), device=device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = diffusion.p_losses(model, trajectory, t, condition)

            if torch.isnan(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

            # Print progress
            pct = (batch_idx + 1) / total_batches * 100
            print(f"\rEpoch {epoch + 1}/{num_epochs} | Batch {batch_idx + 1}/{total_batches} ({pct:.1f}%) | Loss: {loss.item():.6f}", end="")

        train_loss = total_loss / max(num_batches, 1)
        print()  # New line after epoch

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for batch in val_loader:
                delta = batch["delta"].to(device, non_blocking=True)
                current_joints = batch["current_joints"].to(device, non_blocking=True)
                trajectory = batch["trajectory"].to(device, non_blocking=True)

                condition = torch.cat([delta, current_joints], dim=-1)
                t = torch.randint(0, num_timesteps, (trajectory.shape[0],), device=device)

                loss = diffusion.p_losses(model, trajectory, t, condition)
                val_loss += loss.item()
                val_batches += 1

        val_loss = val_loss / max(val_batches, 1)

        # Log to TensorBoard
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Optuna pruning
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                writer.close()
                raise __import__("optuna").TrialPruned()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "traj_mean": train_dataset.traj_mean,
                "traj_std": train_dataset.traj_std,
                "delta_mean": train_dataset.delta_mean,
                "delta_std": train_dataset.delta_std,
                "num_steps": num_steps,
                "num_joints": num_joints,
                "num_timesteps": num_timesteps,
                "hidden_dim": hidden_dim,
                "time_dim": time_dim,
                "condition_dim": condition_dim,
                "num_blocks": num_blocks,
            }, save_path)

        scheduler.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")

    # Test evaluation
    model.eval()
    test_loss = 0
    test_batches = 0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch in test_loader:
            delta = batch["delta"].to(device, non_blocking=True)
            current_joints = batch["current_joints"].to(device, non_blocking=True)
            trajectory = batch["trajectory"].to(device, non_blocking=True)

            condition = torch.cat([delta, current_joints], dim=-1)
            t = torch.randint(0, num_timesteps, (trajectory.shape[0],), device=device)

            loss = diffusion.p_losses(model, trajectory, t, condition)
            test_loss += loss.item()
            test_batches += 1

    test_loss = test_loss / max(test_batches, 1)
    writer.add_scalar("Loss/test", test_loss, num_epochs)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Best model saved to {save_path}")

    writer.close()

    return model, diffusion, train_dataset, best_val_loss


# =============================================================================
# Inference
# =============================================================================

def load_model(save_path="diffusion_model.pt", device="cpu"):
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    if checkpoint.get("dropout", None) is None:
        print("Cannot find 'dropout' key in checkpoint")
        sys.exit(-1)

    if checkpoint['dropout'] > 0.0:
        model = ConditionalDropOutDiffusionModel(
            num_steps=checkpoint["num_steps"],
            num_joints=checkpoint["num_joints"],
            condition_dim=checkpoint.get("condition_dim", 6),
            hidden_dim=checkpoint.get("hidden_dim", 512),
            time_dim=checkpoint.get("time_dim", 256),
            num_blocks=checkpoint.get("num_blocks", 4),
            dropout_rate=checkpoint['dropout']
        ).to(device)
    else:
        model = ConditionalDiffusionModel(
            num_steps=checkpoint["num_steps"],
            num_joints=checkpoint["num_joints"],
            condition_dim=checkpoint.get("condition_dim", 6),
            hidden_dim=checkpoint.get("hidden_dim", 512),
            time_dim=checkpoint.get("time_dim", 256),
            num_blocks=checkpoint.get("num_blocks", 4),
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = GaussianDiffusion(num_timesteps=checkpoint["num_timesteps"], device=device)

    return model, diffusion, checkpoint


def generate_trajectory(model, diffusion, checkpoint, delta, current_joints, device="cpu", ddim_steps=0):
    """
    Generate a trajectory given delta (pos + rpy) and current joint values.

    delta: (6,) array - [dx, dy, dz, droll, dpitch, dyaw]
    current_joints: (12,) array - current joint angles
    ddim_steps: number of DDIM denoising steps (default 50, set to 0 for full DDPM)

    Returns: (num_steps, num_joints) trajectory
    """
    model.eval()

    # Normalize inputs
    delta_mean = checkpoint["delta_mean"]
    delta_std = checkpoint["delta_std"]
    traj_mean = checkpoint["traj_mean"]
    traj_std = checkpoint["traj_std"]

    delta_norm = (np.array(delta) - delta_mean) / delta_std
    current_joints_norm = (np.array(current_joints) - traj_mean) / traj_std

    condition = torch.tensor(
        np.concatenate([delta_norm, current_joints_norm]), dtype=torch.float32
    ).unsqueeze(0).to(device)

    # Sample trajectory
    shape = (1, checkpoint["num_steps"], checkpoint["num_joints"])
    if ddim_steps > 0:
        trajectory = diffusion.ddim_sample(model, condition, shape, ddim_steps=ddim_steps)
    else:
        trajectory = diffusion.sample(model, condition, shape)

    # Denormalize
    trajectory = trajectory.cpu().numpy()[0]
    trajectory = trajectory * checkpoint["traj_std"] + checkpoint["traj_mean"]

    return trajectory


def generate_trajectory_retroaction(model, diffusion, checkpoint, delta, current_joints, prev_joints, prev_actions, device="cpu", ddim_steps=0):
    """
    Generate a trajectory given delta (pos + rpy) and current joint values.

    delta: (6,) array - [dx, dy, dz, droll, dpitch, dyaw]
    current_joints: (12,) array - current joint angles
    ddim_steps: number of DDIM denoising steps (default 50, set to 0 for full DDPM)

    Returns: (num_steps, num_joints) trajectory
    """
    model.eval()

    # Normalize inputs
    delta_mean = checkpoint["delta_mean"]
    delta_std = checkpoint["delta_std"]
    traj_mean = checkpoint["traj_mean"]
    traj_std = checkpoint["traj_std"]

    delta_norm = (np.array(delta) - delta_mean) / delta_std
    current_joints_norm = (np.array(current_joints) - traj_mean) / traj_std
    q_prev_norm = (np.array(prev_joints) - traj_mean) / traj_std
    a_prev_norm = (np.array(prev_actions) - traj_mean) / traj_std

    condition = torch.tensor(
        np.concatenate([delta_norm, current_joints_norm, q_prev_norm, a_prev_norm]), dtype=torch.float32
    ).unsqueeze(0).to(device)

    # Sample trajectory
    shape = (1, checkpoint["num_steps"], checkpoint["num_joints"])
    if ddim_steps > 0:
        traj_retroaction = diffusion.ddim_sample(model, condition, shape, ddim_steps=ddim_steps)
    else:
        traj_retroaction = diffusion.sample(model, condition, shape)

    # Denormalize
    traj_retroaction = traj_retroaction.cpu().numpy()[0]
    traj_retroaction = traj_retroaction * checkpoint["traj_std"] + checkpoint["traj_mean"]

    return traj_retroaction


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    train(
        zarr_path="trajectory_log.zarr",
        save_path="diffusion_model.pt",
    )
