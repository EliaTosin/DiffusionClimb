import numpy as np
import zarr
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Normalised trajectory dataset for diffusion training.

    Works for both still (pos+rpy delta, 12 joints) and step (pos delta, 3 joints).
    """

    def __init__(self, start_pos, goal_pos, trajectories,
                 start_rpy=None, goal_rpy=None, norm_stats=None):
        self.trajectories = trajectories.copy()

        delta_pos = goal_pos - start_pos
        if start_rpy is not None and goal_rpy is not None:
            delta_rpy = goal_rpy - start_rpy
            self.delta = np.concatenate([delta_pos, delta_rpy], axis=1)
        else:
            self.delta = delta_pos

        self.current_joints = self.trajectories[:, 0, :]

        if norm_stats is not None:
            self.traj_mean = norm_stats["traj_mean"]
            self.traj_std = norm_stats["traj_std"]
            self.delta_mean = norm_stats["delta_mean"]
            self.delta_std = norm_stats["delta_std"]
        else:
            self.traj_mean = self.trajectories.mean(axis=(0, 1))
            self.traj_std = self.trajectories.std(axis=(0, 1))
            self.traj_std = np.where(self.traj_std < 1e-6, 1.0, self.traj_std)
            self.delta_mean = self.delta.mean(axis=0)
            self.delta_std = self.delta.std(axis=0)
            self.delta_std = np.where(self.delta_std < 1e-6, 1.0, self.delta_std)

        self.trajectories = (self.trajectories - self.traj_mean) / self.traj_std
        self.delta = (self.delta - self.delta_mean) / self.delta_std
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
    """Load a zarr trajectory store and split into train/val/test datasets.

    Returns (train_dataset, val_dataset, test_dataset, num_steps, num_joints).
    """
    root = zarr.open_group(zarr_path, mode="r")
    print(f"Loading data from {zarr_path}")

    start_pos = np.array(root["start_positions"][:], dtype=np.float32)
    goal_pos = np.array(root["goal_positions"][:], dtype=np.float32)
    trajectories = np.array(root["joint_angles"][:], dtype=np.float32)
    start_rpy = np.array(root["start_rpys"][:], dtype=np.float32) if "start_rpys" in root else None
    goal_rpy = np.array(root["goal_rpys"][:], dtype=np.float32) if "goal_rpys" in root else None

    num_steps = trajectories.shape[1]
    num_joints = trajectories.shape[2]
    print(f"Trajectory shape: {trajectories.shape} (num_steps={num_steps}, num_joints={num_joints})")
    print(f"Loaded {len(trajectories)} trajectories")

    if np.any(np.isnan(trajectories)) or np.any(np.isinf(trajectories)):
        print("WARNING: NaN or Inf found in trajectories, replacing with 0")
        trajectories = np.nan_to_num(trajectories, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(trajectories)
    np.random.seed(seed)
    indices = np.random.permutation(n)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    def _rpy_slice(arr, idx):
        return arr[idx] if arr is not None else None

    train_dataset = TrajectoryDataset(
        start_pos[train_idx], goal_pos[train_idx], trajectories[train_idx],
        _rpy_slice(start_rpy, train_idx), _rpy_slice(goal_rpy, train_idx),
    )
    norm_stats = train_dataset.get_norm_stats()
    print(f"  traj_mean: {norm_stats['traj_mean']}")
    print(f"  traj_std:  {norm_stats['traj_std']}")
    print(f"  delta stats ({norm_stats['delta_mean'].shape[0]}D): "
          f"mean={norm_stats['delta_mean']}, std={norm_stats['delta_std']}")

    val_dataset = TrajectoryDataset(
        start_pos[val_idx], goal_pos[val_idx], trajectories[val_idx],
        _rpy_slice(start_rpy, val_idx), _rpy_slice(goal_rpy, val_idx), norm_stats,
    )
    test_dataset = TrajectoryDataset(
        start_pos[test_idx], goal_pos[test_idx], trajectories[test_idx],
        _rpy_slice(start_rpy, test_idx), _rpy_slice(goal_rpy, test_idx), norm_stats,
    )

    return train_dataset, val_dataset, test_dataset, num_steps, num_joints
