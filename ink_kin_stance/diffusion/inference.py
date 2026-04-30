import numpy as np
import torch

from .model import ConditionalDiffusionModel
from .gaussian_diffusion import GaussianDiffusion


def load_model(model_path, device="cpu"):
    """Load a trained diffusion model and its checkpoint.

    Returns (model, diffusion, checkpoint).
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

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

    diffusion = GaussianDiffusion(
        num_timesteps=checkpoint["num_timesteps"], device=device,
    )

    return model, diffusion, checkpoint


def build_condition(checkpoint, delta, current_joints, device):
    """Normalise inputs and build a condition tensor for a diffusion model."""
    delta_norm = (np.array(delta) - checkpoint["delta_mean"]) / checkpoint["delta_std"]
    joints_norm = (np.array(current_joints) - checkpoint["traj_mean"]) / checkpoint["traj_std"]
    cond = torch.tensor(
        np.concatenate([delta_norm, joints_norm]), dtype=torch.float32,
    ).unsqueeze(0).to(device)
    return cond


def denorm(trajectory_tensor, checkpoint):
    """Convert a normalised torch trajectory → physical numpy array."""
    traj = trajectory_tensor.cpu().numpy()[0]
    return traj * checkpoint["traj_std"] + checkpoint["traj_mean"]


def generate_trajectory(model, diffusion, checkpoint, delta, current_joints,
                        device="cpu", ddim_steps=0):
    """Generate a trajectory given a delta and current joint angles.

    delta:          (D,) array — e.g. [dx,dy,dz] or [dx,dy,dz,dr,dp,dyaw]
    current_joints: (J,) array — current joint angles
    ddim_steps:     number of DDIM denoising steps (0 = full DDPM)

    Returns: (num_steps, num_joints) numpy array of physical joint angles.
    """
    model.eval()

    cond = build_condition(checkpoint, delta, current_joints, device)
    shape = (1, checkpoint["num_steps"], checkpoint["num_joints"])

    if ddim_steps > 0:
        trajectory = diffusion.ddim_sample(model, cond, shape, ddim_steps=ddim_steps)
    else:
        trajectory = diffusion.sample(model, cond, shape)

    return denorm(trajectory, checkpoint)
