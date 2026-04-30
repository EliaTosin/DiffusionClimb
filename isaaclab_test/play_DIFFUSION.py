# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""
import numpy as np
import pinocchio as pin # NEEDS TO BE IMPORTED BEFORE isaaclab.app.AppLauncher (https://github.com/isaac-sim/IsaacLab/issues/1936#issuecomment-3439081801)

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
# import source.grace_template.grace_template as grace_template

from isaaclab.utils.math import quat_apply_inverse
from isaaclab.envs import ManagerBasedRLEnv
from pathlib import Path
import random
from still.diffusion_train import load_model, generate_trajectory
from still.eval_diffusion import stance, rotation_rpy
import matplotlib.pyplot as plt
from isaac.isaac_aliengo import AlengoDiffusion

def find_last_valid_checkpoint(
    outputs_dir: str | Path = "outputs",
    checkpoint_name: str = "latest.ckpt"
) -> Path:

    outputs_dir = Path(outputs_dir).resolve()

    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    # prende tutte le directory sotto outputs (ricorsivamente)
    all_dirs = sorted(
        (p for p in outputs_dir.rglob("*") if p.is_dir()),
        reverse=True
    )

    for d in all_dirs:
        ckpt_path = d / "checkpoints" / checkpoint_name
        if ckpt_path.is_file():
            return ckpt_path.resolve()

    raise FileNotFoundError(
        f"No valid checkpoint '{checkpoint_name}' found under {outputs_dir}"
    )

# getting diffusion obs as done in evaluate_diffusion()
def get_diffusion_observation(env_wrapper, BODY_HEIGHT=0.43):
    env : ManagerBasedRLEnv = env_wrapper.env.env
    robot = env.scene.articulations['robot']

    # Generate random start and goal
    start_pos = np.array([
        robot.data.root_pos_w[0, 0].item(),
        robot.data.root_pos_w[0, 1].item(),
        robot.data.root_pos_w[0, 2].item(),
    ])

    goal_pos = np.array([
        robot.data.root_pos_w[0, 0].item() + random.uniform(-0.15, 0.15),
        robot.data.root_pos_w[0, 1].item() + random.uniform(-0.05, 0.05),
        robot.data.root_pos_w[0, 2].item() + random.uniform(-0.1, 0.1)
    ])

    joint_pos_art = robot.data.joint_pos.cpu().flatten()
    joint_pos_ik = AlengoDiffusion._articulation_to_model_order(AlengoDiffusion, joint_pos_art)

    return start_pos, goal_pos, joint_pos_ik

error_ik = []
error_diff = []
error_base_xyz = []
def plot_trajectories(joint_names, init_pose, goal_pose):
    # Conversione in array numpy
    data_ik = np.array(error_ik)
    data_diff = np.array(error_diff)
    data_base = np.array(error_base_xyz)

    if data_ik.size == 0 or data_diff.size == 0 or data_base.size == 0:
        print("Errore: Liste dei dati vuote.")
        return

    time_steps = np.arange(data_ik.shape[0])

    # Griglia 4x4 (16 slot)
    fig, axes = plt.subplots(4, 4, figsize=(18, 16), sharex=True)
    axes = axes.flatten()

    # --- 1. PLOT GIUNTI (Indici 0-11) ---
    for i in range(12):
        ax = axes[i]
        ax.plot(time_steps, data_ik[:, i], label='IK', color='royalblue', linewidth=1.5)
        ax.plot(time_steps, data_diff[:, i], label='Diff', color='crimson', linestyle='--', linewidth=1.2)

        ax.set_title(f"{joint_names[i]}", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(fontsize=9)

    # --- 2. PLOT BASE COORDINATES (Indici 12, 13, 14) ---
    base_titles = ['Base Position X', 'Base Position Y', 'Base Position Z']
    base_colors = ['#FF4B4B', '#2ecc71', '#3498db']  # RGB standard

    for i in range(3):
        idx = 12 + i
        ax = axes[idx]

        # Traiettoria attuale (Tratteggiata)
        ax.plot(time_steps, data_base[:, i], color=base_colors[i],
                linestyle='--', linewidth=2, label=f'Current {base_titles[i][-1]}')

        # Target/Goal (Continua)
        ax.axhline(y=goal_pose[i], color=base_colors[i],
                   linestyle='-', linewidth=1.5, label=f'Goal {base_titles[i][-1]}')

        ax.set_title(base_titles[i], fontsize=11, fontweight='bold', color='darkslategrey')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=8, loc='best')

    # --- 3. NASCONDI L'ULTIMA CELLA (Indice 15) ---
    axes[15].axis('off')

    # Pulizia estetica finale
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"Analisi Cinematica e Tracking Base\n"
                 f"Start: {np.round(init_pose, 3)} | Goal: {np.round(goal_pose, 3)}",
                 fontsize=18, fontweight='bold')

    plt.show()



@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg, use_full_traj = False, plot_traj=False):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        resume_path = find_last_valid_checkpoint()
    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load checkpoint
    device = torch.device(env_cfg.sim.device)
    diff_model, diffusion, checkpoint = load_model(resume_path, device=device)


    if plot_traj:
        # init IK model
        URDF_PATH = "../aliengo.urdf"
        model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())

        # Initial configuration
        q_init = pin.neutral(model)
        q_init[2] = env_cfg.robot.init_state.pos[2]
        THIGH_ANGLE = env_cfg.robot.init_state.joint_pos['.*_thigh_joint']
        CALF_ANGLE = env_cfg.robot.init_state.joint_pos['.*_calf_joint']
        for leg in range(4):
            base_idx = 7 + leg * 4
            q_init[base_idx + 0] = 0.0
            q_init[base_idx + 1] = np.sin(THIGH_ANGLE)
            q_init[base_idx + 2] = np.cos(THIGH_ANGLE)
            q_init[base_idx + 3] = CALF_ANGLE
        q_neutral = stance.init_stance(q_init)
        use_full_traj = False

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if plot_traj:
                q_current = q_neutral.copy()

            if use_full_traj:
                """Calcolo Action"""
                start_pos, goal_pos, joint_pos = get_diffusion_observation(env)
                delta_pos = goal_pos - start_pos
                t1 = time.perf_counter()
                traj = generate_trajectory(diff_model, diffusion, checkpoint, delta_pos, joint_pos, device=device)
                t2 = time.perf_counter()
                print("time spent diffusion step: ", t2 - t1)

                traj_tensor = torch.tensor(traj, dtype=torch.float32).to(device)
                for i in range(traj_tensor.shape[0]):
                    obs, _, _, _ = env.step(traj_tensor[i, :].unsqueeze(dim=0))
                time.sleep(1)
                env.reset()
            else:
                init_pos, goal_pos, _ = get_diffusion_observation(env)
                num_steps = 20
                for i in range(num_steps):
                    """Calcolo Action"""
                    # start_pos, _, joint_pos = get_diffusion_observation(env)
                    # delta_pos = goal_pos - start_pos
                    # t1 = time.perf_counter()
                    # traj = generate_trajectory(diff_model, diffusion, checkpoint, delta_pos, joint_pos, device=device)
                    # t2 = time.perf_counter()
                    # print("time spent diffusion step: ", t2 - t1)
                    # # USE THE NEXT PREDICTED ACTION
                    # # action = env.env.env.scene.articulations["robot"].data.default_joint_pos
                    # action = torch.tensor(traj[0, :], device=device).unsqueeze(dim=0) # using just the first action described by the traj (HRC tecnique)
                    """Env Stepping"""
                    obs, _, _, _ = env.step(torch.zeros((env.num_envs, 12),device=device))

                    if plot_traj:
                        alpha = i / (num_steps - 1) if num_steps > 1 else 1.0
                        current_pos = start_pos + alpha * (goal_pos - start_pos)

                        R = rotation_rpy(0, 0, 0)
                        q_new, _ = stance.solve_stance(
                            body_translation=current_pos.tolist(),
                            body_rotation=R,
                            q_init=q_current
                        )
                        q_current = q_new
                        angles_ik = stance.get_joint_angles(q_new)

                        angles_diff = traj[0, :]

                        error_ik.append(angles_ik)
                        error_diff.append(angles_diff)
                        error_base_xyz.append(start_pos)

                time.sleep(1)
                env.reset()
                if plot_traj:
                    plot_trajectories(env.env.env.scene.articulations['robot'].data.joint_names, init_pos, goal_pos)
                    break

    # close the simulator
    env.close()



if __name__ == "__main__":
    # run the main function
    main(plot_traj=False)
    # main(plot_traj=False, use_full_traj=True)
    # main(plot_traj=False, use_full_traj=False)

    # close sim app
    simulation_app.close()
