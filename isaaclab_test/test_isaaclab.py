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
parser = argparse.ArgumentParser(description="Test IsaacLab with Diffusion Model.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="aliengo-flat-direct", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")


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
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg
)

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
# import source.grace_template.grace_template as grace_template

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg, use_full_traj = False, plot_traj=False):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.viewer.eye = [3., 3., 3.]
    env_cfg.viewer.lookat = [-1., 1., 0.]

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    resume_path_step = "step/diffusion_model_compat.pt"
    resume_path_still = "still/diffusion_model_compat.pt"

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = ".env_logs"

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load checkpoint
    device = torch.device(env_cfg.sim.device)

    from aliengo_isaaclab_diffusion import get_isaaclab_diffusion_model
    AliengoIsaaclabDiffusion = get_isaaclab_diffusion_model() # using factory method to retrieve the class without the imports
    aliengo_isaaclab_diffusion = AliengoIsaaclabDiffusion(env=env.env.env)

    from aliengo_isaaclab_simulation import get_isaaclab_diffusion_simulation
    AliengoIsaaclabSimulation = get_isaaclab_diffusion_simulation()
    aliengo_isaaclab_simulation = AliengoIsaaclabSimulation(aliengo_isaaclab_diffusion)


    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            aliengo_isaaclab_simulation.run(simulation_app=simulation_app)

    # close the simulator
    env.close()



if __name__ == "__main__":
    # run the main function
    main(plot_traj=False)
    # main(plot_traj=False, use_full_traj=True)
    # main(plot_traj=False, use_full_traj=False)

    # close sim app
    simulation_app.close()
