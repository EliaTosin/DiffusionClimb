import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg

from ink_kin_stance.constants import PROJECT_ROOT

ALIENGO_USD = os.path.join(PROJECT_ROOT, "Collected_aliengo", "aliengo.usd")
STEP_MODEL_PT = os.path.join(PROJECT_ROOT, "step", "diffusion_model_compat.pt")


@configclass
class RlForDiffEnvCfg(DirectRLEnvCfg):
    # -- env timing -----------------------------------------------------------
    decimation = 10          # 200Hz physics / 10 = 20Hz control
    episode_length_s = 5.0   # Safety margin; actual timeout by phase counter

    # -- spaces ---------------------------------------------------------------
    action_space = 12        # Position offsets for all 12 joints (stepping leg zeroed)
    observation_space = 56
    state_space = 0

    # -- simulation -----------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # -- robot ----------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ALIENGO_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.42),
            joint_pos={
                "FL_hip_joint": 0.0, "FR_hip_joint": 0.0,
                "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
                "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
                "RL_thigh_joint": 0.8, "RR_thigh_joint": 0.8,
                "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
                "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=25000.0,
                damping=200.0,
            ),
        },
    )

    # -- scene ----------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=3.0, replicate_physics=True,
    )

    step_visualizer = VisualizationMarkersCfg(
        prim_path="/Visuals/StepMarker",
        markers={
            "stepping_foot": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
            ),
        },
    )

    # -- diffusion model ------------------------------------------------------
    step_model_path: str = STEP_MODEL_PT
    ddim_steps: int = 50

    # -- action ---------------------------------------------------------------
    action_scale: float = 0.25

    # -- reward scales --------------------------------------------------------
    # Zeroed: robot can't fall with vacuum cups, and we WANT movement
    rew_scale_alive: float = 0.0
    rew_scale_terminated: float = 0.0
    rew_scale_lin_vel: float = 0.0
    rew_scale_pose: float = 0.0
    # Kept (soft constraints):
    rew_scale_orientation: float = -1.0
    rew_scale_base_height: float = -2.0
    rew_scale_ang_vel: float = -0.1
    rew_scale_joint_vel: float = -0.001
    rew_scale_action_rate: float = -0.05
    # New: trunk movement toward stepping direction
    rew_scale_trunk_progress: float = 10.0

    # -- trunk movement target ------------------------------------------------
    trunk_move_ratio: float = 0.25   # trunk moves 1/4 of foot delta per step

    # -- targets --------------------------------------------------------------
    target_base_height: float = 0.42

    # -- termination (relaxed — vacuum cups prevent falling) ------------------
    max_body_tilt_cos: float = 0.5   # ~60deg
    min_base_height: float = 0.15

    # -- episode phases (in RL steps) -----------------------------------------
    settle_steps: int = 10       # hold default pose, let physics stabilize
    pre_step_steps: int = 10     # RL prepares (weight shift), stepping leg at default
    num_traj_steps: int = 20     # diffusion drives stepping leg
    post_step_steps: int = 10    # RL recovers, stepping leg at goal

    # -- foot delta sampling ranges -------------------------------------------
    delta_x_range: tuple = (-0.05, 0.05)
    delta_y_range: tuple = (-0.02, 0.02)
    delta_z_range: tuple = (0.0, 0.0)     # flat ground only

    # -- reset randomisation --------------------------------------------------
    init_joint_noise: float = 0.0

    # -- vacuum-cup foot anchoring --------------------------------------------
    vacuum_anchor_enabled: bool = True
    vacuum_stiffness: float = 5000.0    # N/m, translational spring constant
    vacuum_damping: float = 200.0       # N*s/m, translational damper constant
    vacuum_max_force: float = 500.0     # N, clamp per-axis force magnitude
