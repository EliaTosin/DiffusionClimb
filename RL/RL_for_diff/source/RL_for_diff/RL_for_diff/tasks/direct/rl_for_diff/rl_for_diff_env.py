from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.markers import VisualizationMarkers

from ink_kin_stance.constants import (
    MODEL_TO_ARTICULATION,
    ARTICULATION_TO_MODEL,
    LEG_MODEL_INDICES,
    LEG_IS_RIGHT,
    MODEL_JOINT_ORDER,
)
from ink_kin_stance.diffusion.inference import load_model

from .rl_for_diff_env_cfg import RlForDiffEnvCfg


# Phase encoding constants
_PHASE_SETTLE = 0
_PHASE_PRE = 1
_PHASE_STEP = 2
_PHASE_POST = 3


class RlForDiffEnv(DirectRLEnv):
    """Aliengo stance stabilization during a diffusion-driven leg step.

    Episode phases:
        settle   — hold default pose, no RL (robot settles under PD)
        pre-step — RL controls 9 non-stepping joints, stepping leg at default
        step     — diffusion drives stepping leg, RL controls 9 non-stepping
        post-step— RL controls 9 non-stepping joints, stepping leg at last traj target
    """

    cfg: RlForDiffEnvCfg

    def __init__(self, cfg: RlForDiffEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._load_diffusion_model()
        self._build_joint_mappings()
        self._init_buffers()

    # =========================================================================
    # Setup
    # =========================================================================

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._step_visualizer = VisualizationMarkers(self.cfg.step_visualizer)

    def _load_diffusion_model(self):
        device_str = "cuda" if "cuda" in str(self.device) else "cpu"
        self._step_model, self._step_diffusion, self._step_ckpt = load_model(
            self.cfg.step_model_path, device=device_str,
        )
        self._traj_mean = torch.tensor(
            self._step_ckpt["traj_mean"], dtype=torch.float32, device=self.device,
        )
        self._traj_std = torch.tensor(
            self._step_ckpt["traj_std"], dtype=torch.float32, device=self.device,
        )
        self._delta_mean = torch.tensor(
            self._step_ckpt["delta_mean"], dtype=torch.float32, device=self.device,
        )
        self._delta_std = torch.tensor(
            self._step_ckpt["delta_std"], dtype=torch.float32, device=self.device,
        )

    def _build_joint_mappings(self):
        dof_names = self.robot.joint_names
        print(f"Isaac Lab articulation joint order: {dof_names}")

        artic_name_to_idx = {name: i for i, name in enumerate(dof_names)}
        self._model_to_artic = torch.tensor(
            [artic_name_to_idx[name] for name in MODEL_JOINT_ORDER],
            dtype=torch.long, device=self.device,
        )
        model_name_to_idx = {name: i for i, name in enumerate(MODEL_JOINT_ORDER)}
        self._artic_to_model = torch.tensor(
            [model_name_to_idx[name] for name in dof_names],
            dtype=torch.long, device=self.device,
        )

        self._default_pos_artic = self.robot.data.default_joint_pos[0].clone()
        print(f"Default joint pos (artic order): {self._default_pos_artic}")

        self._leg_artic_idx = {}
        for leg, model_indices in LEG_MODEL_INDICES.items():
            artic_indices = [int(self._model_to_artic[mi]) for mi in model_indices]
            self._leg_artic_idx[leg] = torch.tensor(
                artic_indices, dtype=torch.long, device=self.device,
            )

        self._is_right = torch.tensor(
            [LEG_IS_RIGHT[i] for i in range(4)], dtype=torch.bool, device=self.device,
        )

        # Foot/calf body indices for vacuum anchoring
        # The USD merges fixed foot joints into calf bodies, so we anchor calves.
        # The foot contact point is at -0.25m Z offset in the calf's local frame.
        calf_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
        self._foot_body_ids, resolved_names = self.robot.find_bodies(calf_names)
        print(f"Calf body IDs (used for vacuum): {dict(zip(resolved_names, self._foot_body_ids))}")
        print(f"All body names: {self.robot.body_names}")

    def _init_buffers(self):
        N = self.num_envs
        d = self.device

        self._stepping_leg = torch.zeros(N, dtype=torch.long, device=d)
        self._trajectories = torch.zeros(
            N, self.cfg.num_traj_steps, 3, dtype=torch.float32, device=d,
        )
        self._foot_deltas = torch.zeros(N, 3, dtype=torch.float32, device=d)
        self._prev_actions = torch.zeros(N, 12, dtype=torch.float32, device=d)
        self._stepping_mask = torch.zeros(N, 12, dtype=torch.bool, device=d)

        # Phase tracking: global step counter per env
        self._ep_step = torch.zeros(N, dtype=torch.long, device=d)

        # Phase boundaries (computed from config)
        self._settle_end = self.cfg.settle_steps
        self._pre_end = self._settle_end + self.cfg.pre_step_steps
        self._step_end = self._pre_end + self.cfg.num_traj_steps
        self._post_end = self._step_end + self.cfg.post_step_steps

        # Current phase per env
        self._phase = torch.zeros(N, dtype=torch.long, device=d)

        # Trajectory targets
        self._cur_traj_targets = torch.zeros(N, 3, dtype=torch.float32, device=d)
        self._traj_start_targets = torch.zeros(N, 3, dtype=torch.float32, device=d)
        self._traj_goal_targets = torch.zeros(N, 3, dtype=torch.float32, device=d)

        # Flag: whether trajectory has been regenerated for this episode
        self._traj_regenerated = torch.zeros(N, dtype=torch.bool, device=d)

        # Vacuum-cup foot anchoring buffers
        self._foot_anchor_pos = torch.zeros(N, 4, 3, dtype=torch.float32, device=d)
        self._vacuum_forces = torch.zeros(N, 4, 3, dtype=torch.float32, device=d)
        self._vacuum_torques = torch.zeros(N, 4, 3, dtype=torch.float32, device=d)
        self._anchors_set = torch.zeros(N, dtype=torch.bool, device=d)

        # Trunk movement tracking (local frame, relative to env origin)
        self._trunk_start_pos = torch.zeros(N, 3, dtype=torch.float32, device=d)

    # =========================================================================
    # Phase helpers
    # =========================================================================

    def _update_phases(self):
        """Update the phase for each env based on _ep_step."""
        self._phase[:] = _PHASE_POST  # default to post
        self._phase[self._ep_step < self._step_end] = _PHASE_STEP
        self._phase[self._ep_step < self._pre_end] = _PHASE_PRE
        self._phase[self._ep_step < self._settle_end] = _PHASE_SETTLE

    def _get_diffusion_step_idx(self) -> torch.Tensor:
        """Get the trajectory frame index (0..num_traj_steps-1) for envs in STEP phase."""
        return (self._ep_step - self._pre_end).clamp(0, self.cfg.num_traj_steps - 1)

    # =========================================================================
    # Diffusion trajectory generation (batched, all on GPU)
    # =========================================================================

    def _generate_trajectories_batched(
        self, stepping_legs: torch.Tensor, deltas: torch.Tensor,
        current_leg_joints: torch.Tensor,
    ) -> torch.Tensor:
        N = deltas.shape[0]
        if N == 0:
            return torch.zeros(0, self.cfg.num_traj_steps, 3,
                               dtype=torch.float32, device=self.device)
        is_right = self._is_right[stepping_legs]

        d_in = deltas.clone()
        j_in = current_leg_joints.clone()
        d_in[is_right, 1] *= -1
        j_in[is_right, 0] *= -1

        d_norm = (d_in - self._delta_mean) / self._delta_std
        j_norm = (j_in - self._traj_mean) / self._traj_std
        condition = torch.cat([d_norm, j_norm], dim=-1)

        shape = (N, self._step_ckpt["num_steps"], self._step_ckpt["num_joints"])
        with torch.inference_mode():
            if self.cfg.ddim_steps > 0:
                traj = self._step_diffusion.ddim_sample(
                    self._step_model, condition, shape,
                    ddim_steps=self.cfg.ddim_steps,
                )
            else:
                traj = self._step_diffusion.sample(
                    self._step_model, condition, shape,
                )

        traj = traj * self._traj_std + self._traj_mean
        traj[is_right, :, 0] *= -1
        return traj

    def _regenerate_trajectories(self, env_ids: torch.Tensor):
        """Regenerate diffusion trajectories using actual stepping leg joints.

        Forces frame 0 to match actual joints and blends the first few frames
        to avoid a discontinuity at the pre→step transition.
        """
        num = len(env_ids)
        if num == 0:
            return

        # Read actual joint positions from sim
        actual_pos = self.robot.data.joint_pos[env_ids]  # (num, 12)

        cur_leg_joints = torch.zeros(num, 3, dtype=torch.float32, device=self.device)
        for i in range(num):
            eid = env_ids[i]
            leg = int(self._stepping_leg[eid])
            ai = self._leg_artic_idx[leg]
            cur_leg_joints[i] = actual_pos[i, ai]

        traj = self._generate_trajectories_batched(
            self._stepping_leg[env_ids],
            self._foot_deltas[env_ids],
            cur_leg_joints,
        )

        # Blend: force frame 0 to actual joints, linearly blend over first few frames
        blend_frames = min(5, self.cfg.num_traj_steps)
        for f in range(blend_frames):
            alpha = f / blend_frames  # 0 at frame 0, approaching 1 at blend_frames
            traj[:, f, :] = (1 - alpha) * cur_leg_joints + alpha * traj[:, f, :]

        self._trajectories[env_ids] = traj
        self._traj_start_targets[env_ids] = self._trajectories[env_ids, 0, :]
        self._traj_goal_targets[env_ids] = self._trajectories[env_ids, -1, :]
        self._traj_regenerated[env_ids] = True

    # =========================================================================
    # DirectRLEnv interface
    # =========================================================================

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # Update phases
        self._update_phases()

        # Regenerate trajectories at pre→step transition
        needs_regen = (self._phase == _PHASE_STEP) & (~self._traj_regenerated)
        if needs_regen.any():
            regen_ids = torch.where(needs_regen)[0]
            self._regenerate_trajectories(regen_ids)

        # Read current trajectory targets for envs in step phase
        diff_idx = self._get_diffusion_step_idx()
        self._cur_traj_targets = self._trajectories[
            torch.arange(self.num_envs, device=self.device), diff_idx,
        ]

        # Advance episode step counter
        self._ep_step += 1

    def _apply_action(self) -> None:
        targets = self._default_pos_artic.unsqueeze(0).expand(
            self.num_envs, -1,
        ).clone()

        is_settle = self._phase == _PHASE_SETTLE
        is_pre = self._phase == _PHASE_PRE
        is_step = self._phase == _PHASE_STEP
        is_post = self._phase == _PHASE_POST
        is_active = is_pre | is_step | is_post  # RL controls 9 joints

        # Apply RL offsets to non-stepping joints for active phases
        if is_active.any():
            # Zero out actions for stepping leg joints (RL never controls them)
            masked_actions = self.actions.clone()
            for leg_idx in range(4):
                leg_mask = (self._stepping_leg == leg_idx)
                if not leg_mask.any():
                    continue
                ai = self._leg_artic_idx[leg_idx]
                for j in range(3):
                    masked_actions[leg_mask, ai[j]] = 0.0

            targets[is_active] += masked_actions[is_active] * self.cfg.action_scale

        # During step phase: override stepping leg with diffusion trajectory
        if is_step.any():
            traj = self._cur_traj_targets
            for leg_idx in range(4):
                mask = is_step & (self._stepping_leg == leg_idx)
                if not mask.any():
                    continue
                ai = self._leg_artic_idx[leg_idx]
                for j in range(3):
                    targets[mask, ai[j]] = traj[mask, j]

        # During post phase: hold stepping leg at last trajectory target
        if is_post.any():
            goal = self._traj_goal_targets
            for leg_idx in range(4):
                mask = is_post & (self._stepping_leg == leg_idx)
                if not mask.any():
                    continue
                ai = self._leg_artic_idx[leg_idx]
                for j in range(3):
                    targets[mask, ai[j]] = goal[mask, j]

        self.robot.set_joint_position_target(targets)

        # --- Vacuum-cup foot anchoring ---
        if self.cfg.vacuum_anchor_enabled:
            # Capture anchor positions after settle phase (when robot has stabilized).
            # Only capture once: when env is past settle and anchors not yet set.
            needs_anchor = (~self._anchors_set) & (self._phase >= _PHASE_PRE)
            if needs_anchor.any():
                ids = torch.where(needs_anchor)[0]
                for i, bid in enumerate(self._foot_body_ids):
                    self._foot_anchor_pos[ids, i] = self.robot.data.body_pos_w[ids, bid]
                # Also record trunk start position for progress tracking
                self._trunk_start_pos[ids] = (
                    self.robot.data.root_pos_w[ids, :3] - self.scene.env_origins[ids]
                )
                self._anchors_set[ids] = True

            # Compute spring-damper forces for all 4 feet (calves).
            self._vacuum_forces.zero_()
            for i, bid in enumerate(self._foot_body_ids):
                pos = self.robot.data.body_pos_w[:, bid]        # (N, 3)
                vel = self.robot.data.body_lin_vel_w[:, bid]     # (N, 3)
                disp = pos - self._foot_anchor_pos[:, i]         # (N, 3)
                self._vacuum_forces[:, i] = (
                    -self.cfg.vacuum_stiffness * disp
                    - self.cfg.vacuum_damping * vel
                )

            # Clamp force magnitude to prevent explosion
            self._vacuum_forces.clamp_(
                -self.cfg.vacuum_max_force, self.cfg.vacuum_max_force,
            )

            # Zero out stepping foot force (it must be free to move)
            step_idx = self._stepping_leg.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3)
            self._vacuum_forces.scatter_(1, step_idx, 0.0)

            # Zero out forces for envs still in settle (anchors not yet set)
            not_anchored = ~self._anchors_set
            if not_anchored.any():
                self._vacuum_forces[not_anchored] = 0.0

            # Apply external forces (torques stay zero → rotation free)
            self.robot.set_external_force_and_torque(
                self._vacuum_forces,
                self._vacuum_torques,
                body_ids=self._foot_body_ids,
                is_global=True,
            )

            if self.sim.has_gui():
                # QUI estrai posizione dei piedi che fanno stepping
                foot_ids = torch.tensor(self._foot_body_ids, device=self.device)[self._stepping_leg]
                translation = self.robot.data.body_pos_w[torch.arange(self.num_envs), foot_ids] # stepping foot pos w
                self._step_visualizer.visualize(translations=translation)


    def _get_observations(self) -> dict:
        proj_grav = self.robot.data.projected_gravity_b        # (N, 3)
        ang_vel = self.robot.data.root_ang_vel_b               # (N, 3)
        base_h = (
            self.robot.data.root_pos_w[:, 2:3]
            - self.scene.env_origins[:, 2:3]
        )                                                       # (N, 1)
        j_pos = self.robot.data.joint_pos                       # (N, 12)
        j_vel = self.robot.data.joint_vel                       # (N, 12)

        # Stepping leg one-hot
        onehot = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device,
        )
        onehot.scatter_(1, self._stepping_leg.unsqueeze(-1), 1.0)

        # Phase one-hot (settle=0, pre=1, step=2, post=3)
        phase_onehot = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device,
        )
        phase_onehot.scatter_(1, self._phase.unsqueeze(-1), 1.0)

        # Trajectory progress [0, 1] within step phase (0 otherwise)
        diff_step = (self._ep_step - self._pre_end).clamp(0, self.cfg.num_traj_steps).float()
        progress = (diff_step / self.cfg.num_traj_steps).unsqueeze(-1)  # (N, 1)

        # Trunk displacement from start (XY only)
        trunk_pos_local = (
            self.robot.data.root_pos_w[:, :3] - self.scene.env_origins
        )
        trunk_disp_xy = (trunk_pos_local[:, :2] - self._trunk_start_pos[:, :2])  # (N, 2)

        # Trunk target displacement (XY)
        target_disp_xy = self._foot_deltas[:, :2] * self.cfg.trunk_move_ratio    # (N, 2)

        obs = torch.cat([
            proj_grav,                # 3
            ang_vel,                  # 3
            base_h,                   # 1
            j_pos,                    # 12
            j_vel,                    # 12
            onehot,                   # 4  which leg steps
            phase_onehot,             # 4  current phase
            progress,                 # 1  step progress
            self._cur_traj_targets,   # 3  current diffusion frame
            self._traj_start_targets, # 3  first frame of trajectory
            self._traj_goal_targets,  # 3  last frame of trajectory
            self._foot_deltas,        # 3
            trunk_disp_xy,            # 2  how far trunk has moved (XY)
            target_disp_xy,           # 2  where trunk should go (XY)
        ], dim=-1)                    # total: 56

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Compute trunk displacement and target for reward
        trunk_pos_local = (
            self.robot.data.root_pos_w[:, :3] - self.scene.env_origins
        )
        trunk_disp_xy = trunk_pos_local[:, :2] - self._trunk_start_pos[:, :2]
        target_disp_xy = self._foot_deltas[:, :2] * self.cfg.trunk_move_ratio

        total, terms = compute_rewards(
            self.cfg.rew_scale_orientation,
            self.cfg.rew_scale_base_height,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_trunk_progress,
            self.cfg.target_base_height,
            self.robot.data.projected_gravity_b,
            self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2],
            self.robot.data.root_ang_vel_b,
            self.robot.data.joint_vel,
            self.actions,
            self._prev_actions,
            self._stepping_mask,
            trunk_disp_xy,
            target_disp_xy,
        )
        self._prev_actions = self.actions.clone()

        # Log individual reward terms to TensorBoard
        term_names = [
            "orientation", "base_height", "ang_vel",
            "joint_vel", "action_rate", "trunk_progress",
        ]
        log_dict = {}
        for i, name in enumerate(term_names):
            log_dict[f"rew/{name}"] = terms[:, i].mean()

        # Debug: trunk movement info
        trunk_disp_mag = torch.norm(trunk_disp_xy, dim=-1)
        target_mag = torch.norm(target_disp_xy, dim=-1)
        log_dict["debug/trunk_displacement_mm"] = trunk_disp_mag.mean() * 1000
        log_dict["debug/trunk_target_mm"] = target_mag.mean() * 1000

        if self.cfg.vacuum_anchor_enabled:
            force_norms = torch.norm(self._vacuum_forces, dim=-1)
            log_dict["debug/vacuum_force_mean"] = force_norms.mean()
            log_dict["debug/vacuum_force_max"] = force_norms.max()

        self.extras["log"] = log_dict

        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self._ep_step >= self._post_end

        grav_z = self.robot.data.projected_gravity_b[:, 2]
        too_tilted = grav_z > -self.cfg.max_body_tilt_cos

        base_h = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        too_low = base_h < self.cfg.min_base_height

        terminated = too_tilted | too_low
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        # 1. Random stepping leg
        self._stepping_leg[env_ids] = torch.randint(
            0, 4, (num_reset,), device=self.device,
        )

        # 2. Stepping mask
        self._stepping_mask[env_ids] = False
        for i in range(num_reset):
            eid = env_ids[i]
            leg = int(self._stepping_leg[eid])
            ai = self._leg_artic_idx[leg]
            self._stepping_mask[eid, ai] = True

        # 3. Random foot deltas
        self._foot_deltas[env_ids, 0] = sample_uniform(
            self.cfg.delta_x_range[0], self.cfg.delta_x_range[1],
            (num_reset,), self.device,
        )
        self._foot_deltas[env_ids, 1] = sample_uniform(
            self.cfg.delta_y_range[0], self.cfg.delta_y_range[1],
            (num_reset,), self.device,
        )
        self._foot_deltas[env_ids, 2] = 0.0  # flat ground, no z

        # 4. Initial joints (nominal)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)

        # 5. Generate initial trajectory (will be regenerated at pre→step transition)
        cur_leg_joints = torch.zeros(
            num_reset, 3, dtype=torch.float32, device=self.device,
        )
        for i in range(num_reset):
            eid = env_ids[i]
            leg = int(self._stepping_leg[eid])
            ai = self._leg_artic_idx[leg]
            cur_leg_joints[i] = joint_pos[i, ai]

        self._trajectories[env_ids] = self._generate_trajectories_batched(
            self._stepping_leg[env_ids],
            self._foot_deltas[env_ids],
            cur_leg_joints,
        )
        self._traj_start_targets[env_ids] = self._trajectories[env_ids, 0, :]
        self._traj_goal_targets[env_ids] = self._trajectories[env_ids, -1, :]

        # 6. Reset counters
        self._ep_step[env_ids] = 0
        self._traj_regenerated[env_ids] = False
        self._prev_actions[env_ids] = 0.0

        # 7. Write state to sim
        default_root = self.robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 8. Mark vacuum anchors for recapture on next _apply_action
        self._anchors_set[env_ids] = False


# =============================================================================
# JIT-compiled reward function
# =============================================================================

@torch.jit.script
def compute_rewards(
    rew_scale_orientation: float,
    rew_scale_base_height: float,
    rew_scale_ang_vel: float,
    rew_scale_joint_vel: float,
    rew_scale_action_rate: float,
    rew_scale_trunk_progress: float,
    target_base_height: float,
    projected_gravity: torch.Tensor,  # (N, 3)
    base_height: torch.Tensor,       # (N,)
    ang_vel: torch.Tensor,           # (N, 3)
    joint_vel: torch.Tensor,         # (N, 12)
    actions: torch.Tensor,           # (N, 12)
    prev_actions: torch.Tensor,      # (N, 12)
    stepping_mask: torch.Tensor,     # (N, 12)
    trunk_disp_xy: torch.Tensor,     # (N, 2) actual trunk XY displacement
    target_disp_xy: torch.Tensor,    # (N, 2) target trunk XY displacement
) -> tuple[torch.Tensor, torch.Tensor]:
    # Orientation: penalize tilt (soft)
    orient = rew_scale_orientation * torch.sum(
        torch.square(projected_gravity[:, :2]), dim=-1,
    )

    # Base height: maintain reasonable height
    height = rew_scale_base_height * torch.square(base_height - target_base_height)

    # Angular velocity: penalize spinning
    ang = rew_scale_ang_vel * torch.sum(torch.square(ang_vel), dim=-1)

    # Joint velocity: penalize non-stepping joint speeds
    non_step_mask = (~stepping_mask).float()
    non_step_vel = joint_vel * non_step_mask
    jv = rew_scale_joint_vel * torch.sum(torch.square(non_step_vel), dim=-1)

    # Action rate: smooth actions
    ar = rew_scale_action_rate * torch.sum(
        torch.square(actions - prev_actions), dim=-1,
    )

    # Trunk progress: reward movement toward target (MAIN reward)
    target_norm = torch.norm(target_disp_xy, dim=-1).clamp(min=1e-6)  # (N,)
    target_dir = target_disp_xy / target_norm.unsqueeze(-1)           # (N, 2)
    # Project actual displacement onto target direction, normalize by target magnitude
    progress = torch.sum(trunk_disp_xy * target_dir, dim=-1) / target_norm  # (N,)
    progress = progress.clamp(-0.5, 1.5)
    trunk_rew = rew_scale_trunk_progress * progress

    total = orient + height + ang + jv + ar + trunk_rew

    # Stack individual terms for logging (N, 6)
    terms = torch.stack([orient, height, ang, jv, ar, trunk_rew], dim=-1)

    return total, terms
