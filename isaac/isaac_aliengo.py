# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Optional

import numpy as np
import torch

# Numpy compatibility fix for models saved with newer numpy
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core

from pxr import UsdGeom, Usd, UsdPhysics, Sdf, Gf

from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage

from ink_kin_stance.diffusion.inference import load_model
from ink_kin_stance.constants import (
    MODEL_JOINT_ORDER, ARTICULATION_JOINT_ORDER,
    MODEL_TO_ARTICULATION, ARTICULATION_TO_MODEL,
    LEG_MODEL_INDICES, LEG_IS_RIGHT, MIRROR_LEG,
    FOOT_FRAMES, LEG_NAMES,
)
from still.diffusion_train import generate_trajectory_retroaction


class AlengoDiffusion:
    """Aliengo Robot controlled by diffusion models with interpolation and vacuum adhesion."""

    DEFAULT_JOINT_POS = np.array([
        0.0, 0.0, 0.0, 0.0,
        0.8, 0.8, 0.8, 0.8,
        -1.5, -1.5, -1.5, -1.5,
    ])

    def __init__(
        self,
        prim_path: str,
        name: str = "aliengo",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        still_model_path: Optional[str] = None,
        step_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        debug: bool = False,
    ) -> None:
        self.name = name
        self.prim_path = prim_path
        self._device = device
        self._debug = debug

        if position is None:
            position = np.array([0.0, 0.0, 0.45])
        if orientation is None:
            orientation = np.array([1.0, 0.0, 0.0, 0.0])

        add_reference_to_stage(usd_path, prim_path)

        self.robot = Robot(
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
        )

        # Resolve default model paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)

        if still_model_path is None:
            still_model_path = os.path.join(project_root, "still", "models", "diff_model_vel_retroaction_Wloss.pt")
        if step_model_path is None:
            step_model_path = os.path.join(project_root, "step", "models", "diffusion_model_retroaction_BEST.pt")

        from still.diffusion_train import load_model as load_new_model
        self._still_model, self._still_diffusion, self._still_checkpoint = load_new_model(
            still_model_path, device=self._device
        )
        self._step_model, self._step_diffusion, self._step_checkpoint = load_new_model(
            step_model_path, device=self._device
        )

        # Joint ordering
        self._model_to_artic = list(MODEL_TO_ARTICULATION)
        self._artic_to_model = list(ARTICULATION_TO_MODEL)

        # Trajectory state
        self._current_trajectory = None
        self._trajectory_step = 0
        self._num_trajectory_steps = self._still_checkpoint["num_steps"]

        # Per-leg step trajectory state
        self._step_trajectories = {i: None for i in range(4)}
        self._step_trajectory_steps = {i: 0 for i in range(4)}

        # Interpolation state
        self._decimation = 3  # physics steps per trajectory frame (200Hz / 3 ≈ 67Hz control)
        self._upsample_factor = 3  # interpolate 20 diffusion frames → 60 frames
        self._step_counter = 0
        self._sub_step = 0
        self._prev_target = None  # previous frame target (12,)
        self._next_target = None  # next frame target (12,)

        # Held joint positions
        self._held_positions = None

        # Foot tracking
        self._foot_start_positions = {i: None for i in range(4)}
        self._foot_step_deltas = {i: None for i in range(4)}

        # Vacuum adhesion state
        self._vacuum_stepping_leg = None
        self._vacuum_anchors = {i: None for i in range(4)}

    # =========================================================================
    # Trajectory generation
    # =========================================================================

    # def generate_trajectory(self, delta_pos: np.ndarray, current_joints: np.ndarray,
    #                         ddim_steps: int = 0) -> np.ndarray:
    #     """Generate a trunk joint trajectory."""
    #     delta_mean = np.array(self._still_checkpoint["delta_mean"], dtype=np.float32)
    #     delta_std = np.array(self._still_checkpoint["delta_std"], dtype=np.float32)
    #     traj_mean = np.array(self._still_checkpoint["traj_mean"], dtype=np.float32)
    #     traj_std = np.array(self._still_checkpoint["traj_std"], dtype=np.float32)
    #
    #     delta_pos_norm = (np.array(delta_pos, dtype=np.float32) - delta_mean) / delta_std
    #     current_joints_norm = (np.array(current_joints, dtype=np.float32) - traj_mean) / traj_std
    #
    #     condition = torch.tensor(
    #         np.concatenate([delta_pos_norm, current_joints_norm]), dtype=torch.float32
    #     ).unsqueeze(0).to(self._device)
    #
    #     shape = (1, self._still_checkpoint["num_steps"], self._still_checkpoint["num_joints"])
    #     if ddim_steps > 0:
    #         trajectory = self._still_diffusion.ddim_sample(self._still_model, condition, shape, ddim_steps=ddim_steps)
    #     else:
    #         trajectory = self._still_diffusion.sample(self._still_model, condition, shape)
    #
    #     trajectory = trajectory.cpu().numpy()[0]
    #     trajectory = trajectory * traj_std + traj_mean
    #     return trajectory

    def generate_trajectory(self, delta_pos: np.ndarray, current_joints: np.ndarray, ddim_steps: int = 0, device="cuda") -> np.ndarray:
        """Generate a trunk joint trajectory."""
        return generate_trajectory_retroaction(
            model=self._still_model,
            diffusion=self._still_diffusion,
            checkpoint=self._still_checkpoint,
            delta=delta_pos,
            current_joints=current_joints,
            prev_joints=current_joints,
            prev_actions=current_joints,
            ddim_steps=ddim_steps,
            device=device
        )

    # def generate_leg_step_trajectory(self, leg_idx: int, delta_foot_pos: np.ndarray,
    #                                   current_leg_joints: np.ndarray,
    #                                   ddim_steps: int = 0) -> np.ndarray:
    #     """Generate a leg trajectory with mirroring for right-side legs."""
    #     delta_mean = np.array(self._step_checkpoint["delta_mean"], dtype=np.float32)
    #     delta_std = np.array(self._step_checkpoint["delta_std"], dtype=np.float32)
    #     traj_mean = np.array(self._step_checkpoint["traj_mean"], dtype=np.float32)
    #     traj_std = np.array(self._step_checkpoint["traj_std"], dtype=np.float32)
    #
    #     delta = np.array(delta_foot_pos, dtype=np.float32).copy()
    #     joints = np.array(current_leg_joints, dtype=np.float32).copy()
    #
    #     if LEG_IS_RIGHT[leg_idx]:
    #         delta[1] = -delta[1]
    #         joints[0] = -joints[0]
    #
    #     delta_pos_norm = (delta - delta_mean) / delta_std
    #     current_joints_norm = (joints - traj_mean) / traj_std
    #
    #     condition = torch.tensor(
    #         np.concatenate([delta_pos_norm, current_joints_norm]), dtype=torch.float32
    #     ).unsqueeze(0).to(self._device)
    #
    #     shape = (1, self._step_checkpoint["num_steps"], self._step_checkpoint["num_joints"])
    #     if ddim_steps > 0:
    #         trajectory = self._step_diffusion.ddim_sample(self._step_model, condition, shape, ddim_steps=ddim_steps)
    #     else:
    #         trajectory = self._step_diffusion.sample(self._step_model, condition, shape)
    #
    #     trajectory = trajectory.cpu().numpy()[0]
    #     trajectory = trajectory * traj_std + traj_mean
    #
    #     if LEG_IS_RIGHT[leg_idx]:
    #         trajectory[:, 0] = -trajectory[:, 0]
    #
    #     return trajectory

    def generate_leg_step_trajectory(self, leg_idx: int, delta_foot_pos: np.ndarray, current_leg_joints: np.ndarray, ddim_steps: int = 0) -> np.ndarray:
        """Generate a leg trajectory with mirroring for right-side legs."""

        delta = np.array(delta_foot_pos, dtype=np.float32).copy()
        joints = np.array(current_leg_joints, dtype=np.float32).copy()

        if LEG_IS_RIGHT[leg_idx]:
            delta[1] = -delta[1]
            joints[0] = -joints[0]

        trajectory =  generate_trajectory_retroaction(
            model=self._step_model,
            diffusion=self._step_diffusion,
            checkpoint=self._step_checkpoint,
            delta=delta_foot_pos,
            current_joints=current_leg_joints,
            prev_joints=current_leg_joints,
            prev_actions=current_leg_joints,
            ddim_steps=ddim_steps
        )

        if LEG_IS_RIGHT[leg_idx]:
            trajectory[:, 0] = -trajectory[:, 0]

        return trajectory

    def generate_combined_trajectory(self, delta_body: np.ndarray, delta_foot: np.ndarray,
                                      current_joints_12: np.ndarray, stepping_leg: int,
                                      ddim_steps: int = 0) -> np.ndarray:
        """Generate a combined trunk+step trajectory."""
        sym_joints = current_joints_12.copy()
        mirror = MIRROR_LEG[stepping_leg]
        mirror_joints = current_joints_12[mirror * 3: (mirror + 1) * 3].copy()
        mirror_joints[0] = -mirror_joints[0]
        sym_joints[stepping_leg * 3: (stepping_leg + 1) * 3] = mirror_joints

        trunk_delta = np.concatenate([delta_body, np.zeros(3)])
        trunk_traj = self.generate_trajectory(trunk_delta, sym_joints, ddim_steps=ddim_steps)

        leg_joints = current_joints_12[stepping_leg * 3: (stepping_leg + 1) * 3].copy()
        step_traj = self.generate_leg_step_trajectory(
            stepping_leg, delta_foot, leg_joints, ddim_steps=ddim_steps
        )

        combined = trunk_traj.copy()
        combined[:, stepping_leg * 3: (stepping_leg + 1) * 3] = step_traj

        if self._debug:
            leg_start = combined[0, stepping_leg * 3: (stepping_leg + 1) * 3]
            leg_end = combined[-1, stepping_leg * 3: (stepping_leg + 1) * 3]
            print(f"\n  === TRAJECTORY DIAGNOSTIC (leg {LEG_NAMES[stepping_leg]}) ===")
            print(f"  Step leg start: hip={leg_start[0]:+.4f} thigh={leg_start[1]:+.4f} calf={leg_start[2]:+.4f}")
            print(f"  Step leg end:   hip={leg_end[0]:+.4f} thigh={leg_end[1]:+.4f} calf={leg_end[2]:+.4f}")
            print(f"  ===================================\n")

        return combined

    # =========================================================================
    # Trajectory setters
    # =========================================================================

    def _upsample_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Upsample trajectory by interpolating between frames."""
        if self._upsample_factor <= 1:
            return trajectory
        N, J = trajectory.shape
        N_up = (N - 1) * self._upsample_factor + 1
        t_orig = np.linspace(0, 1, N)
        t_up = np.linspace(0, 1, N_up)
        upsampled = np.zeros((N_up, J))
        for j in range(J):
            upsampled[:, j] = np.interp(t_up, t_orig, trajectory[:, j])
        return upsampled

    def set_trajectory(self, trajectory: np.ndarray, stepping_leg: Optional[int] = None):
        """Set a new trajectory to execute."""
        self._current_trajectory = self._upsample_trajectory(trajectory)
        self._trajectory_step = 0
        self._capture_held_positions()
        self._init_interpolation()
        if stepping_leg is not None:
            self._capture_vacuum_anchors(stepping_leg)

    def set_combined_trajectory(self, trajectory: np.ndarray, stepping_leg: Optional[int] = None):
        """Set a combined 12-joint trajectory (trunk+step merged)."""
        self._current_trajectory = self._upsample_trajectory(trajectory)
        self._trajectory_step = 0
        for i in range(4):
            self._step_trajectories[i] = None
        self._capture_held_positions()
        self._init_interpolation()
        if stepping_leg is not None:
            self._capture_vacuum_anchors(stepping_leg)

    def set_leg_step_trajectory(self, leg_idx: int, trajectory: np.ndarray,
                                 delta_foot_pos: Optional[np.ndarray] = None):
        """Set a new step trajectory for a specific leg."""
        self._step_trajectories[leg_idx] = trajectory
        self._step_trajectory_steps[leg_idx] = 0
        self._capture_held_positions()
        self._foot_start_positions[leg_idx] = self.get_foot_body_position(leg_idx)
        self._foot_step_deltas[leg_idx] = delta_foot_pos

    # =========================================================================
    # Interpolation helpers
    # =========================================================================

    def _init_interpolation(self):
        """Initialize interpolation state for a new trajectory."""
        self._sub_step = 0
        if self._held_positions is not None:
            self._prev_target = self._held_positions.copy()
        else:
            self._prev_target = np.array([0.0, 0.8, -1.5] * 4)
        self._next_target = self._compute_frame_target()

    def _compute_frame_target(self) -> np.ndarray:
        """Compute the target for the current trajectory frame index.

        The full combined trajectory (trunk + step) is applied to all 12 joints.
        The trunk model controls the support legs (shifting the trunk), while
        the step model controls the stepping leg (merged into the combined traj).
        """
        target = self._held_positions.copy() if self._held_positions is not None else np.array([0.0, 0.8, -1.5] * 4)

        if (self._current_trajectory is not None
                and self._trajectory_step < len(self._current_trajectory)):
            target = self._current_trajectory[self._trajectory_step].copy()

        # Per-leg step trajectories override
        for leg_idx in range(4):
            traj = self._step_trajectories[leg_idx]
            step = self._step_trajectory_steps[leg_idx]
            if traj is not None and step < len(traj):
                leg_joints = traj[step]
                for i, mi in enumerate(LEG_MODEL_INDICES[leg_idx]):
                    target[mi] = leg_joints[i]

        return target

    def _advance_trajectory_frame(self):
        """Advance trajectory frame counters by one step."""
        if (self._current_trajectory is not None
                and self._trajectory_step < len(self._current_trajectory)):
            self._trajectory_step += 1
            if self._trajectory_step >= len(self._current_trajectory):
                self._current_trajectory = None

        for leg_idx in range(4):
            traj = self._step_trajectories[leg_idx]
            step = self._step_trajectory_steps[leg_idx]
            if traj is not None and step < len(traj):
                self._step_trajectory_steps[leg_idx] = step + 1
                if step + 1 >= len(traj):
                    self._step_trajectories[leg_idx] = None
                    self._update_held_for_leg(leg_idx)

    # =========================================================================
    # Vacuum adhesion
    # =========================================================================

    def _get_calf_world_pos(self, leg_idx: int) -> np.ndarray:
        """Get world position of a calf body."""
        stage = get_current_stage()
        path = self._calf_prim_paths[leg_idx]
        prim = stage.GetPrimAtPath(path)
        xformable = UsdGeom.Xformable(prim)
        world_tf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = world_tf.ExtractTranslation()
        return np.array([t[0], t[1], t[2]])

    def _capture_vacuum_anchors(self, stepping_leg: Optional[int] = None):
        """Enable/disable D6 vacuum joints. Stepping leg is freed, others locked.

        Updates the joint anchor (world-frame position) to the foot's current
        position so the foot is locked exactly where it is right now.
        """
        self._vacuum_stepping_leg = stepping_leg
        if not hasattr(self, '_vacuum_joints'):
            return
        stage = get_current_stage()
        xfCache = UsdGeom.XformCache()
        for i, joint_path in self._vacuum_joints.items():
            joint_prim = stage.GetPrimAtPath(joint_path)
            if not joint_prim.IsValid():
                continue
            joint_api = UsdPhysics.Joint(joint_prim)
            if i == stepping_leg:
                joint_api.GetJointEnabledAttr().Set(False)
            else:
                # Update anchor to current foot world position
                foot_prim = stage.GetPrimAtPath(self._foot_prim_paths[i])
                foot_pose = xfCache.GetLocalToWorldTransform(foot_prim)
                foot_pose = foot_pose.RemoveScaleShear()
                pos = Gf.Vec3f(foot_pose.ExtractTranslation())
                rot = Gf.Quatf(foot_pose.ExtractRotationQuat())
                joint_api.GetLocalPos0Attr().Set(pos)
                joint_api.GetLocalRot0Attr().Set(rot)
                joint_api.GetJointEnabledAttr().Set(True)

    def _apply_vacuum_forces(self):
        """Vacuum handled by D6 joints — nothing to do per step."""
        pass

    def release_vacuum(self):
        """Disable all vacuum joints."""
        self._vacuum_stepping_leg = None
        if not hasattr(self, '_vacuum_joints'):
            return
        stage = get_current_stage()
        for i, joint_path in self._vacuum_joints.items():
            joint_prim = stage.GetPrimAtPath(joint_path)
            if joint_prim.IsValid():
                UsdPhysics.Joint(joint_prim).GetJointEnabledAttr().Set(False)

    # =========================================================================
    # Main control loop
    # =========================================================================

    def _capture_held_positions(self):
        """Snapshot current actual joint positions as held targets."""
        artic_positions = self.robot.get_joint_positions()
        self._held_positions = self._articulation_to_model_order(artic_positions)

    def _update_held_for_leg(self, leg_idx):
        """Update held positions for a specific leg after its step completes."""
        artic_positions = self.robot.get_joint_positions()
        current = self._articulation_to_model_order(artic_positions)
        for mi in LEG_MODEL_INDICES[leg_idx]:
            self._held_positions[mi] = current[mi]

    def forward(self, dt: float, delta_pos: Optional[np.ndarray] = None):
        """Execute one physics step with interpolated trajectory.

        Support legs hold their positions (vacuum effect via PD controller).
        Only the stepping leg follows the diffusion trajectory.
        """
        self._step_counter += 1

        if delta_pos is not None and self._current_trajectory is None:
            artic_positions = self.robot.get_joint_positions()
            current_joints = self._articulation_to_model_order(artic_positions)
            self._current_trajectory = self.generate_trajectory(delta_pos, current_joints)
            self._trajectory_step = 0
            self._init_interpolation()

        any_active = (self._current_trajectory is not None or
                      any(t is not None for t in self._step_trajectories.values()))
        if not any_active and self._prev_target is None:
            return

        # Interpolation: lerp between prev and next frame every physics step
        if self._prev_target is not None and self._next_target is not None:
            alpha = self._sub_step / self._decimation
            target_pos = (1.0 - alpha) * self._prev_target + alpha * self._next_target
        elif self._next_target is not None:
            target_pos = self._next_target.copy()
        elif self._held_positions is not None:
            target_pos = self._held_positions.copy()
        else:
            return

        self._apply_joint_positions(target_pos)
        self._apply_vacuum_forces()

        self._sub_step += 1

        # At decimation boundary: advance to next frame
        if self._sub_step >= self._decimation:
            self._sub_step = 0
            self._prev_target = self._next_target.copy() if self._next_target is not None else None

            # Advance frame counters
            self._advance_trajectory_frame()

            # Compute next frame target
            any_active = (self._current_trajectory is not None or
                          any(t is not None for t in self._step_trajectories.values()))
            if any_active:
                self._next_target = self._compute_frame_target()
            else:
                # Trajectory finished — hold last target
                self._next_target = self._prev_target.copy() if self._prev_target is not None else None
                self._prev_target = None

    # =========================================================================
    # Joint order conversion
    # =========================================================================

    def _model_to_articulation_order(self, model_positions: np.ndarray) -> np.ndarray:
        articulation_positions = np.zeros(12)
        for model_idx, artic_idx in enumerate(self._model_to_artic):
            articulation_positions[artic_idx] = model_positions[model_idx]
        return articulation_positions

    def _articulation_to_model_order(self, articulation_positions: np.ndarray) -> np.ndarray:
        model_positions = np.zeros(12)
        for artic_idx, model_idx in enumerate(self._artic_to_model):
            model_positions[model_idx] = articulation_positions[artic_idx]
        return model_positions

    def _apply_joint_positions(self, target_positions: np.ndarray):
        articulation_positions = self._model_to_articulation_order(target_positions)
        self.robot.apply_action(ArticulationAction(joint_positions=articulation_positions))

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize(self, physics_sim_view=None) -> None:
        """Initialize the robot articulation."""
        self.robot.initialize(physics_sim_view)
        self._build_joint_mapping()
        self.print_joint_info()

        stage = get_current_stage()

        # Discover foot prim paths
        self._foot_prim_paths = {}
        for prim in stage.Traverse():
            name = prim.GetName()
            for i, foot_name in enumerate(FOOT_FRAMES):
                if name == foot_name and str(prim.GetPath()).startswith(self.prim_path):
                    self._foot_prim_paths[i] = str(prim.GetPath())

        print(f"\nFoot prim paths:")
        for i, path in self._foot_prim_paths.items():
            print(f"  {LEG_NAMES[i]}: {path}")

        # Set very high friction on foot collision shapes to simulate vacuum adhesion
        from pxr import UsdPhysics, PhysxSchema
        for i, path in self._foot_prim_paths.items():
            foot_prim = stage.GetPrimAtPath(path)
            # Find collision shape children
            for child in foot_prim.GetChildren():
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    # Apply or get material
                    mat_api = UsdPhysics.MaterialAPI.Apply(child)
                    mat_api.CreateStaticFrictionAttr(10.0)
                    mat_api.CreateDynamicFrictionAttr(10.0)
                    mat_api.CreateRestitutionAttr(0.0)
                    print(f"  {LEG_NAMES[i]} foot: set friction=10.0")
            # Also try on the foot prim itself
            if foot_prim.HasAPI(UsdPhysics.CollisionAPI) or True:
                try:
                    mat_api = UsdPhysics.MaterialAPI.Apply(foot_prim)
                    mat_api.CreateStaticFrictionAttr(10.0)
                    mat_api.CreateDynamicFrictionAttr(10.0)
                    mat_api.CreateRestitutionAttr(0.0)
                except Exception:
                    pass

        # Also set high friction on the ground plane
        ground_prim = stage.GetPrimAtPath("/World/defaultGroundPlane")
        if ground_prim.IsValid():
            for desc in stage.Traverse():
                if str(desc.GetPath()).startswith("/World/defaultGroundPlane"):
                    try:
                        mat_api = UsdPhysics.MaterialAPI.Apply(desc)
                        mat_api.CreateStaticFrictionAttr(10.0)
                        mat_api.CreateDynamicFrictionAttr(10.0)
                    except Exception:
                        pass
        print("  High friction set for vacuum adhesion")

        # Create D6 vacuum joints: lock foot translation, free rotation
        self._vacuum_joints = {}
        xfCache = UsdGeom.XformCache()
        for i, foot_path in self._foot_prim_paths.items():
            foot_prim = stage.GetPrimAtPath(foot_path)
            foot_pose = xfCache.GetLocalToWorldTransform(foot_prim)
            foot_pose = foot_pose.RemoveScaleShear()
            pos = Gf.Vec3f(foot_pose.ExtractTranslation())
            rot = Gf.Quatf(foot_pose.ExtractRotationQuat())

            joint_path = f"{self.prim_path}/vacuum_joint_{LEG_NAMES[i]}"
            joint = UsdPhysics.Joint.Define(stage, joint_path)

            # Body0 = world (no target), Body1 = foot
            joint.CreateBody1Rel().SetTargets([Sdf.Path(foot_path)])
            joint.CreateLocalPos0Attr().Set(pos)
            joint.CreateLocalRot0Attr().Set(rot)
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
            joint.CreateBreakForceAttr().Set(1e20)
            joint.CreateBreakTorqueAttr().Set(1e20)

            # Lock translation (low > high = locked), leave rotation free (no limits)
            prim = joint.GetPrim()
            for axis in ["transX", "transY", "transZ"]:
                limit = UsdPhysics.LimitAPI.Apply(prim, axis)
                limit.CreateLowAttr(1.0)
                limit.CreateHighAttr(-1.0)  # low > high = locked

            # Start disabled — will be enabled when stepping begins
            joint.GetJointEnabledAttr().Set(False)

            self._vacuum_joints[i] = joint_path
            print(f"  {LEG_NAMES[i]} vacuum D6 joint created at {joint_path}")

        # PD gains
        # Lower PD gains to reduce overshoot/oscillation during trajectory tracking
        kps = np.array([5000.0] * 12)
        kds = np.array([100.0] * 12)
        self.robot.get_articulation_controller().set_gains(kps, kds)

        default_model = np.array([0.0, 0.8, -1.5] * 4)
        default_artic = self._model_to_articulation_order(default_model)
        self.robot.set_joints_default_state(positions=default_artic)
        self.robot.post_reset()

    def _build_joint_mapping(self):
        dof_names = self.robot.dof_names
        artic_name_to_idx = {name: i for i, name in enumerate(dof_names)}

        model_to_artic = []
        for model_name in MODEL_JOINT_ORDER:
            if model_name in artic_name_to_idx:
                model_to_artic.append(artic_name_to_idx[model_name])
            else:
                raise RuntimeError(f"Model joint '{model_name}' not found in articulation DOFs: {dof_names}")

        model_name_to_idx = {name: i for i, name in enumerate(MODEL_JOINT_ORDER)}
        artic_to_model = []
        for dof_name in dof_names:
            if dof_name in model_name_to_idx:
                artic_to_model.append(model_name_to_idx[dof_name])
            else:
                raise RuntimeError(f"Articulation DOF '{dof_name}' not found in model joints: {MODEL_JOINT_ORDER}")

        old_m2a = list(self._model_to_artic)
        self._model_to_artic = model_to_artic
        self._artic_to_model = artic_to_model

        if model_to_artic != old_m2a:
            print(f"\n*** JOINT MAPPING UPDATED from actual DOF order ***")
        else:
            print(f"\n  Joint mapping matches hardcoded values.")

    def print_joint_info(self):
        dof_names = self.robot.dof_names
        print("\n" + "=" * 60)
        print("ARTICULATION JOINT INFORMATION")
        print("=" * 60)
        print(f"Number of DOFs: {self.robot.num_dof}")
        for i, name in enumerate(dof_names):
            print(f"  [{i:2d}] {name}")
        print("=" * 60 + "\n")

    # =========================================================================
    # Foot position queries
    # =========================================================================

    def get_foot_position(self, leg_idx: int) -> np.ndarray:
        stage = get_current_stage()
        foot_path = self._foot_prim_paths[leg_idx]
        foot_prim = stage.GetPrimAtPath(foot_path)
        xformable = UsdGeom.Xformable(foot_prim)
        world_tf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = world_tf.ExtractTranslation()
        return np.array([t[0], t[1], t[2]])

    def get_all_foot_positions(self) -> dict:
        return {i: self.get_foot_position(i) for i in range(4)}

    def get_foot_body_position(self, leg_idx: int) -> np.ndarray:
        foot_world = self.get_foot_position(leg_idx)
        body_pos, body_quat = self.robot.get_world_pose()
        w, x, y, z = body_quat[0], body_quat[1], body_quat[2], body_quat[3]
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])
        return R.T @ (foot_world - body_pos)

    def check_foot_errors(self) -> dict:
        result = {"positions": {}, "body_positions": {}, "ground_penetration": {}}
        for i in range(4):
            name = LEG_NAMES[i]
            pos = self.get_foot_position(i)
            body_pos = self.get_foot_body_position(i)
            result["positions"][name] = pos
            result["body_positions"][name] = body_pos
            result["ground_penetration"][name] = pos[2]
        return result

    def print_foot_status(self):
        errors = self.check_foot_errors()
        body_pos, body_quat = self.robot.get_world_pose()
        print(f"\n--- Body quat: [{body_quat[0]:+.4f}, {body_quat[1]:+.4f}, {body_quat[2]:+.4f}, {body_quat[3]:+.4f}] ---")

        artic_pos = self.robot.get_joint_positions()
        model_pos = self._articulation_to_model_order(artic_pos)
        for i in range(4):
            name = LEG_NAMES[i]
            idx = LEG_MODEL_INDICES[i]
            joints = model_pos[idx]
            print(f"  {name} joints: [{joints[0]:+.4f}, {joints[1]:+.4f}, {joints[2]:+.4f}]")

        print("--- Foot Status (body frame) ---")
        for name in LEG_NAMES:
            bpos = errors["body_positions"][name]
            print(f"  {name}: [{bpos[0]:+.4f}, {bpos[1]:+.4f}, {bpos[2]:+.4f}]")

        if self._vacuum_stepping_leg is not None:
            print(f"--- Vacuum: stepping leg = {LEG_NAMES[self._vacuum_stepping_leg]} ---")
        print("-------------------")

    def get_world_pose(self):
        return self.robot.get_world_pose()

    def get_joint_positions(self):
        return self.robot.get_joint_positions()

    def get_joint_velocities(self):
        return self.robot.get_joint_velocities()
