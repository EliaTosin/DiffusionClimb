from triton.language import dtype


def get_isaaclab_diffusion_model():
    """
        Factory function for Aliengo robot with diffusion model with dynamic import after SimulationApp is called.
    """
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../isaac/")))
    from isaac_aliengo import load_diffusion_model, AlengoDiffusion


    from typing import Optional
    import numpy as np
    import torch

    # Numpy compatibility fix for models saved with newer numpy
    # Maps numpy._core to numpy.core for older numpy versions
    if not hasattr(np, '_core'):
        sys.modules['numpy._core'] = np.core

    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

    class AliengoIsaaclabDiffusion:
        MODEL_JOINT_ORDER = AlengoDiffusion.MODEL_JOINT_ORDER
        ARTICULATION_JOINT_ORDER = AlengoDiffusion.ARTICULATION_JOINT_ORDER
        MODEL_TO_ARTICULATION = AlengoDiffusion.MODEL_TO_ARTICULATION
        ARTICULATION_TO_MODEL = AlengoDiffusion.ARTICULATION_TO_MODEL
        LEG_FL, LEG_FR, LEG_RL, LEG_RR = 0, 1, 2, 3
        LEG_MODEL_INDICES = AlengoDiffusion.LEG_MODEL_INDICES
        LEG_IS_RIGHT = AlengoDiffusion.LEG_IS_RIGHT
        FOOT_NAMES = AlengoDiffusion.FOOT_NAMES
        LEG_NAMES = AlengoDiffusion.LEG_NAMES

        def __init__(
                self,
                env : DirectRLEnv | ManagerBasedRLEnv,
                diffusion_model_path: Optional[str] = None,
                step_model_path: Optional[str] = None,
        ) -> None:
            """
            Initialize Aliengo robot with diffusion model controller.

            Args:
                env: Aliengo environment
                diffusion_model_path: Path to still/trunk diffusion model checkpoint
                step_model_path: Path to step/FL leg diffusion model checkpoint
            """
            self.env = env
            self.DEFAULT_JOINT_POS_TENSOR = self.env.scene.articulations['robot'].data.default_joint_pos
            self.DEFAULT_JOINT_POS = self.DEFAULT_JOINT_POS_TENSOR.cpu().tolist()
            self.target_pos = self.DEFAULT_JOINT_POS_TENSOR.clone()
            self._foot_prim_ids, self._foot_prim_names = self.env.scene.articulations['robot'].find_bodies(self.FOOT_NAMES)


            # Load still/trunk diffusion model
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            diffusion_model_path = os.path.join(project_root, "still", "diffusion_model_compat.pt")

            self._model, self._diffusion, self._checkpoint = load_diffusion_model(
                diffusion_model_path, device=self.env.device
            )

            # Load step/FL leg diffusion model
            if step_model_path is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                step_model_path = os.path.join(project_root, "step", "diffusion_model_compat.pt")

            self._step_model, self._step_diffusion, self._step_checkpoint = load_diffusion_model(
                step_model_path, device=self.env.device
            )

            # Trunk trajectory state
            self._current_trajectory = None
            self._trajectory_step = 0
            self._num_trajectory_steps = self._checkpoint["num_steps"]

            # Per-leg step trajectory state: {leg_idx: trajectory or None}
            self._step_trajectories = {i: None for i in range(4)}
            self._step_trajectory_steps = {i: 0 for i in range(4)}

            # Control parameters
            self._decimation = 5  # decimation for diffusion output step
            self._step_counter = 0

            # Held joint positions: snapshot taken when a trajectory starts,
            # used as persistent targets for non-active legs to prevent drift
            self._held_positions = None

            # Foot tracking state
            self._foot_start_positions = {i: None for i in range(4)}  # captured when step begins
            self._foot_step_deltas = {i: None for i in range(4)}  # requested delta per leg

        def generate_trajectory(self, delta: np.ndarray, current_joints: np.ndarray, ddim_steps: int = 50) -> np.ndarray:
            """
            Generate a joint trajectory using the STILL diffusion model.

            Args:
                delta_pos: Delta position [goal_x - start_x, goal_y - start_y, goal_z - start_z]
                current_joints: Current joint angles in model order (num_joints,)
                ddim_steps: Number of DDIM denoising steps (default 50)

            Returns:
                trajectory: (num_steps, num_joints) array of joint positions
            """
            # Normalize inputs (handle both list and numpy array formats)
            delta_mean = np.array(self._checkpoint["delta_mean"], dtype=np.float32)
            delta_std = np.array(self._checkpoint["delta_std"], dtype=np.float32)
            traj_mean = np.array(self._checkpoint["traj_mean"], dtype=np.float32)
            traj_std = np.array(self._checkpoint["traj_std"], dtype=np.float32)

            # Clamp input joints to training range (mean ± 2*std)
            current_joints_clamped = np.clip(
                np.array(current_joints, dtype=np.float32),
                traj_mean - 2 * traj_std, traj_mean + 2 * traj_std
            )

            delta_pos_norm = (np.array(delta, dtype=np.float32) - delta_mean) / delta_std
            current_joints_norm = (current_joints_clamped - traj_mean) / traj_std

            condition = torch.tensor(
                np.concatenate([delta_pos_norm, current_joints_norm]), dtype=torch.float32
            ).unsqueeze(0).to(self.env.device)

            shape = (1, self._checkpoint["num_steps"], self._checkpoint["num_joints"])
            trajectory = self._diffusion.ddim_sample(self._model, condition, shape, ddim_steps=ddim_steps)

            # Denormalize
            trajectory = trajectory.cpu().numpy()[0]
            trajectory = trajectory * traj_std + traj_mean

            # Clamp output to training range (mean ± 2*std)
            trajectory = np.clip(trajectory, traj_mean - 2 * traj_std, traj_mean + 2 * traj_std)

            return trajectory

        def set_trajectory(self, trajectory: np.ndarray):
            """Set a new trajectory to execute."""
            self._current_trajectory = trajectory
            self._trajectory_step = 0
            self._capture_held_positions()

        def generate_leg_step_trajectory(self, leg_idx: int, delta_foot_pos: np.ndarray, current_leg_joints: np.ndarray, ddim_steps: int = 50) -> np.ndarray:
            """
            Generate a leg trajectory using the step diffusion model (trained on FL).
            Reuses the same model for all legs by mirroring inputs/outputs for right-side legs.

            Args:
                leg_idx: Leg index (0=FL, 1=FR, 2=RL, 3=RR)
                delta_foot_pos: Foot position delta in body frame [dx, dy, dz]
                current_leg_joints: Current leg joint angles (3,) [hip, thigh, calf]
                ddim_steps: Number of DDIM denoising steps

            Returns:
                trajectory: (num_steps, 3) array of leg joint positions
            """
            delta_mean = np.array(self._step_checkpoint["delta_mean"], dtype=np.float32)
            delta_std = np.array(self._step_checkpoint["delta_std"], dtype=np.float32)
            traj_mean = np.array(self._step_checkpoint["traj_mean"], dtype=np.float32)
            traj_std = np.array(self._step_checkpoint["traj_std"], dtype=np.float32)

            delta = np.array(delta_foot_pos, dtype=np.float32).copy()
            joints = np.array(current_leg_joints, dtype=np.float32).copy()

            # Clamp deltas to training range (start/goal offsets ±0.10/±0.05/±0.10 → max delta ±0.20/±0.10/±0.20)
            delta_limits = np.array([0.10, 0.10, 0.10])
            delta = np.clip(delta, -delta_limits, delta_limits)

            # Mirror for right-side legs: negate y in delta, negate hip angle
            if self.LEG_IS_RIGHT[leg_idx]:
                delta[1] = -delta[1]
                joints[0] = -joints[0]

            # Clamp input joints to training range (mean ± 2*std) to avoid out-of-distribution
            joints = np.clip(joints, traj_mean - 2 * traj_std, traj_mean + 2 * traj_std)

            delta_pos_norm = (delta - delta_mean) / delta_std
            current_joints_norm = (joints - traj_mean) / traj_std

            print(f"  Step model input: delta={delta} joints={joints}")
            print(f"  Step model norms: delta_mean={delta_mean} delta_std={delta_std}")
            print(f"  Step model norms: traj_mean={traj_mean} traj_std={traj_std}")
            print(f"  Step model normalized: delta_norm={delta_pos_norm} joints_norm={current_joints_norm}")

            condition = torch.tensor(
                np.concatenate([delta_pos_norm, current_joints_norm]), dtype=torch.float32
            ).unsqueeze(0).to(self.env.device)

            shape = (1, self._step_checkpoint["num_steps"], self._step_checkpoint["num_joints"])
            trajectory = self._step_diffusion.ddim_sample(self._step_model, condition, shape, ddim_steps=ddim_steps)

            trajectory = trajectory.cpu().numpy()[0]
            trajectory = trajectory * traj_std + traj_mean

            # Clamp output to training range (mean ± 2*std) to prevent divergence
            traj_min = traj_mean - 2 * traj_std
            traj_max = traj_mean + 2 * traj_std
            trajectory = np.clip(trajectory, traj_min, traj_max)

            # Mirror output hip angles back for right-side legs
            if self.LEG_IS_RIGHT[leg_idx]:
                trajectory[:, 0] = -trajectory[:, 0]

            print(f"  Step model output: first={trajectory[0]} last={trajectory[-1]}")

            return trajectory

        def set_leg_step_trajectory(self, leg_idx: int, trajectory: np.ndarray, delta_foot_pos: Optional[np.ndarray] = None):
            """Set a new step trajectory for a specific leg."""
            self._step_trajectories[leg_idx] = trajectory
            self._step_trajectory_steps[leg_idx] = 0
            self._capture_held_positions()
            # Capture foot start position in body frame and requested delta for error tracking
            self._foot_start_positions[leg_idx] = self.get_foot_body_position(leg_idx)
            self._foot_step_deltas[leg_idx] = delta_foot_pos

        def _capture_held_positions(self):
            """Snapshot current joint positions if not already held."""
            if self._held_positions is None:
                artic_positions = self.env.scene.articulations['robot'].data.joint_pos.flatten().cpu().numpy()
                self._held_positions = self._articulation_to_model_order(artic_positions)

        def _update_held_for_leg(self, leg_idx):
            """Update held positions for a specific leg after its step completes."""
            artic_positions = self.env.scene.articulations['robot'].data.joint_pos.cpu().numpy()
            current = self._articulation_to_model_order(artic_positions)
            for mi in self.LEG_MODEL_INDICES[leg_idx]:
                self._held_positions[mi] = current[mi]

        def forward(self, dt: float, delta_pos: Optional[np.ndarray] = None):
            """
            Execute one step of the controller.

            Args:
                dt: Physics timestep
                delta_pos: If provided, generate new trajectory with this delta position
            """
            self._step_counter += 1

            # Generate new trajectory if delta_pos provided and no trajectory active
            if delta_pos is not None and self._current_trajectory is None:
                # Get current joint positions and convert to model order
                artic_positions = self.env.scene.articulations['robot'].data.joint_pos.cpu().numpy()
                current_joints = self._articulation_to_model_order(artic_positions)
                self._current_trajectory = self.generate_trajectory(delta_pos, current_joints)
                self._trajectory_step = 0

            any_step_active = any(t is not None for t in self._step_trajectories.values())
            if self._current_trajectory is None and not any_step_active:
                return

            # Update trajectory step at decimation rate
            if self._step_counter % self._decimation == 0:
                has_trunk = (self._current_trajectory is not None
                             and self._trajectory_step < len(self._current_trajectory))

                # Start from held positions (snapshot from trajectory start) to prevent drift
                target_pos = self._held_positions.copy()

                if has_trunk:
                    trunk_pos = self._current_trajectory[self._trajectory_step].copy()
                    self._trajectory_step += 1
                    if self._trajectory_step >= len(self._current_trajectory):
                        self._current_trajectory = None

                    # Apply trunk targets clamped to small delta from held positions
                    # This allows body shifting but prevents cumulative drift
                    max_trunk_delta = 0.05  # radians
                    for i in range(12):
                        delta = trunk_pos[i] - self._held_positions[i]
                        delta = np.clip(delta, -max_trunk_delta, max_trunk_delta)
                        target_pos[i] = self._held_positions[i] + delta

                # Override each active leg's joints from its step trajectory
                for leg_idx in range(4):
                    traj = self._step_trajectories[leg_idx]
                    step = self._step_trajectory_steps[leg_idx]
                    if traj is not None and step < len(traj):
                        leg_joints = traj[step]
                        for i, mi in enumerate(self.LEG_MODEL_INDICES[leg_idx]):
                            target_pos[mi] = leg_joints[i]
                        self._step_trajectory_steps[leg_idx] = step + 1
                        if step + 1 >= len(traj):
                            self._step_trajectories[leg_idx] = None
                            # Update anchor for this leg to its post-step position
                            self._update_held_for_leg(leg_idx)

                self._apply_joint_positions(target_pos)

        def _model_to_articulation_order(self, model_positions: np.ndarray) -> np.ndarray:
            """Convert joint positions from model order to articulation order."""
            articulation_positions = np.zeros(12)
            for model_idx, artic_idx in enumerate(self.MODEL_TO_ARTICULATION):
                articulation_positions[artic_idx] = model_positions[model_idx]
            return articulation_positions

        def _articulation_to_model_order(self, articulation_positions: np.ndarray) -> np.ndarray:
            """Convert joint positions from articulation order to model order."""
            model_positions = np.zeros(12)
            for artic_idx, model_idx in enumerate(self.ARTICULATION_TO_MODEL):
                model_positions[model_idx] = articulation_positions[artic_idx]
            return model_positions

        def _apply_joint_positions(self, target_positions: np.ndarray):
            """Apply target joint positions using position control.

            Args:
                target_positions: Joint positions in MODEL order (from diffusion model)
            """
            # Convert from model order to articulation order
            articulation_positions = self._model_to_articulation_order(target_positions)
            # self.robot.apply_action(ArticulationAction(joint_positions=articulation_positions))
            articulation_positions = torch.tensor(articulation_positions, dtype=torch.float, device=self.env.device)
            self.env.scene.articulations['robot'].set_joint_position_target(articulation_positions, self._all_joints)
            # self.target_pos = articulation_positions

        def initialize(self, physics_sim_view=None) -> None:
            # """Initialize the robot articulation."""
            # self.robot.initialize(physics_sim_view)
            #
            # Print joint information for debugging
            self.print_joint_info()

            # Discover foot prim paths by traversing the USD stage
            # self._foot_prim_paths = {}
            # stage = get_current_stage()
            # for prim in stage.Traverse():
            #     name = prim.GetName()
            #     for i, foot_name in enumerate(self.FOOT_NAMES):
            #         if name == foot_name and str(prim.GetPath()).startswith(self.prim_path):
            #             self._foot_prim_paths[i] = str(prim.GetPath())

            print(f"\nFoot prim paths:")
            for i, path in enumerate(self._foot_prim_names):
                print(f"  {self.LEG_NAMES[i]}: {path}")

            # # Set joint PD gains for stable position tracking
            # kps = np.array([60.0] * 12)
            # kds = np.array([2.0] * 12)
            # self.robot.get_articulation_controller().set_gains(kps, kds)
            #
            # # Set default standing pose as the default state
            # self.robot.set_joints_default_state(positions=self.DEFAULT_JOINT_POS)
            # self.robot.post_reset()

        def print_joint_info(self):
            """Print articulation joint names and ordering for debugging."""
            print("\n" + "=" * 60)
            print("ARTICULATION JOINT INFORMATION")
            print("=" * 60)

            # Get DOF names from the articulation
            # dof_names = self.robot.dof_names
            # num_dof = self.robot.num_dof
            self._all_joints, dof_names = self.env.scene.articulations['robot'].find_joints([".*"])
            num_dof = self.env.scene.articulations['robot'].num_joints

            print(f"Number of DOFs: {num_dof}")
            print(f"\nActual articulation joint order:")
            print("-" * 40)
            for i, name in enumerate(dof_names):
                print(f"  [{i:2d}] {name}")

            print(f"\nExpected articulation order (ARTICULATION_JOINT_ORDER):")
            print("-" * 40)
            for i, name in enumerate(self.ARTICULATION_JOINT_ORDER):
                print(f"  [{i:2d}] {name}")

            # Check if orderings match
            print(f"\nOrdering verification:")
            print("-" * 40)
            matches = True
            for i, (actual, expected) in enumerate(zip(dof_names, self.ARTICULATION_JOINT_ORDER)):
                status = "✓" if actual == expected else "✗ MISMATCH"
                if actual != expected:
                    matches = False
                print(f"  [{i:2d}] {actual:20s} vs {expected:20s} {status}")

            if len(dof_names) != len(self.ARTICULATION_JOINT_ORDER):
                print(f"\n⚠ WARNING: DOF count mismatch! Articulation has {len(dof_names)}, expected {len(self.ARTICULATION_JOINT_ORDER)}")
                matches = False

            if matches:
                print(f"\n✓ Articulation matches expected ordering!")
            else:
                print(f"\n✗ Articulation DOES NOT match expected - check ARTICULATION_JOINT_ORDER!")

            print(f"\nModel-to-Articulation mapping:")
            print("-" * 40)
            for model_idx, artic_idx in enumerate(self.MODEL_TO_ARTICULATION):
                print(f"  Model[{model_idx:2d}] {self.MODEL_JOINT_ORDER[model_idx]:20s} -> Artic[{artic_idx:2d}] {self.ARTICULATION_JOINT_ORDER[artic_idx]}")

            print("=" * 60 + "\n")

        def get_foot_position(self, leg_idx: int) -> np.ndarray:
            """Get world position of a foot using USD transforms."""
            # stage = get_current_stage()
            # foot_path = self._foot_prim_paths[leg_idx]
            # foot_prim = stage.GetPrimAtPath(foot_path)
            # xformable = UsdGeom.Xformable(foot_prim)
            # world_tf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            # t = world_tf.ExtractTranslation()
            # return np.array([t[0], t[1], t[2]])

            # takes the 3 coord XYZ of the chosen foot from the first (and only) env
            return self.env.scene.articulations['robot'].data.body_com_pos_w[0, self._foot_prim_ids[leg_idx], :].cpu().numpy()

        def get_all_foot_positions(self) -> dict:
            """Get world positions of all 4 feet. Returns {leg_idx: np.array([x,y,z])}."""
            return {i: self.get_foot_position(i) for i in range(4)}

        def get_foot_body_position(self, leg_idx: int) -> np.ndarray:
            """Get foot position in body frame."""
            # foot_world = self.get_foot_position(leg_idx)
            # body_pos, body_quat = self.robot.get_world_pose()
            # # body_quat is [w, x, y, z]
            # w, x, y, z = body_quat[0], body_quat[1], body_quat[2], body_quat[3]
            # # Rotation matrix from quaternion
            # R = np.array([
            #     [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            #     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            #     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            # ])
            # # Transform world → body: R^T * (foot_world - body_pos)
            # return R.T @ (foot_world - body_pos)

            return self.env.scene.articulations['robot'].data.body_com_pos_b[0, self._foot_prim_ids[leg_idx], :].cpu().numpy()

        def check_foot_errors(self) -> dict:
            """
            Check foot positions for errors. Returns a dict with:
            - positions: {leg_name: [x, y, z]} world positions
            - body_positions: {leg_name: [x, y, z]} body-frame positions
            - ground_penetration: {leg_name: z} (negative = below ground)
            - step_errors: {leg_name: {actual_delta, requested_delta, error}} for completed steps
            """
            result = {"positions": {}, "body_positions": {}, "ground_penetration": {}, "step_errors": {}}

            for i in range(4):
                name = self.LEG_NAMES[i]
                pos = self.get_foot_position(i)
                body_pos = self.get_foot_body_position(i)
                result["positions"][name] = pos
                result["body_positions"][name] = body_pos
                result["ground_penetration"][name] = pos[2]

            return result

        def print_foot_status(self):
            """Print current foot positions and any errors."""
            errors = self.check_foot_errors()

            # Print body orientation to detect rotation during trajectories
            body_pos = self.env.scene.articulations['robot'].data.root_pos_w.flatten().cpu().tolist()
            body_quat = self.env.scene.articulations['robot'].data.root_quat_w.flatten().cpu().tolist()
            print(f"\n--- Body quat: [{body_quat[0]:+.4f}, {body_quat[1]:+.4f}, {body_quat[2]:+.4f}, {body_quat[3]:+.4f}] ---")

            # Print actual joint angles vs model order
            artic_pos = self.env.scene.articulations['robot'].data.joint_pos.flatten().cpu().tolist()
            model_pos = self._articulation_to_model_order(artic_pos)
            for i in range(4):
                name = self.LEG_NAMES[i]
                idx = self.LEG_MODEL_INDICES[i]
                joints = model_pos[idx]
                print(f"  {name} joints: [{joints[0]:+.4f}, {joints[1]:+.4f}, {joints[2]:+.4f}]")

            print("--- Foot Status (body frame) ---")
            for name in self.LEG_NAMES:
                bpos = errors["body_positions"][name]
                print(f"  {name}: [{bpos[0]:+.4f}, {bpos[1]:+.4f}, {bpos[2]:+.4f}]")

            # Report step trajectory results for legs that just finished (body-frame comparison)
            for i in range(4):
                name = self.LEG_NAMES[i]
                if (self._foot_start_positions[i] is not None
                        and self._step_trajectories[i] is None
                        and self._foot_step_deltas[i] is not None):
                    start_body = self._foot_start_positions[i]
                    current_body = errors["body_positions"][name]
                    actual_delta = current_body - start_body
                    requested = self._foot_step_deltas[i]
                    err = actual_delta - requested
                    print(f"  {name} step result (body frame):"
                          f" requested=[{requested[0]:+.4f}, {requested[1]:+.4f}, {requested[2]:+.4f}]"
                          f" actual=[{actual_delta[0]:+.4f}, {actual_delta[1]:+.4f}, {actual_delta[2]:+.4f}]"
                          f" error=[{err[0]:+.4f}, {err[1]:+.4f}, {err[2]:+.4f}] |err|={np.linalg.norm(err):.4f}")
                    # Clear after reporting
                    self._foot_start_positions[i] = None
                    self._foot_step_deltas[i] = None
            print("-------------------")

        def get_world_pose(self):
            """Get robot base world pose."""
            return self.env.scene.articulations['robot'].data.root_pos_w.flatten().cpu().numpy(), self.env.scene.articulations['robot'].data.root_quat_w.flatten().cpu().numpy()

        def get_joint_positions(self):
            """Get current joint positions."""
            return self.env.scene.articulations['robot'].data.joint_pos.flatten().cpu().numpy()

        def get_joint_velocities(self):
            """Get current joint velocities."""
            return self.env.scene.articulations['robot'].data.joint_vel.flatten().cpu().numpy()

    return AliengoIsaaclabDiffusion