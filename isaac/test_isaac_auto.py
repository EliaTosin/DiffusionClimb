# SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone Isaac Sim script to run Aliengo with diffusion model control.

Run with:
    ~/.local/share/ov/pkg/isaac-sim-*/python.sh test_isaac.py
"""

import sys
import os
import select
# import tty
from ink_kin_stance.kinematics import QuadrupedKinematics, build_pinocchio_model, build_neutral_q

# Add the isaac directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaacsim import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp({"headless": False})

import numpy as np
from pxr import UsdPhysics, Gf
from isaacsim.core.api import World
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaac_aliengo import AlengoDiffusion
from ink_kin_stance.constants import LEG_NAMES, LEG_MODEL_INDICES
import scipy.spatial.transform as tf
import time
from still.still_diff_utils import set_seed

def quat_from_euler_rpy(roll, pitch, yaw, degrees=False):
    """Converts Euler XYZ to Quaternion (w, x, y, z)."""
    quat = tf.Rotation.from_euler("xyz", (roll, pitch, yaw), degrees=degrees).as_quat()
    return tuple(quat[[3, 0, 1, 2]].tolist())

class AliengoSimulation:
    def __init__(self):
        self._physics_rate = 200
        self._physics_dt = 1 / self._physics_rate

        # Create world with aligned physics and render update
        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=self._physics_dt,
            rendering_dt=1.0 / 60
            # rendering_dt=self._physics_dt
        )
        self._world.scene.add_default_ground_plane()

        stage = get_current_stage()
        physics_scene = UsdPhysics.Scene.Get(stage, "/physicsScene")
        # physics_scene.GetGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
        physics_scene.GetGravityDirectionAttr().Set(Gf.Vec3f(0, -1, 0))
        physics_scene.GetGravityMagnitudeAttr().Set(9.81)
        init_rot = np.array(quat_from_euler_rpy(0.0, 0.0, 90.0, degrees=True))

        self._aliengo = AlengoDiffusion(
            prim_path="/World/aliengo",
            usd_path="../Collected_aliengo/aliengo.usd",
            name="aliengo",
            position=np.array([0, 0, 0.3]),
            orientation=init_rot
        )

        # Pinocchio IK reference
        self._pin_model, self._pin_data, _, _ = build_pinocchio_model()
        self._kin = QuadrupedKinematics(self._pin_model)
        self._q_neutral = build_neutral_q(self._pin_model)
        self._kin.init_stance(self._q_neutral)

        # Diagnostic logging state
        self._step_log = None
        self._logging_active = False
        self._stepping_leg_for_log = 0
        self._all_steps_data = []

    def setup(self):
        """Initialize after world is ready."""
        self._world.reset()
        self._aliengo.initialize()
        self._aliengo._capture_vacuum_anchors(stepping_leg=None)
        self._world.add_physics_callback("aliengo_control", self.on_physics_step)

    def on_physics_step(self, step_size):
        """Physics callback — execute trajectory and log diagnostics."""
        self._aliengo.forward(step_size)

        # Log at decimation boundaries (when a new frame is applied)
        if self._logging_active and self._aliengo._sub_step == 1:
            self._log_frame()

    def _log_frame(self):
        """Capture one frame of diagnostic data."""
        if self._step_log is None:
            return

        a = self._aliengo

        # Diffusion target for this frame
        targets = a._next_target.copy() if a._next_target is not None else np.zeros(12)

        # Actual joint positions
        actuals = a._articulation_to_model_order(a.robot.get_joint_positions())

        # Body pose
        body_pos, body_quat = a.get_world_pose()

        # Foot world positions
        feet = [a.get_foot_position(i) for i in range(4)]

        # IK comparison: solve for current body position with feet anchored
        # try:
        q_pin = self._kin.set_body_pose(self._q_neutral.copy(), body_pos[:3])
        q_pin = self._kin.set_joint_angles(q_pin, actuals)
        q_ik, converged = self._kin.solve_stance(body_pos[:3], q_init=q_pin)
        ik_joints = self._kin.get_joint_angles(q_ik)

        # Joint velocities
        joint_vel = a._articulation_to_model_order(a.robot.get_joint_velocities())

        self._step_log["targets"].append(targets)
        self._step_log["actuals"].append(actuals)
        self._step_log["ik_joints"].append(ik_joints)
        self._step_log["body_pos"].append(body_pos[:3].copy())
        self._step_log["body_quat"].append(body_quat.copy())
        self._step_log["foot_pos"].append([f.copy() for f in feet])
        self._step_log["joint_vel"].append(joint_vel)

    def _print_step_summary(self, stepping_leg):
        """Print diagnostic summary after a step completes."""
        log = self._step_log
        if log is None or len(log["targets"]) == 0:
            return

        targets = np.array(log["targets"])
        actuals = np.array(log["actuals"])
        ik_joints = np.array(log["ik_joints"])
        body_pos = np.array(log["body_pos"])
        foot_pos = np.array(log["foot_pos"])  # (N, 4, 3)
        N = len(targets)

        print(f"\n{'=' * 70}")
        print(f"  STEP SUMMARY: {LEG_NAMES[stepping_leg]} ({N} frames)")
        print(f"{'=' * 70}")

        # 1. PD tracking error
        tracking_err = np.abs(actuals - targets)
        print(f"  PD tracking error:  mean={tracking_err.mean():.4f}  max={tracking_err.max():.4f} rad")

        # 2. Diffusion vs IK
        if not np.any(np.isnan(ik_joints)):
            diff_vs_ik = np.abs(targets - ik_joints)
            print(f"  Diffusion vs IK:    mean={diff_vs_ik.mean():.4f}  max={diff_vs_ik.max():.4f} rad")
        else:
            diff_vs_ik = None
            print(f"  Diffusion vs IK:    (IK failed)")

        # 3. Smoothness
        if N > 1:
            frame_deltas = np.diff(targets, axis=0)
            print(f"  Smoothness (delta): mean={np.abs(frame_deltas).mean():.4f}  max={np.abs(frame_deltas).max():.4f} rad/frame")
        else:
            frame_deltas = np.zeros((0, 12))

        # 4. Body displacement
        total_disp = body_pos[-1] - body_pos[0]
        print(f"  Body displacement:  [{total_disp[0]:+.4f}, {total_disp[1]:+.4f}, {total_disp[2]:+.4f}] m")

        # 5. Per-leg detail
        print(f"\n  {'Leg':4s} {'Track err':>10s} {'Diff vs IK':>11s} {'Smoothness':>11s} {'Foot drift':>11s}")
        print(f"  {'-'*4} {'-'*10} {'-'*11} {'-'*11} {'-'*11}")
        for leg in range(4):
            ji = slice(leg * 3, leg * 3 + 3)
            te = tracking_err[:, ji].mean()
            sm = np.abs(frame_deltas[:, ji]).mean() if N > 1 else 0.0
            drift = np.linalg.norm(foot_pos[-1, leg] - foot_pos[0, leg]) * 1000

            dik_str = f"{diff_vs_ik[:, ji].mean():.4f}" if diff_vs_ik is not None else "N/A"
            marker = " (Active Swing)" if leg == stepping_leg else ""
            print(f"  {LEG_NAMES[leg]:4s} {te:10.4f} {dik_str:>11s} {sm:11.4f} {drift:8.1f} mm{marker}")
        # 7. Compute IK-based reference trajectory for comparison
        # This is what the joints SHOULD be if we used pure IK instead of diffusion
        ik_walk_joints = self._compute_ik_walk(stepping_leg, log)

        # 8. Save all data to file
        step_data = {
            "stepping_leg": stepping_leg, # index of which leg is stepping
            "targets": targets, # diff prediction (with interpolation -> 58 steps)
            "actuals": actuals, # joint pos from isaacsim
            "ik_joints": ik_joints,
            "ik_walk_joints": ik_walk_joints, # ik prediction
            "body_pos": body_pos,
            "body_quat": np.array(log["body_quat"]),
            "foot_pos": foot_pos,
            "joint_vel": np.array(log["joint_vel"]),
            "raw_trajectory": log["raw_trajectory"],
            "delta_body": log["delta_body"],
            "delta_foot": log["delta_foot"],
        }
        self._all_steps_data.append(step_data)
        self._save_log_file()
        print(f"{'=' * 70}\n")

    def _compute_ik_walk(self, stepping_leg, log):
        """Compute a pure-IK reference trajectory for comparison.

        Uses Pinocchio IK to find joint angles that:
        - Keep support feet at their initial positions
        - Move the stepping foot along a cycloid path to the target
        - Interpolate the body position linearly
        """
        body_pos_arr = np.array(log["body_pos"])
        foot_pos_arr = np.array(log["foot_pos"])
        delta_foot = log["delta_foot"]
        delta_body = log["delta_body"]
        N = len(body_pos_arr)

        if N == 0:
            return np.zeros((0, 12))

        # Starting config
        start_body = body_pos_arr[0]
        start_joints = np.array(log["actuals"])[0]

        # Foot start positions (from first frame)
        foot_starts = foot_pos_arr[0]  # (4, 3)

        # Target foot position for stepping leg
        step_foot_start = foot_starts[stepping_leg].copy()
        # delta_foot is in body frame convention (negative X = forward in model space)
        step_foot_target = step_foot_start + np.array([delta_foot[0], delta_foot[1], delta_foot[2]])

        # Generate IK trajectory frame by frame
        ik_joints_list = []
        q_pin = self._kin.set_body_pose(self._q_neutral.copy(), start_body)
        q_pin = self._kin.set_joint_angles(q_pin, start_joints)

        # Store support feet positions for IK
        self._kin.update_feet_positions(q_pin)
        support_feet = [self._kin.feet_world_positions[i].copy() for i in range(4)]

        # Step height for cycloid
        step_height = 0.03

        for f in range(N):
            progress = f / max(N - 1, 1)

            # Interpolate body position
            body_target = start_body + progress * np.array([delta_body[0], delta_body[1], delta_body[2]])

            # Set body pose
            q_pin = self._kin.set_body_pose(q_pin, body_target)

            # Restore support feet targets
            for i in range(4):
                self._kin.feet_world_positions[i] = support_feet[i].copy()

            # Stepping foot follows cycloid
            theta = progress * 2 * np.pi
            cycloid_x = (theta - np.sin(theta)) / (2 * np.pi)
            cycloid_z = (1 - np.cos(theta)) / 2
            step_pos = step_foot_start + (step_foot_target - step_foot_start) * cycloid_x
            step_pos[2] += step_height * cycloid_z
            self._kin.feet_world_positions[stepping_leg] = step_pos

            # Solve IK for all legs
            q_pin, converged = self._kin.solve_stance(body_target, q_init=q_pin)
            if not converged:
                # Use last good solution
                pass

            ik_joints_list.append(self._kin.get_joint_angles(q_pin))

        return np.array(ik_joints_list)

    def _save_log_file(self):
        """Save all accumulated step data to a .npz file."""
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walk_log.npz")

        save_dict = {}
        for i, step in enumerate(self._all_steps_data):
            prefix = f"step{i}_"
            save_dict[prefix + "stepping_leg"] = np.array(step["stepping_leg"])
            save_dict[prefix + "targets"] = step["targets"]
            save_dict[prefix + "actuals"] = step["actuals"]
            save_dict[prefix + "ik_joints"] = step["ik_joints"]
            save_dict[prefix + "ik_walk_joints"] = step["ik_walk_joints"]
            save_dict[prefix + "body_pos"] = step["body_pos"]
            save_dict[prefix + "body_quat"] = step["body_quat"]
            save_dict[prefix + "foot_pos"] = step["foot_pos"]
            save_dict[prefix + "joint_vel"] = step["joint_vel"]
            save_dict[prefix + "raw_trajectory"] = step["raw_trajectory"]
            save_dict[prefix + "delta_body"] = step["delta_body"]
            save_dict[prefix + "delta_foot"] = step["delta_foot"]
        save_dict["num_steps"] = np.array(len(self._all_steps_data))

        np.savez(save_path, **save_dict)
        print(f"  Log saved to {save_path} ({len(self._all_steps_data)} steps)")

    def _execute_advance(self, current_leg):
        """Step one leg using combined trunk + step diffusion models."""
        # Both models use negative X = forward in Isaac Sim.
        # Trunk moves 1/4 of footstep.
        delta_foot = np.array([-0.1, 0.0, 0.0])
        delta_body = np.array([-0.0, 0.0, 0.0])

        artic_positions = self._aliengo.get_joint_positions()
        current_joints = self._aliengo._articulation_to_model_order(artic_positions)

        print(f"Generating combined trajectory: body={delta_body}, foot={delta_foot}, leg={LEG_NAMES[current_leg]}")
        trajectory = self._aliengo.generate_combined_trajectory(
            delta_body, delta_foot, current_joints, current_leg, ddim_steps=5,
        )
        self._aliengo.set_combined_trajectory(trajectory, stepping_leg=current_leg)
        print(f"Combined trajectory: {trajectory.shape[0]} steps")

        # Start diagnostic logging
        self._step_log = {
            "targets": [], "actuals": [], "ik_joints": [],
            "body_pos": [], "body_quat": [], "foot_pos": [],
            "joint_vel": [],
            "raw_trajectory": trajectory.copy(),
            "delta_body": delta_body.copy(),
            "delta_foot": delta_foot.copy(),
        }
        self._stepping_leg_for_log = current_leg
        self._logging_active = True

        # Capture IK reference feet positions at step start
        body_pos, _ = self._aliengo.get_world_pose()
        q_start = self._kin.set_body_pose(self._q_neutral.copy(), body_pos[:3])
        q_start = self._kin.set_joint_angles(q_start, current_joints)
        self._kin.update_feet_positions(q_start)

    def _execute_reset(self):
        """Reset robot to starting state."""
        self._aliengo._current_trajectory = None
        self._aliengo._held_positions = None
        self._aliengo._prev_target = None
        self._aliengo._next_target = None
        for i in range(4):
            self._aliengo._step_trajectories[i] = None
        self._aliengo.release_vacuum()

        self._aliengo.robot.set_world_pose(self._start_pos, self._start_rot)
        default_model = np.array([0.0, 0.8, -1.5] * 4)
        default_artic = self._aliengo._model_to_articulation_order(default_model)
        self._aliengo.robot.set_joint_positions(default_artic)
        self._aliengo.robot.set_joint_velocities(np.zeros(12))

        # Reset IK reference
        self._kin.init_stance(self._q_neutral)

        self._logging_active = False
        self._step_log = None
        print("Reset to starting configuration.")

    def _trajectories_active(self):
        trunk_active = self._aliengo._current_trajectory is not None
        steps_active = any(t is not None for t in self._aliengo._step_trajectories.values())
        return trunk_active or steps_active

    def run(self, mode="planar"):
        """Main simulation loop with automated step execution"""
        self.setup()

        if mode == "planar":
            set_camera_view(eye=np.array([0.0, 0.5, 3]), target=np.array([0.0, 0.5, 0]))
        elif mode == "isometric":
            distance = 2.0
            set_camera_view(
                eye=np.array([0.0 + distance, 0.5 + distance, 0.0 + distance]),  # Posizione della camera [X, Y, Z]
                target=np.array([0.0, 0.5, 0.0])  # Centro del robot [X, Y, Z]
            )
        elif mode == "lateral":
            set_camera_view(eye=np.array([3, 0.3, 0.2]), target=np.array([0, 0.3, 0.2]))

        # Initial warmup
        for _ in range(500):
            self._world.step(render=True)

        self._start_pos, self._start_rot = self._aliengo.get_world_pose()

        current_leg = 0
        state = "idle"

        # Intervals
        STEP_PAUSE_SEC = 0.15 # Time between steps
        SETTLING_FRAMES = 10
        settling_counter = 0

        last_step_sim_time = self._world.current_time
        set_seed()

        print(f"\n=== Fast Automated Step Execution (Pause: {STEP_PAUSE_SEC}s) ===")
        print(f"Starting with leg: {LEG_NAMES[current_leg]}")
        print("=================================================================\n")

        try:
            while simulation_app.is_running():
                self._world.step(render=True)
                current_sim_time = self._world.current_time

                if state == "idle":
                    if current_sim_time - last_step_sim_time >= STEP_PAUSE_SEC:
                        print(f"[AUTO] Stepping: {LEG_NAMES[current_leg]}")
                        self._execute_advance(current_leg)
                        state = "advancing"

                elif state == "advancing":
                    if not self._trajectories_active():
                        self._logging_active = False
                        state = "settling"
                        settling_counter = 0

                elif state == "settling":
                    settling_counter += 1
                    if settling_counter >= SETTLING_FRAMES:
                        # self._print_step_summary(self._stepping_leg_for_log)
                        current_leg = (current_leg + 1) % 4
                        state = "idle"
                        last_step_sim_time = self._world.current_time

        except KeyboardInterrupt:
            print("\nSimulation stopped by user (Ctrl+C).")

        finally:
            simulation_app.close()


def main():
    sim = AliengoSimulation()
    sim.run(mode="lateral")


if __name__ == "__main__":
    main()