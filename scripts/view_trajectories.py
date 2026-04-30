#!/usr/bin/env python3
"""Interactive visualisation of stored trajectory data."""

import argparse
import time

import numpy as np
import zarr
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from ink_kin_stance.kinematics import (
    QuadrupedKinematics, build_pinocchio_model, build_neutral_q, rotation_rpy,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_path", nargs="?", default="still/trajectory_log.zarr")
    args = parser.parse_args()

    root = zarr.open_group(args.zarr_path, mode="r")
    start_positions = root["start_positions"]
    goal_positions = root["goal_positions"]
    joint_angles = root["joint_angles"]
    num_traj = start_positions.shape[0]
    num_steps = joint_angles.shape[1]

    has_rpy = "start_rpys" in root
    if has_rpy:
        start_rpys = root["start_rpys"]
        goal_rpys = root["goal_rpys"]

    print(f"Loaded {num_traj} trajectories ({num_steps} steps each)")

    pin_model, _, collision_model, visual_model = build_pinocchio_model()
    q_base = build_neutral_q(pin_model)
    kin = QuadrupedKinematics(pin_model)

    viz = MeshcatVisualizer(pin_model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    print("Meshcat viewer opened.\n")
    time.sleep(1)

    dt = 0.05
    idx = 0
    while idx < num_traj:
        start_pos = np.array(start_positions[idx])
        goal_pos = np.array(goal_positions[idx])
        traj = np.array(joint_angles[idx])

        print(f"--- Trajectory {idx}/{num_traj} ---")
        print(f"  Delta pos: {np.round(goal_pos - start_pos, 4)}")

        if has_rpy:
            s_rpy = np.array(start_rpys[idx])
            g_rpy = np.array(goal_rpys[idx])
            print(f"  Start RPY: {np.round(np.degrees(s_rpy), 2)} deg")
            print(f"  Goal  RPY: {np.round(np.degrees(g_rpy), 2)} deg")

        for step in range(num_steps):
            alpha = step / (num_steps - 1) if num_steps > 1 else 1.0
            body_pos = start_pos + alpha * (goal_pos - start_pos)

            q = q_base.copy()
            q[0:3] = body_pos

            if has_rpy:
                rpy = s_rpy + alpha * (g_rpy - s_rpy)
                R = rotation_rpy(*rpy)
                quat = pin.Quaternion(R)
                q[3:7] = np.array([quat.x, quat.y, quat.z, quat.w])

            q = kin.set_joint_angles(q, traj[step])
            viz.display(q)
            time.sleep(dt)

        try:
            user = input("\n  [Enter] next | [number] jump | [r] replay | [q] quit: ").strip()
            if user == "q":
                break
            elif user == "r":
                continue
            elif user.isdigit():
                idx = int(user)
                continue
            else:
                idx += 1
        except (KeyboardInterrupt, EOFError):
            break

    print("Done.")


if __name__ == "__main__":
    main()
