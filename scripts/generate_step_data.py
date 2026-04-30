#!/usr/bin/env python3
"""Generate FL leg step trajectory data for diffusion training."""

import random
import numpy as np
import zarr
from multiprocessing import Pool, cpu_count

from ink_kin_stance.constants import (
    URDF_PATH, FOOT_FRAMES, NUM_TRAJ_STEPS, STEP_HEIGHT, CHOSEN_LEG,
)
from ink_kin_stance.kinematics import QuadrupedKinematics, build_neutral_q
import pinocchio as pin


_worker_kin = None
_worker_q_neutral = None
_worker_foot_init_body = None


def init_worker():
    global _worker_kin, _worker_q_neutral, _worker_foot_init_body

    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    kin = QuadrupedKinematics(model, FOOT_FRAMES)
    q_neutral = build_neutral_q(model)
    kin.init_stance(q_neutral)
    foot_init_body = kin.get_foot_position_body(CHOSEN_LEG, q_neutral)

    _worker_kin = kin
    _worker_q_neutral = q_neutral
    _worker_foot_init_body = foot_init_body


def generate_trajectory(seed):
    random.seed(seed)
    np.random.seed(seed)

    kin = _worker_kin
    q_current = _worker_q_neutral.copy()
    foot_init = _worker_foot_init_body.copy()

    start_foot_body = foot_init + np.array([
        random.uniform(-0.15, 0.15),
        random.uniform(-0.05, 0.05),
        random.uniform(-0.10, 0.10),
    ])
    goal_foot_body = foot_init + np.array([
        random.uniform(-0.15, 0.15),
        random.uniform(-0.05, 0.05),
        random.uniform(-0.10, 0.10),
    ])

    start_world = kin.body_to_world(start_foot_body, q_current)
    q_current, _, _ = kin.compute_leg_ik(CHOSEN_LEG, start_world, q_current)

    trajectory_joints = []
    initial_angles = kin.get_joint_angles(q_current, leg_idx=CHOSEN_LEG)
    trajectory_joints.append(initial_angles.tolist())

    waypoints = QuadrupedKinematics.generate_cycloid_waypoints(
        start_foot_body, goal_foot_body, STEP_HEIGHT, NUM_TRAJ_STEPS,
    )

    for waypoint in waypoints[1:]:
        target_world = kin.body_to_world(waypoint, q_current)
        q_new, _, _ = kin.compute_leg_ik(CHOSEN_LEG, target_world, q_current)
        q_current = q_new
        angles = kin.get_joint_angles(q_new, leg_idx=CHOSEN_LEG)
        trajectory_joints.append(angles.tolist())

    return (start_foot_body.tolist(), goal_foot_body.tolist(), trajectory_joints)


def main():
    num_workers = cpu_count()
    num_trajectories = 2_000_000

    print(f"Generating {num_trajectories} FL leg step trajectories using {num_workers} workers...")

    root = zarr.open_group("step/trajectory_log.zarr", mode="w")
    num_joints = 3
    start_positions = root.zeros("start_positions", shape=(num_trajectories, 3), dtype=np.float32)
    goal_positions = root.zeros("goal_positions", shape=(num_trajectories, 3), dtype=np.float32)
    joint_angles = root.zeros("joint_angles", shape=(num_trajectories, NUM_TRAJ_STEPS, num_joints), dtype=np.float32)

    batch_size = 10000
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        seeds = range(num_trajectories)
        batch_starts, batch_goals, batch_joints = [], [], []
        completed = 0

        for result in pool.imap_unordered(generate_trajectory, seeds, chunksize=50):
            sp, gp, tj = result
            batch_starts.append(sp)
            batch_goals.append(gp)
            batch_joints.append(tj)

            if len(batch_starts) >= batch_size:
                end = completed + len(batch_starts)
                start_positions[completed:end] = batch_starts
                goal_positions[completed:end] = batch_goals
                joint_angles[completed:end] = batch_joints
                completed = end
                batch_starts, batch_goals, batch_joints = [], [], []
                print(f"Completed {completed}/{num_trajectories}")

        if batch_starts:
            end = completed + len(batch_starts)
            start_positions[completed:end] = batch_starts
            goal_positions[completed:end] = batch_goals
            joint_angles[completed:end] = batch_joints
            completed = end

    print(f"\nDone! Saved {num_trajectories} trajectories")


if __name__ == "__main__":
    main()
