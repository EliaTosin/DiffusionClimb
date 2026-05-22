#!/usr/bin/env python3
"""Generate trunk (still) trajectory data for diffusion training."""

import random
import numpy as np
import zarr
from multiprocessing import Pool, cpu_count

from ink_kin_stance.constants import (
    URDF_PATH, FOOT_FRAMES, BODY_HEIGHT, THIGH_ANGLE, CALF_ANGLE,
    NUM_TRAJ_STEPS, X_RANGE, Y_RANGE, Z_RANGE,
    ROLL_RANGE, PITCH_RANGE, YAW_RANGE, IDENTITY_FRACTION,
)
from ink_kin_stance.kinematics import (
    QuadrupedKinematics, build_neutral_q, rotation_rpy, check_joint_limits,
)
import pinocchio as pin


# Worker state (initialised per process)
_worker_kin = None
_worker_q_neutral = None
_worker_feet_neutral = None


def init_worker():
    global _worker_kin, _worker_q_neutral, _worker_feet_neutral

    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    kin = QuadrupedKinematics(model, FOOT_FRAMES)
    q_neutral = build_neutral_q(model)
    kin.init_stance(q_neutral)

    _worker_kin = kin
    _worker_q_neutral = q_neutral
    _worker_feet_neutral = [fp.copy() for fp in kin.feet_world_positions]


def generate_trajectory(seed):
    random.seed(seed)
    np.random.seed(seed)

    kin = _worker_kin
    q_current = _worker_q_neutral.copy()

    # Perturb foot positions
    perturbed_feet = []
    for i in range(4):
        fp = _worker_feet_neutral[i].copy()
        fp[0] += random.uniform(-0.08, 0.08)
        fp[1] += random.uniform(-0.05, 0.05)
        fp[2] += random.uniform(-0.04, 0.04)
        perturbed_feet.append(fp)
    kin.feet_world_positions = perturbed_feet

    start_pos = np.array([
        random.uniform(-X_RANGE, X_RANGE),
        random.uniform(-Y_RANGE, Y_RANGE),
        random.uniform(BODY_HEIGHT - Z_RANGE, BODY_HEIGHT + Z_RANGE),
    ])
    start_rpy = np.array([
        random.uniform(-ROLL_RANGE, ROLL_RANGE),
        random.uniform(-PITCH_RANGE, PITCH_RANGE),
        random.uniform(-YAW_RANGE, YAW_RANGE),
    ])

    if random.random() < IDENTITY_FRACTION:
        goal_pos = start_pos + np.array([
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01),
        ])
        goal_rpy = start_rpy + np.array([
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01),
        ])
    else:
        goal_pos = np.array([
            random.uniform(-X_RANGE, X_RANGE),
            random.uniform(-Y_RANGE, Y_RANGE),
            random.uniform(BODY_HEIGHT - Z_RANGE, BODY_HEIGHT + Z_RANGE),
        ])
        goal_rpy = np.array([
            random.uniform(-ROLL_RANGE, ROLL_RANGE),
            random.uniform(-PITCH_RANGE, PITCH_RANGE),
            random.uniform(-YAW_RANGE, YAW_RANGE),
        ])

    R_start = rotation_rpy(*start_rpy)
    q_current, ok = kin.solve_stance(
        body_translation=start_pos, body_rotation=R_start, q_init=q_current,
    )
    if not ok:
        return None
    if not check_joint_limits(kin.get_joint_angles(q_current)):
        return None

    trajectory_joints = []
    body_positions, body_velocities = kin.compute_spline_target_cmd(start_pos, goal_pos, NUM_TRAJ_STEPS)
    for step in range(NUM_TRAJ_STEPS):
        alpha = step / (NUM_TRAJ_STEPS - 1)
        # current_pos = start_pos + alpha * (goal_pos - start_pos)
        current_pos = body_positions[step]
        current_vel = body_velocities[step]
        current_rpy = start_rpy + alpha * (goal_rpy - start_rpy)
        R = rotation_rpy(*current_rpy)

        # q_new, ok = kin.solve_stance(
        #     body_translation=current_pos, body_rotation=R, q_init=q_current,
        # )
        q_new, _ = kin.solve_stance_vel(
            body_translation=current_pos, body_rotation=R, q_init=q_current, body_vel=current_vel,
        )

        q_current = q_new
        angles = kin.get_joint_angles(q_new)
        if not check_joint_limits(angles):
            return None
        trajectory_joints.append(angles.tolist())

    return (start_pos.tolist(), goal_pos.tolist(),
            start_rpy.tolist(), goal_rpy.tolist(), trajectory_joints)


def main():
    num_workers = cpu_count()
    num_trajectories = 2_000_000

    print(f"Generating {num_trajectories} trajectories using {num_workers} workers...")

    root = zarr.open_group("still/trajectory_log_vel.zarr", mode="w")
    start_positions = root.zeros("start_positions", shape=(num_trajectories, 3), dtype=np.float32)
    goal_positions = root.zeros("goal_positions", shape=(num_trajectories, 3), dtype=np.float32)
    start_rpys = root.zeros("start_rpys", shape=(num_trajectories, 3), dtype=np.float32)
    goal_rpys = root.zeros("goal_rpys", shape=(num_trajectories, 3), dtype=np.float32)
    joint_angles = root.zeros("joint_angles", shape=(num_trajectories, NUM_TRAJ_STEPS, 12), dtype=np.float32)

    batch_size = 10000
    seed_offset = 0
    skipped = 0

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        batch_starts, batch_goals = [], []
        batch_start_rpys, batch_goal_rpys = [], []
        batch_joints = []
        completed = 0

        while completed + len(batch_starts) < num_trajectories:
            remaining = num_trajectories - completed - len(batch_starts)
            chunk = remaining + max(remaining // 5, 1000)
            seeds = range(seed_offset, seed_offset + chunk)
            seed_offset += chunk

            for result in pool.imap_unordered(generate_trajectory, seeds, chunksize=50):
                if result is None:
                    skipped += 1
                    continue

                sp, gp, sr, gr, tj = result
                batch_starts.append(sp)
                batch_goals.append(gp)
                batch_start_rpys.append(sr)
                batch_goal_rpys.append(gr)
                batch_joints.append(tj)

                plot = False
                if plot:
                    import matplotlib.pyplot as plt
                    print("plotting")
                    tj_numpy = np.array(tj)
                    # IK comparison plot (updated each walk step)
                    JOINT_NAMES = ["hip", "thigh", "calf"]
                    LEG_NAMES = ["FL", "FR", "RL", "RR"]
                    fig2, axes2 = plt.subplots(4, 3, figsize=(14, 10))
                    offset_trl = np.array(gp) - np.array(sp)
                    offset_rot = np.array(gr) - np.array(gp)
                    fig2.suptitle(f"{offset_trl}, \n\n {offset_rot}")
                    for leg in range(4):
                        for j in range(3):
                            ax = axes2[leg, j]
                            ax.set_title(f"{LEG_NAMES[leg]} {JOINT_NAMES[j]}")
                            ax.set_xlabel("step")
                            ax.set_ylabel("rad")
                            ax.plot(np.arange(NUM_TRAJ_STEPS), tj_numpy[:, leg*j], "r", label="CLIK")
                            if leg == 0 and j == 0:
                                ax.legend(loc="upper left", fontsize=7)
                    fig2.tight_layout()
                    plt.show()

                if len(batch_starts) >= batch_size:
                    end = completed + len(batch_starts)
                    start_positions[completed:end] = batch_starts
                    goal_positions[completed:end] = batch_goals
                    start_rpys[completed:end] = batch_start_rpys
                    goal_rpys[completed:end] = batch_goal_rpys
                    joint_angles[completed:end] = batch_joints
                    completed = end
                    batch_starts, batch_goals = [], []
                    batch_start_rpys, batch_goal_rpys = [], []
                    batch_joints = []
                    print(f"Completed {completed}/{num_trajectories} (skipped {skipped})")

                if completed + len(batch_starts) >= num_trajectories:
                    break

        if batch_starts:
            remaining = num_trajectories - completed
            end = completed + min(len(batch_starts), remaining)
            start_positions[completed:end] = batch_starts[:remaining]
            goal_positions[completed:end] = batch_goals[:remaining]
            start_rpys[completed:end] = batch_start_rpys[:remaining]
            goal_rpys[completed:end] = batch_goal_rpys[:remaining]
            joint_angles[completed:end] = batch_joints[:remaining]
            completed = end

    print(f"\nDone! Saved {completed} trajectories (skipped {skipped})")


if __name__ == "__main__":
    main()
