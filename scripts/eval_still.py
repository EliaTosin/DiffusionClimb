#!/usr/bin/env python3
"""Evaluate the trunk (still) diffusion model against IK ground truth."""

import argparse
import random
import time

import numpy as np

from ink_kin_stance.constants import (
    FOOT_FRAMES, BODY_HEIGHT, X_RANGE, Y_RANGE, Z_RANGE,
)
from ink_kin_stance.kinematics import (
    QuadrupedKinematics, build_pinocchio_model, build_neutral_q, rotation_rpy,
)
from ink_kin_stance.diffusion.inference import load_model, generate_trajectory


def evaluate_diffusion(model_path="still/diffusion_model.pt", num_tests=10,
                       device="cuda", visualize=True, ddim_steps=0):
    pin_model, _, collision_model, visual_model = build_pinocchio_model()
    q_neutral = build_neutral_q(pin_model)
    kin = QuadrupedKinematics(pin_model, FOOT_FRAMES)
    kin.init_stance(q_neutral)

    print(f"Loading model from {model_path}...")
    diff_model, diffusion, checkpoint = load_model(model_path, device=device)
    num_steps = checkpoint["num_steps"]
    sampling_mode = f"DDIM ({ddim_steps})" if ddim_steps > 0 else f"DDPM ({checkpoint['num_timesteps']})"
    print(f"Model: num_steps={num_steps}, num_joints={checkpoint['num_joints']}, sampling={sampling_mode}")

    # Ranges (rotations disabled for now)
    roll_range = pitch_range = yaw_range = 0.0

    viz = None
    if visualize:
        from pinocchio.visualize import MeshcatVisualizer
        viz = MeshcatVisualizer(pin_model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()

    feet_neutral = [fp.copy() for fp in kin.feet_world_positions]
    all_errors, all_max_errors, all_diff_times, all_ik_times = [], [], [], []

    for test_idx in range(num_tests):
        # Perturb feet
        perturbed_feet = []
        for i in range(4):
            fp = feet_neutral[i].copy()
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
            random.uniform(-roll_range, roll_range),
            random.uniform(-pitch_range, pitch_range),
            random.uniform(-yaw_range, yaw_range),
        ])
        goal_pos = np.array([
            random.uniform(-X_RANGE, X_RANGE),
            random.uniform(-Y_RANGE, Y_RANGE),
            random.uniform(BODY_HEIGHT - Z_RANGE, BODY_HEIGHT + Z_RANGE),
        ])
        goal_rpy = np.array([
            random.uniform(-roll_range, roll_range),
            random.uniform(-pitch_range, pitch_range),
            random.uniform(-yaw_range, yaw_range),
        ])

        R_start = rotation_rpy(*start_rpy)
        q_start, _ = kin.solve_stance(
            body_translation=start_pos, body_rotation=R_start, q_init=q_neutral,
        )
        current_joints = kin.get_joint_angles(q_start)

        print(f"\n{'=' * 60}")
        print(f"Test {test_idx + 1}/{num_tests}")

        delta = np.concatenate([goal_pos - start_pos, goal_rpy - start_rpy])

        t0 = time.perf_counter()
        diff_traj = generate_trajectory(
            diff_model, diffusion, checkpoint,
            delta, current_joints, device=device, ddim_steps=ddim_steps,
        )
        t_diff = time.perf_counter() - t0

        # IK ground truth
        t0 = time.perf_counter()
        ik_traj = []
        ik_configs = []
        q_current = q_start.copy()
        for step in range(num_steps):
            alpha = step / (num_steps - 1) if num_steps > 1 else 1.0
            pos = start_pos + alpha * (goal_pos - start_pos)
            rpy = start_rpy + alpha * (goal_rpy - start_rpy)
            R = rotation_rpy(*rpy)
            q_new, _ = kin.solve_stance(body_translation=pos, body_rotation=R, q_init=q_current)
            q_current = q_new
            ik_configs.append(q_new.copy())
            ik_traj.append(kin.get_joint_angles(q_new))
        ik_traj = np.array(ik_traj)
        t_ik = time.perf_counter() - t0

        errors = np.abs(diff_traj - ik_traj)
        mean_err = errors.mean()
        max_err = errors.max()
        all_errors.append(mean_err)
        all_max_errors.append(max_err)
        all_diff_times.append(t_diff)
        all_ik_times.append(t_ik)

        print(f"  MAE: {mean_err:.6f} rad  Max: {max_err:.6f} rad  "
              f"Diff: {t_diff*1000:.1f}ms  IK: {t_ik*1000:.1f}ms  Speedup: {t_ik/t_diff:.1f}x")

        if viz is not None:
            for cfg in ik_configs:
                viz.display(cfg)
                time.sleep(0.05)
            time.sleep(0.5)

    kin.feet_world_positions = feet_neutral

    print(f"\n{'=' * 60}")
    print(f"SUMMARY ({num_tests} tests)")
    print(f"  Mean MAE: {np.mean(all_errors):.6f} rad")
    print(f"  Worst max: {np.max(all_max_errors):.6f} rad")
    print(f"  Diffusion: {np.mean(all_diff_times)*1000:.1f}ms  "
          f"IK: {np.mean(all_ik_times)*1000:.1f}ms  "
          f"Speedup: {np.mean(all_ik_times)/np.mean(all_diff_times):.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="still/diffusion_model.pt")
    parser.add_argument("--num-tests", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ddim-steps", type=int, default=0)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    evaluate_diffusion(
        model_path=args.model, num_tests=args.num_tests,
        device=args.device, visualize=not args.no_viz, ddim_steps=args.ddim_steps,
    )


if __name__ == "__main__":
    main()
