#!/usr/bin/env python3
"""Evaluate the step (leg) diffusion model against IK ground truth."""

import argparse
import random
import time

import numpy as np

from ink_kin_stance.constants import (
    FOOT_FRAMES, LEG_NAMES, LEG_IS_RIGHT, NUM_TRAJ_STEPS, STEP_HEIGHT,
)
from ink_kin_stance.kinematics import (
    QuadrupedKinematics, build_pinocchio_model, build_neutral_q,
)
from ink_kin_stance.diffusion.inference import load_model, generate_trajectory


def evaluate_diffusion(model_path="step/diffusion_model.pt", num_tests=10,
                       device="cuda", visualize=True, ddim_steps=0):
    pin_model, _, collision_model, visual_model = build_pinocchio_model()
    q_neutral = build_neutral_q(pin_model)
    kin = QuadrupedKinematics(pin_model, FOOT_FRAMES)
    kin.init_stance(q_neutral)
    foot_init_per_leg = [kin.get_foot_position_body(i, q_neutral) for i in range(4)]

    print(f"Loading model from {model_path}...")
    diff_model, diffusion, checkpoint = load_model(model_path, device=device)
    num_steps = checkpoint["num_steps"]
    print(f"Model: num_steps={num_steps}, num_joints={checkpoint['num_joints']}")

    viz = None
    if visualize:
        from pinocchio.visualize import MeshcatVisualizer
        viz = MeshcatVisualizer(pin_model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()

    all_errors, all_max_errors, all_diff_times, all_ik_times = [], [], [], []

    for test_idx in range(num_tests):
        chosen_leg = test_idx % 4
        is_right = LEG_IS_RIGHT[chosen_leg]
        foot_init = foot_init_per_leg[chosen_leg]
        leg_name = LEG_NAMES[chosen_leg]

        start_foot = foot_init + np.array([
            random.uniform(-0.05, 0.05),
            random.uniform(-0.03, 0.03),
            random.uniform(-0.05, 0.05),
        ])
        goal_foot = foot_init + np.array([
            random.uniform(-0.05, 0.10),
            random.uniform(-0.03, 0.03),
            random.uniform(-0.05, 0.05),
        ])

        print(f"\n{'=' * 60}")
        print(f"Test {test_idx + 1}/{num_tests} -- {leg_name} (mirror={'yes' if is_right else 'no'})")

        # IK ground truth
        t0 = time.perf_counter()
        q_current = q_neutral.copy()
        start_world = kin.body_to_world(start_foot, q_current)
        q_current, _, _ = kin.compute_leg_ik(chosen_leg, start_world, q_current)

        ik_traj = [kin.get_joint_angles(q_current, leg_idx=chosen_leg)]
        waypoints = QuadrupedKinematics.generate_cycloid_waypoints(
            start_foot, goal_foot, STEP_HEIGHT, num_steps,
        )
        for wp in waypoints[1:]:
            tw = kin.body_to_world(wp, q_current)
            q_current, _, _ = kin.compute_leg_ik(chosen_leg, tw, q_current)
            ik_traj.append(kin.get_joint_angles(q_current, leg_idx=chosen_leg))
        ik_traj = np.array(ik_traj)
        t_ik = time.perf_counter() - t0

        # Diffusion
        delta_pos = goal_foot - start_foot
        current_joints = ik_traj[0].copy()
        if is_right:
            delta_pos = delta_pos.copy()
            delta_pos[1] = -delta_pos[1]
            current_joints[0] = -current_joints[0]

        t0 = time.perf_counter()
        diff_traj = generate_trajectory(
            diff_model, diffusion, checkpoint,
            delta_pos, current_joints, device=device, ddim_steps=ddim_steps,
        )
        if is_right:
            diff_traj[:, 0] = -diff_traj[:, 0]
        t_diff = time.perf_counter() - t0

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
            q_display = q_neutral.copy()
            for step in range(num_steps):
                angles = diff_traj[step]
                base_idx = 7 + chosen_leg * 4
                q_display[base_idx + 0] = angles[0]
                q_display[base_idx + 1] = np.sin(angles[1])
                q_display[base_idx + 2] = np.cos(angles[1])
                q_display[base_idx + 3] = angles[2]
                viz.display(q_display)
                time.sleep(0.05)
            time.sleep(0.5)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY ({num_tests} tests)")
    print(f"  Mean MAE: {np.mean(all_errors):.6f} rad")
    print(f"  Worst max: {np.max(all_max_errors):.6f} rad")
    print(f"  Diffusion: {np.mean(all_diff_times)*1000:.1f}ms  "
          f"IK: {np.mean(all_ik_times)*1000:.1f}ms  "
          f"Speedup: {np.mean(all_ik_times)/np.mean(all_diff_times):.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="step/diffusion_model.pt")
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
