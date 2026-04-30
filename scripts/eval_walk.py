#!/usr/bin/env python3
"""Combined body-shift + leg-step walking evaluation using diffusion models."""

import argparse
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

from ink_kin_stance.constants import (
    BODY_HEIGHT, LEG_NAMES, LEG_IS_RIGHT, MIRROR_LEG, NUM_TRAJ_STEPS,
    JOINT_NAMES,
)
from ink_kin_stance.kinematics import (
    QuadrupedKinematics, build_pinocchio_model, build_neutral_q,
)
from ink_kin_stance.diffusion.inference import (
    load_model, build_condition, denorm, generate_trajectory,
)
from ink_kin_stance.guidance import StabilityGuidance, ddim_sample_guided

log = logging.getLogger(__name__)


# =============================================================================
# Combined diffusion trajectory generation
# =============================================================================

def generate_combined_diffusion(
    t_model, t_diffusion, t_checkpoint,
    s_model, s_diffusion, s_checkpoint,
    delta_body, delta_foot, current_joints_12,
    stepping_leg, kin,
    device="cuda", ddim_steps=0,
    use_guidance=False, guidance_scale=1.0, guidance_start=0.4,
    body_pos=None, body_quat_xyzw=None,
    pin_model=None, pin_data=None,
    debug=False,
):
    """Generate combined trunk + step trajectory.

    Returns: (num_steps, 12) joint angles.
    """
    # Symmetrize input for trunk model
    sym_joints = current_joints_12.copy()
    mirror = MIRROR_LEG[stepping_leg]
    mirror_joints = current_joints_12[mirror * 3: (mirror + 1) * 3].copy()
    mirror_joints[0] = -mirror_joints[0]
    sym_joints[stepping_leg * 3: (stepping_leg + 1) * 3] = mirror_joints

    # Step trajectory inputs
    leg_joints = current_joints_12[stepping_leg * 3: (stepping_leg + 1) * 3].copy()
    delta = delta_foot.copy()
    is_right = LEG_IS_RIGHT[stepping_leg]
    if is_right:
        delta[1] = -delta[1]
        leg_joints[0] = -leg_joints[0]

    # Trunk delta: 6D (pos + rpy, no rotation during walk)
    trunk_delta = np.concatenate([delta_body, np.zeros(3)])

    if use_guidance and ddim_steps > 0:
        if body_pos is None:
            body_pos = np.array([0.0, 0.0, BODY_HEIGHT])
        if body_quat_xyzw is None:
            body_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])

        all_feet = kin.feet_world_positions
        planted_idx = [i for i in range(4) if i != stepping_leg]
        planted_positions = np.array([all_feet[i] for i in planted_idx])
        all_positions = np.array(all_feet)

        trunk_guidance = StabilityGuidance(
            pin_model, pin_data,
            body_pos=body_pos, body_quat_xyzw=body_quat_xyzw,
            planted_foot_positions=all_positions,
            traj_mean=t_checkpoint["traj_mean"], traj_std=t_checkpoint["traj_std"],
            full_12_joints=sym_joints, stepping_leg=-1,
            margin=0.025, lambda_com=3.0, lambda_slip=1.0,
            protect_start=1, protect_end=3,
        )

        trunk_cond = build_condition(t_checkpoint, trunk_delta, sym_joints, device)
        trunk_shape = (1, t_checkpoint["num_steps"], t_checkpoint["num_joints"])
        trunk_raw = ddim_sample_guided(
            t_diffusion, t_model, trunk_cond, trunk_shape,
            ddim_steps=ddim_steps, guidance_fn=trunk_guidance,
            guidance_scale=guidance_scale, guidance_start=guidance_start,
        )
        trunk_traj = denorm(trunk_raw, t_checkpoint)

        step_guidance = StabilityGuidance(
            pin_model, pin_data,
            body_pos=body_pos, body_quat_xyzw=body_quat_xyzw,
            planted_foot_positions=planted_positions,
            traj_mean=s_checkpoint["traj_mean"], traj_std=s_checkpoint["traj_std"],
            full_12_joints=current_joints_12, stepping_leg=stepping_leg,
            margin=0.015, lambda_com=4.0, lambda_slip=0.0,
            protect_start=1, protect_end=5,
        )

        step_cond = build_condition(s_checkpoint, delta, leg_joints, device)
        step_shape = (1, s_checkpoint["num_steps"], s_checkpoint["num_joints"])
        step_raw = ddim_sample_guided(
            s_diffusion, s_model, step_cond, step_shape,
            ddim_steps=ddim_steps, guidance_fn=step_guidance,
            guidance_scale=guidance_scale, guidance_start=guidance_start,
        )
        step_traj = denorm(step_raw, s_checkpoint)
    else:
        trunk_traj = generate_trajectory(
            t_model, t_diffusion, t_checkpoint,
            trunk_delta, sym_joints, device=device, ddim_steps=ddim_steps,
        )
        step_traj = generate_trajectory(
            s_model, s_diffusion, s_checkpoint,
            delta, leg_joints, device=device, ddim_steps=ddim_steps,
        )

    # Unmirror step output for right-side legs
    if is_right:
        step_traj[:, 0] = -step_traj[:, 0]

    # Combine: trunk for all legs, step overrides the stepping leg
    combined = trunk_traj.copy()
    combined[:, stepping_leg * 3: (stepping_leg + 1) * 3] = step_traj

    if debug:
        leg_start = combined[0, stepping_leg * 3: (stepping_leg + 1) * 3]
        leg_end = combined[-1, stepping_leg * 3: (stepping_leg + 1) * 3]
        log.debug(
            "TRAJECTORY (%s): start=[%.4f,%.4f,%.4f] end=[%.4f,%.4f,%.4f]",
            LEG_NAMES[stepping_leg], *leg_start, *leg_end,
        )
        np.save("/tmp/evalwalk_combined_traj.npy", combined)
        np.save("/tmp/evalwalk_step_traj.npy", step_traj)
        np.save("/tmp/evalwalk_trunk_traj.npy", trunk_traj)

    return combined


# =============================================================================
# Main evaluation
# =============================================================================

def evaluate_walk(
    trunk_model_path, step_model_path,
    num_walk_steps=8, delta_body=None, delta_foot=None,
    device="cuda", visualize=True, ddim_steps=0,
    use_guidance=False, guidance_scale=1.0, guidance_start=0.4,
    interactive=False, debug=False,
):
    if delta_body is None:
        delta_body = np.array([0.02, 0.0, 0.0])
    if delta_foot is None:
        delta_foot = np.array([0.08, 0.0, 0.0])

    # Build Pinocchio model
    pin_model, pin_data, collision_model, visual_model = build_pinocchio_model()
    q_neutral = build_neutral_q(pin_model)
    kin = QuadrupedKinematics(pin_model)
    kin.init_stance(q_neutral)

    # Load diffusion models
    print(f"Loading trunk model from {trunk_model_path}...")
    t_model, t_diffusion, t_checkpoint = load_model(trunk_model_path, device=device)
    print(f"  num_steps={t_checkpoint['num_steps']}, num_joints={t_checkpoint['num_joints']}")

    print(f"Loading step model from {step_model_path}...")
    s_model, s_diffusion, s_checkpoint = load_model(step_model_path, device=device)
    print(f"  num_steps={s_checkpoint['num_steps']}, num_joints={s_checkpoint['num_joints']}")

    # Visualization
    viz = None
    if visualize:
        from pinocchio.visualize import MeshcatVisualizer
        viz = MeshcatVisualizer(pin_model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()
        print("Meshcat viewer opened.")
        time.sleep(1)
        viz.display(q_neutral)
        time.sleep(0.5)

    # Walk state
    body_pos = np.array([0.0, 0.0, BODY_HEIGHT])
    q_current = q_neutral.copy()
    current_joints_12 = kin.get_joint_angles(q_current)

    dt = 0.05
    all_diff_times = []

    # Foot position plots
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Foot Body-Frame Positions")
    axes = axes.flatten()
    foot_history = {i: {"x": [], "y": [], "z": []} for i in range(4)}
    frame_count = 0
    foot_lines = {}
    for i, ax in enumerate(axes):
        ax.set_title(LEG_NAMES[i])
        ax.set_xlabel("frame")
        ax.set_ylabel("position (m)")
        lx, = ax.plot([], [], label="x")
        ly, = ax.plot([], [], label="y")
        lz, = ax.plot([], [], label="z")
        ax.legend(loc="upper left")
        foot_lines[i] = (lx, ly, lz)
    fig.tight_layout()
    plt.show()

    # Diffusion joint trajectory plot
    fig2, axes2 = plt.subplots(4, 3, figsize=(14, 10))
    fig2.suptitle("Diffusion Joint Trajectories")
    cmp_lines = {}
    for leg in range(4):
        for j in range(3):
            ax = axes2[leg, j]
            ax.set_title(f"{LEG_NAMES[leg]} {JOINT_NAMES[j]}")
            ax.set_xlabel("step")
            ax.set_ylabel("rad")
            l_diff, = ax.plot([], [], "b-", label="diffusion")
            cmp_lines[(leg, j)] = (l_diff,)
    fig2.tight_layout()
    plt.show()

    for walk_step in range(num_walk_steps):
        leg_idx = walk_step % 4
        leg_name = LEG_NAMES[leg_idx]
        kin.update_feet_positions(q_current)

        print(f"\n{'=' * 60}")
        print(f"Walk step {walk_step + 1}/{num_walk_steps}: {leg_name} stepping")
        print(f"  Body pos: {np.round(body_pos, 4)}")

        body_goal = body_pos + delta_body

        body_quat = q_current[3:7]
        t_start = time.perf_counter()
        diff_traj = generate_combined_diffusion(
            t_model, t_diffusion, t_checkpoint,
            s_model, s_diffusion, s_checkpoint,
            delta_body, delta_foot, current_joints_12,
            leg_idx, kin,
            device=device, ddim_steps=ddim_steps,
            use_guidance=use_guidance, guidance_scale=guidance_scale,
            guidance_start=guidance_start,
            body_pos=body_pos, body_quat_xyzw=body_quat,
            pin_model=pin_model, pin_data=pin_data,
            debug=debug,
        )
        t_diff = time.perf_counter() - t_start
        print(f"  Diffusion generated in {t_diff * 1000:.1f} ms")
        all_diff_times.append(t_diff)

        # Update joint trajectory plot
        steps_x = np.arange(NUM_TRAJ_STEPS)
        for leg in range(4):
            for j in range(3):
                ji = leg * 3 + j
                l_diff, = cmp_lines[(leg, j)]
                l_diff.set_data(steps_x, diff_traj[:, ji])
                ax2 = axes2[leg, j]
                ax2.relim()
                ax2.autoscale_view()
        fig2.suptitle(f"Diffusion -- step {walk_step + 1} ({leg_name})")
        fig2.canvas.draw_idle()
        fig2.canvas.flush_events()

        if interactive:
            input("  [Enter to continue...]")

        for ax in axes:
            ax.axvline(x=frame_count, color="gray", linestyle="--", linewidth=0.8)

        # Visualize / advance state
        for step in range(NUM_TRAJ_STEPS):
            alpha = step / (NUM_TRAJ_STEPS - 1) if NUM_TRAJ_STEPS > 1 else 1.0
            bp = body_pos + alpha * (body_goal - body_pos)
            q_current = kin.set_body_pose(q_current, bp)
            q_current = kin.set_joint_angles(q_current, diff_traj[step])

            if visualize:
                viz.display(q_current)
                time.sleep(dt)

            for i in range(4):
                pos = kin.get_foot_position_body(i, q_current)
                foot_history[i]["x"].append(pos[0])
                foot_history[i]["y"].append(pos[1])
                foot_history[i]["z"].append(pos[2])
                lx, ly, lz = foot_lines[i]
                frames = range(len(foot_history[i]["x"]))
                lx.set_data(frames, foot_history[i]["x"])
                ly.set_data(frames, foot_history[i]["y"])
                lz.set_data(frames, foot_history[i]["z"])
                axes[i].relim()
                axes[i].autoscale_view()
            frame_count += 1
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        body_pos = body_goal.copy()
        current_joints_12 = kin.get_joint_angles(q_current)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Walk steps: {num_walk_steps}")
    print(f"Final body pos: {np.round(body_pos, 4)}")
    print(f"Final joints: {np.round(current_joints_12, 3)}")
    print(f"\nTiming:")
    print(f"  Diffusion: {np.mean(all_diff_times)*1000:.2f} ms "
          f"(std: {np.std(all_diff_times)*1000:.2f} ms)")

    if visualize:
        print("\nViewer is still open. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate walking with diffusion models")
    parser.add_argument("--trunk-model", default="still/diffusion_model.pt")
    parser.add_argument("--step-model", default="step/diffusion_model.pt")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--delta-body", type=float, nargs=3, default=[0.02, 0.0, 0.0])
    parser.add_argument("--delta-foot", type=float, nargs=3, default=[0.08, 0.0, 0.0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--no-guidance", action="store_true")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-start", type=float, default=0.4)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    evaluate_walk(
        trunk_model_path=args.trunk_model,
        step_model_path=args.step_model,
        num_walk_steps=args.num_steps,
        delta_body=np.array(args.delta_body),
        delta_foot=np.array(args.delta_foot),
        device=args.device,
        visualize=not args.no_viz,
        ddim_steps=args.ddim_steps,
        use_guidance=not args.no_guidance,
        guidance_scale=args.guidance_scale,
        guidance_start=args.guidance_start,
        interactive=args.interactive,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
