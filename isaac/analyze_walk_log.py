#!/usr/bin/env python3
"""Analyze walk_log.npz: compare diffusion model vs IK reference trajectories.

Usage:
    python scripts/analyze_walk_log.py [path/to/walk_log.npz]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from ink_kin_stance.constants import JOINT_NAMES

# Full leg names to avoid confusion with "RL" (Reinforcement Learning)
LEG_FULL_NAMES = ["Front-Left", "Front-Right", "Rear-Left", "Rear-Right"]
LEG_SHORT = ["FL", "FR", "RL", "RR"]
JOINT_LABELS = [f"{LEG_SHORT[l]}_{JOINT_NAMES[j]}" for l in range(4) for j in range(3)]


def load_log(path):
    """Load walk_log.npz and return list of step dicts."""
    data = np.load(path, allow_pickle=True)
    num_steps = int(data["num_steps"])
    steps = []
    for i in range(num_steps):
        p = f"step{i}_"
        step = {
            "stepping_leg": int(data[p + "stepping_leg"]),
            "targets": data[p + "targets"],
            "actuals": data[p + "actuals"],
            "ik_joints": data[p + "ik_joints"],
            "ik_walk_joints": data[p + "ik_walk_joints"],
            "body_pos": data[p + "body_pos"],
            "body_quat": data[p + "body_quat"],
            "foot_pos": data[p + "foot_pos"],
            "joint_vel": data[p + "joint_vel"],
            "raw_trajectory": data[p + "raw_trajectory"],
            "delta_body": data[p + "delta_body"],
            "delta_foot": data[p + "delta_foot"],
        }
        steps.append(step)
    return steps


def print_summary(steps):
    """Print text summary for each step."""
    for i, s in enumerate(steps):
        sl = s["stepping_leg"]
        targets = s["targets"]
        actuals = s["actuals"]
        ik_walk = s["ik_walk_joints"]
        body_pos = s["body_pos"]
        foot_pos = s["foot_pos"]
        N = len(targets)

        print(f"\n{'=' * 70}")
        print(f"  Step {i}: {LEG_FULL_NAMES[sl]} ({N} frames)")
        print(f"  delta_body={s['delta_body']}  delta_foot={s['delta_foot']}")
        print(f"{'=' * 70}")

        # Tracking error
        te = np.abs(actuals - targets)
        print(f"  PD tracking:       mean={te.mean():.4f}  max={te.max():.4f} rad")

        # Diffusion vs IK walk
        if len(ik_walk) == N:
            dik = np.abs(targets - ik_walk)
            print(f"  Diffusion vs IK:   mean={dik.mean():.4f}  max={dik.max():.4f} rad")

        # Smoothness
        if N > 1:
            fd = np.diff(targets, axis=0)
            fd_ik = np.diff(ik_walk, axis=0) if len(ik_walk) == N else None
            print(f"  Smoothness (diff): mean={np.abs(fd).mean():.4f}  max={np.abs(fd).max():.4f} rad/frame")
            if fd_ik is not None:
                print(f"  Smoothness (IK):   mean={np.abs(fd_ik).mean():.4f}  max={np.abs(fd_ik).max():.4f} rad/frame")

        # Body displacement
        disp = body_pos[-1] - body_pos[0]
        print(f"  Body displacement: [{disp[0]:+.4f}, {disp[1]:+.4f}, {disp[2]:+.4f}] m")

        # Per-leg foot drift
        for leg in range(4):
            drift = np.linalg.norm(foot_pos[-1, leg] - foot_pos[0, leg]) * 1000
            marker = " <<STEP" if leg == sl else ""
            print(f"  {LEG_SHORT[leg]} foot drift: {drift:6.1f} mm{marker}")


def plot_joint_comparison(steps, step_idx=0):
    """Plot diffusion targets vs IK reference vs actuals for one step."""
    s = steps[step_idx]
    sl = s["stepping_leg"]
    targets = s["targets"]
    actuals = s["actuals"]
    ik_walk = s["ik_walk_joints"]
    N = len(targets)
    frames = np.arange(N)

    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True)
    fig.suptitle(f"Step {step_idx}: {LEG_FULL_NAMES[sl]} stepping  |  Diffusion vs IK vs Actual", fontsize=14)

    for leg in range(4):
        for j in range(3):
            ax = axes[leg, j]
            ji = leg * 3 + j

            ax.plot(frames, targets[:, ji], "b-", linewidth=1.5, label="Diffusion target")
            ax.plot(frames, actuals[:, ji], "r--", linewidth=1, alpha=0.7, label="Actual (PD)")
            if len(ik_walk) == N:
                ax.plot(frames, ik_walk[:, ji], "g:", linewidth=1.5, label="IK reference")

            ax.set_title(f"{LEG_SHORT[leg]} {JOINT_NAMES[j]}", fontsize=10)
            if j == 0:
                ax.set_ylabel("rad")
            if leg == 3:
                ax.set_xlabel("Frame")

            # Highlight stepping leg
            if leg == sl:
                ax.set_facecolor("#fff8f0")

    axes[0, 2].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_smoothness(steps, step_idx=0):
    """Plot frame-to-frame joint angle changes (smoothness metric)."""
    s = steps[step_idx]
    sl = s["stepping_leg"]
    targets = s["targets"]
    ik_walk = s["ik_walk_joints"]
    N = len(targets)

    if N < 2:
        return None

    fd_diff = np.diff(targets, axis=0)  # (N-1, 12)
    fd_ik = np.diff(ik_walk, axis=0) if len(ik_walk) == N else None

    fig, axes = plt.subplots(4, 3, figsize=(16, 10), sharex=True)
    fig.suptitle(f"Step {step_idx}: Frame-to-frame changes (smoothness)", fontsize=14)

    frames = np.arange(N - 1)
    for leg in range(4):
        for j in range(3):
            ax = axes[leg, j]
            ji = leg * 3 + j

            ax.plot(frames, fd_diff[:, ji], "b-", linewidth=1, label="Diffusion")
            if fd_ik is not None:
                ax.plot(frames, fd_ik[:, ji], "g--", linewidth=1, label="IK")
            ax.axhline(0, color="k", linewidth=0.3)

            ax.set_title(f"{LEG_SHORT[leg]} {JOINT_NAMES[j]}", fontsize=10)
            if j == 0:
                ax.set_ylabel("delta (rad)")
            if leg == 3:
                ax.set_xlabel("Frame")
            if leg == sl:
                ax.set_facecolor("#fff8f0")

    axes[0, 2].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_foot_positions(steps, step_idx=0):
    """Plot foot world positions over time."""
    s = steps[step_idx]
    sl = s["stepping_leg"]
    foot_pos = s["foot_pos"]  # (N, 4, 3)
    body_pos = s["body_pos"]  # (N, 3)
    N = len(foot_pos)
    frames = np.arange(N)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Step {step_idx}: Foot world positions ({LEG_FULL_NAMES[sl]} stepping)", fontsize=14)

    axis_names = ["X", "Y", "Z"]
    for a_idx, ax in enumerate(axes):
        for leg in range(4):
            style = "-" if leg == sl else "--"
            alpha = 1.0 if leg == sl else 0.6
            ax.plot(frames, foot_pos[:, leg, a_idx], style, alpha=alpha, label=LEG_SHORT[leg])
        ax.plot(frames, body_pos[:, a_idx], "k:", linewidth=1, label="Body")
        ax.set_ylabel(f"{axis_names[a_idx]} (m)")
        ax.legend(loc="upper right", fontsize=8)

    axes[2].set_xlabel("Frame")
    fig.tight_layout()
    return fig


def plot_body_trajectory(steps):
    """Plot body XY position across all steps."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Body XY trajectory across gait cycle", fontsize=14)

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    for i, s in enumerate(steps):
        body_pos = s["body_pos"]
        sl = s["stepping_leg"]
        ax.plot(body_pos[:, 0], body_pos[:, 1], "-", color=colors[sl % 4],
                linewidth=2, label=f"Step {i} ({LEG_FULL_NAMES[sl]})")
        ax.plot(body_pos[0, 0], body_pos[0, 1], "o", color=colors[sl % 4], markersize=6)
        ax.plot(body_pos[-1, 0], body_pos[-1, 1], "s", color=colors[sl % 4], markersize=6)

        # Plot foot positions at start and end
        fp = s["foot_pos"]
        for leg in range(4):
            ax.plot(fp[0, leg, 0], fp[0, leg, 1], "^", color=colors[leg], markersize=4, alpha=0.3)
            ax.plot(fp[-1, leg, 0], fp[-1, leg, 1], "v", color=colors[leg], markersize=4, alpha=0.6)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_error_histogram(steps):
    """Histogram of diffusion-vs-IK errors across all steps."""
    all_errors = []
    for s in steps:
        targets = s["targets"]
        ik_walk = s["ik_walk_joints"]
        if len(ik_walk) == len(targets):
            all_errors.append(np.abs(targets - ik_walk).flatten())

    if not all_errors:
        return None

    all_errors = np.concatenate(all_errors)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_errors, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(all_errors.mean(), color="r", linestyle="--", label=f"mean={all_errors.mean():.4f}")
    ax.axvline(np.percentile(all_errors, 95), color="orange", linestyle="--",
               label=f"95th={np.percentile(all_errors, 95):.4f}")
    ax.set_xlabel("Absolute error (rad)")
    ax.set_ylabel("Count")
    ax.set_title("Diffusion vs IK joint angle error distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "isaac/walk_log.npz"

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    steps = load_log(path)
    print(f"Loaded {len(steps)} steps from {path}")

    # Text summary
    print_summary(steps)

    # Plots
    for i in range(min(len(steps), 4)):
        plot_joint_comparison(steps, step_idx=i)
        plot_smoothness(steps, step_idx=i)
        plot_foot_positions(steps, step_idx=i)

    if len(steps) > 1:
        plot_body_trajectory(steps)

    plot_error_histogram(steps)

    # Save plots to a dedicated folder
    save_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "walk_analysis_plots")
    os.makedirs(save_dir, exist_ok=True)
    for i, fig_num in enumerate(plt.get_fignums()):
        fig = plt.figure(fig_num)
        fig_path = os.path.join(save_dir, f"plot_{i:02d}.png")
        fig.savefig(fig_path, dpi=150)
    print(f"\nSaved {len(plt.get_fignums())} plots to {save_dir}/")

    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    import os
    main()
