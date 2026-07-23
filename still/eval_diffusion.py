import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time
import os
import random
import torch
import matplotlib.pyplot as plt

# Import diffusion model utilities
from still.diffusion_train import load_model, generate_trajectory, generate_trajectory_retroaction
from ink_kin_stance.kinematics import rotation_rpy, QuadrupedKinematics
from still.still_diff_utils import StillDiffusionHelper
from ink_kin_stance.constants import ARTICULATION_JOINT_ORDER

URDF_PATH = "../aliengo.urdf"
MESH_DIR = os.path.dirname(os.path.abspath(URDF_PATH))

# Load the URDF with floating base
model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
data = model.createData()

# Load collision and visual models
try:
    collision_model = pin.buildGeomFromUrdf(model, URDF_PATH, pin.GeometryType.COLLISION, package_dirs=[MESH_DIR])
    visual_model = pin.buildGeomFromUrdf(model, URDF_PATH, pin.GeometryType.VISUAL, package_dirs=[MESH_DIR])
except Exception as e:
    print(f"Warning: Could not load geometry models: {e}")
    collision_model = pin.GeometryModel()
    visual_model = pin.GeometryModel()

FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
stance = QuadrupedKinematics(model, FOOT_FRAMES)
kin = QuadrupedKinematics(model, FOOT_FRAMES)
BODY_HEIGHT = 0.42

# Initial configuration
q_init = pin.neutral(model)
q_init[2] = BODY_HEIGHT
THIGH_ANGLE = 0.8
CALF_ANGLE = -1.6
for leg in range(4):
    base_idx = 7 + leg * 4
    q_init[base_idx + 0] = 0.0
    q_init[base_idx + 1] = np.sin(THIGH_ANGLE)
    q_init[base_idx + 2] = np.cos(THIGH_ANGLE)
    q_init[base_idx + 3] = CALF_ANGLE

q_neutral = stance.init_stance(q_init)
_ = kin.init_stance(q_init)


def clean_joint_name(name):
    # Rende il nome leggibile per i grafici (es. "FL_hip_joint" -> "FL Hip")
    return name.replace("_joint", "").replace("_", " ").upper()


def plot_trajectories(
        diff_trajectory,
        ik_trajectory,
        delta,
        helper_trajectory=None,
        ik_traj_vel=None,
        label_ik="Ground Truth (IK)",
        label_diff="Diff Retroaction",
        label_helper="Diff Retroaction Wloss",
        label_vel="IK Velocity",
        in_degrees=True,
        save_pdf = False,
):
    scale = np.degrees(1.0) if in_degrees else 1.0
    unit_label = "Position (deg)" if in_degrees else "Position (rad)"

    ik_traj = ik_trajectory * scale
    diff_traj = diff_trajectory * scale
    if helper_trajectory is not None:
        helper_traj = helper_trajectory * scale
    if ik_traj_vel is not None:
        ik_vel_traj = ik_traj_vel * scale

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'grid.linestyle': ':',
        'grid.alpha': 0.65
    })

    x_diff = np.linspace(0, 1, len(diff_trajectory))
    x_ik = np.linspace(0, 1, len(ik_trajectory))
    if helper_trajectory is not None:
        x_helper = np.linspace(0, 1, len(helper_trajectory))

    fig, axes = plt.subplots(6, 2, figsize=(8.5, 12.5), sharex=True, sharey=False)
    axes = axes.flatten()

    lines = []
    labels = []

    for i in range(12):
        joint_name = clean_joint_name(ARTICULATION_JOINT_ORDER[i])

        # 1. Ground Truth (IK)
        l_ik, = axes[i].plot(x_ik, ik_traj[:, i], color='#2c3e50', linestyle='-', linewidth=2.3, alpha=0.9)

        # 2. Modello Retroaction
        l_ret, = axes[i].plot(x_diff, diff_traj[:, i], color='#2980b9', linestyle='--', linewidth=2.0)

        if i == 0:
            lines.extend([l_ik, l_ret])
            labels.extend([label_ik, label_diff])

        # 3. Modello Wloss
        if helper_trajectory is not None:
            l_wloss, = axes[i].plot(x_helper, helper_traj[:, i], color='#e74c3c', linestyle='-.', linewidth=2.0)
            if i == 0:
                lines.append(l_wloss)
                labels.append(label_helper)

        # --- AGGIUNTA BADGE ERRORE FINALE ---
        # Calcoliamo l'errore finale all'ultimo passo temporale (index -1)
        err_wloss = abs(ik_traj[-1, i] - helper_traj[-1, i]) if helper_trajectory is not None else 0
        err_ret = abs(ik_traj[-1, i] - diff_traj[-1, i])

        # Testo compatto con lo scostamento finale in gradi
        err_text = f"e_fin: {err_wloss:.1f}°" if helper_trajectory is not None else f"e_fin: {err_ret:.1f}°"

        # Inseriamo una box discreta in alto a sinistra di ogni grafico (coordinate trasformate dell'asse [0,1])
        axes[i].text(
            0.04, 0.86, err_text,
            transform=axes[i].transAxes,
            fontsize=9.5, fontweight='bold',
            color='#c0392b' if helper_trajectory is not None else '#2980b9',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#fcfcfc', edgecolor='#e0e0e0', alpha=0.85)
        )

        axes[i].set_title(joint_name, fontweight='bold', pad=5)
        axes[i].grid(True)

        if i % 2 == 0:
            axes[i].set_ylabel(unit_label, fontweight='bold')

    for i in range(10, 12):
        axes[i].set_xlabel('Normalized Progress', fontweight='bold')

    # Rende in GRASSETTO i tick numerici degli assi X e Y per tutti e due i subplots
    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')
        for tick in ax.get_yticklabels():
            tick.set_fontweight('bold')

    # Titolo spostato leggermente più in alto
    title_str = (r"$\Delta$ Position target $\rightarrow$ "
                 f"X: {delta[0]:.3f} m | Y: {delta[1]:.3f} m | Z: {delta[2]:.3f} m")
    fig.suptitle(title_str, fontsize=12, fontweight='bold', y=0.985)

    # Legenda sotto il titolo
    fig.legend(
        lines, labels,
        loc='upper center',
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.965),
        frameon=True,
        facecolor='#fcfcfc',
        edgecolor='#ccc',
        fontsize=10.5,
        prop={'weight': 'bold', 'size': 10.5}
    )

    plt.tight_layout()
    # top=0.93 per adattare la nuova proporzione verticale
    fig.subplots_adjust(top=0.9, bottom=0.05, hspace=0.35, wspace=0.20)

    if save_pdf:
        plt.savefig("wloss_vs_retro_plot.pdf", bbox_inches="tight")

    plt.show()


def evaluate_diffusion(model_path="diffusion_model.pt", num_tests=10, device="cuda", visualize=True, ddim_steps=0):
    """
    Evaluate diffusion model against IK ground truth.
    Matches data generation: random foot perturbations and body RPY rotations.
    Visualization shows feet fixed on ground while body moves.
    """
    # Load diffusion model
    print(f"Loading model from {model_path}...")
    diff_model, diffusion, checkpoint = load_model(model_path, device=device)
    num_steps = checkpoint["num_steps"]
    num_joints = checkpoint["num_joints"]
    sampling_mode = f"DDIM ({ddim_steps} steps)" if ddim_steps > 0 else f"DDPM ({checkpoint['num_timesteps']} steps)"
    print(f"Model loaded: num_steps={num_steps}, num_joints={num_joints}, sampling={sampling_mode}")

    # Ranges matching generate_still_data.py
    X_RANGE, Y_RANGE, Z_RANGE = 0.10, 0.05, 0.1
    # ROLL_RANGE, PITCH_RANGE, YAW_RANGE = 0.25, 0.25, 0.3
    ROLL_RANGE, PITCH_RANGE, YAW_RANGE = 0.0, 0.0, 0.0
    random.seed(42)  # setting the same goals at every run

    # Setup visualization
    viz = None
    if visualize:
        import meshcat.geometry as g
        import meshcat.transformations as tf
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()

    # Save neutral feet so perturbation doesn't accumulate across tests
    feet_neutral = [fp.copy() for fp in stance.feet_world_positions]

    # Metrics storage
    all_errors = []
    all_max_errors = []
    all_diff_times = []
    all_ik_times = []

    for test_idx in range(num_tests):
        # Perturb foot positions from neutral (matching data gen)
        perturbed_feet = []
        for i in range(4):
            fp = feet_neutral[i].copy()
            fp[0] += random.uniform(-0.08, 0.08)
            fp[1] += random.uniform(-0.05, 0.05)
            fp[2] += random.uniform(-0.04, 0.04)
            perturbed_feet.append(fp)
        stance.feet_world_positions = perturbed_feet
        kin.feet_world_positions = perturbed_feet

        # Update foot markers in visualizer
        if viz is not None:
            for i, fp in enumerate(perturbed_feet):
                viz.viewer[f"foot_target_{i}"].set_object(
                    g.Sphere(0.02),
                    g.MeshLambertMaterial(color=0xFF0000, opacity=0.5)
                )
                viz.viewer[f"foot_target_{i}"].set_transform(tf.translation_matrix(fp))

        # Random start position + RPY
        start_pos = np.array([
            random.uniform(-X_RANGE, X_RANGE),
            random.uniform(-Y_RANGE, Y_RANGE),
            random.uniform(BODY_HEIGHT - Z_RANGE, BODY_HEIGHT + Z_RANGE)
        ])
        start_rpy = np.array([
            random.uniform(-ROLL_RANGE, ROLL_RANGE),
            random.uniform(-PITCH_RANGE, PITCH_RANGE),
            random.uniform(-YAW_RANGE, YAW_RANGE),
        ])

        # Random goal position + RPY
        goal_pos = np.array([
            random.uniform(-X_RANGE, X_RANGE),
            random.uniform(-Y_RANGE, Y_RANGE),
            random.uniform(BODY_HEIGHT - Z_RANGE, BODY_HEIGHT + Z_RANGE)
        ])
        goal_rpy = np.array([
            random.uniform(-ROLL_RANGE, ROLL_RANGE),
            random.uniform(-PITCH_RANGE, PITCH_RANGE),
            random.uniform(-YAW_RANGE, YAW_RANGE),
        ])

        # Solve IK at start pose to get current joint angles
        R_start = rotation_rpy(*start_rpy)
        q_start, _ = stance.solve_stance(
            body_translation=start_pos.tolist(),
            body_rotation=R_start,
            q_init=q_neutral
        )
        current_joints = stance.get_joint_angles(q_start)

        # Padding for t=0
        prev_joints = current_joints.copy()
        prev_actions = current_joints.copy()

        print(f"\n{'=' * 60}")
        print(f"Test {test_idx + 1}/{num_tests}")
        print(f"Start pos: {start_pos}, rpy: {np.rad2deg(start_rpy).round(1)} deg")
        print(f"Goal  pos: {goal_pos}, rpy: {np.rad2deg(goal_rpy).round(1)} deg")

        # Compute delta for diffusion conditioning (pos + rpy)
        delta = np.concatenate([goal_pos - start_pos, goal_rpy - start_rpy])

        # Generate trajectory with diffusion model
        t_start = time.perf_counter()
        # diff_trajectory = generate_trajectory(
        #     diff_model, diffusion, checkpoint,
        #     delta, current_joints, device=device, ddim_steps=ddim_steps
        # )
        diff_trajectory = generate_trajectory_retroaction(
            diff_model, diffusion, checkpoint,
            delta, current_joints, prev_joints, prev_actions,
            device=device, ddim_steps=ddim_steps
        )
        t_diff = time.perf_counter() - t_start

        # Generate ground truth with IK (interpolating both position and RPY)
        t_ik_start = time.perf_counter()
        ik_trajectory = []
        ik_configs = []  # full q for visualization
        q_current = q_start.copy()

        for step in range(num_steps):
            alpha = step / (num_steps - 1) if num_steps > 1 else 1.0
            current_pos = start_pos + alpha * (goal_pos - start_pos)
            current_rpy = start_rpy + alpha * (goal_rpy - start_rpy)
            R = rotation_rpy(*current_rpy)

            q_new, _ = stance.solve_stance(
                body_translation=current_pos.tolist(),
                body_rotation=R,
                q_init=q_current
            )
            q_current = q_new
            ik_configs.append(q_new.copy())
            angles = stance.get_joint_angles(q_new)
            ik_trajectory.append(angles)

        ik_trajectory = np.array(ik_trajectory)
        t_ik = time.perf_counter() - t_ik_start


        ### VEL
        ik_traj_vel = []
        NUM_TRAJ_STEPS = num_steps
        q_current = q_start.copy()
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
            ik_traj_vel.append(angles)
        ik_traj_vel = np.array(ik_traj_vel)

        helper = StillDiffusionHelper("diffusion_model.pt", device)
        helper_traj = helper.generate_trajectory_multienv(
            delta=torch.tensor(delta, dtype=torch.float, device=device).unsqueeze(dim=0),
            current_joints=torch.tensor(current_joints, dtype=torch.float, device=device).unsqueeze(dim=0),
            ddim_steps=ddim_steps
        )

        plot_trajectories(ik_trajectory=ik_trajectory, diff_trajectory=diff_trajectory, delta=delta, helper_trajectory=helper_traj[0, :, :].cpu().numpy(), ik_traj_vel=ik_traj_vel)

        # Compute errors
        errors = np.abs(diff_trajectory - ik_trajectory)
        mean_error = errors.mean()
        max_error = errors.max()
        per_joint_error = errors.mean(axis=0)

        all_errors.append(mean_error)
        all_max_errors.append(max_error)
        all_diff_times.append(t_diff)
        all_ik_times.append(t_ik)

        print(f"\nResults:")
        print(f"  Mean Absolute Error: {mean_error:.6f} rad ({np.rad2deg(mean_error):.4f} deg)")
        print(f"  Max Absolute Error:  {max_error:.6f} rad ({np.rad2deg(max_error):.4f} deg)")
        print(f"  Per-joint MAE: {np.round(per_joint_error, 4)}")
        print(f"  Diffusion time: {t_diff * 1000:.2f} ms | IK time: {t_ik * 1000:.2f} ms | Speedup: {t_ik / t_diff:.1f}x")

        # Visualize: body moves with feet fixed on ground
        if viz is not None:
            dt = 0.05
            for step in range(num_steps):
                viz.display(ik_configs[step])
                time.sleep(dt)
            time.sleep(0.5)

    # Restore neutral feet
    stance.feet_world_positions = feet_neutral

    # Summary statistics
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total tests: {num_tests}")
    print(f"Mean MAE: {np.mean(all_errors):.6f} rad ({np.rad2deg(np.mean(all_errors)):.4f} deg)")
    print(f"Std MAE:  {np.std(all_errors):.6f} rad ({np.rad2deg(np.std(all_errors)):.4f} deg)")
    print(f"Mean Max Error: {np.mean(all_max_errors):.6f} rad ({np.rad2deg(np.mean(all_max_errors)):.4f} deg)")
    print(f"Worst Max Error: {np.max(all_max_errors):.6f} rad ({np.rad2deg(np.max(all_max_errors)):.4f} deg)")
    print(f"\nTiming:")
    print(f"  Diffusion: {np.mean(all_diff_times) * 1000:.2f} ms (std: {np.std(all_diff_times) * 1000:.2f} ms)")
    print(f"  IK:        {np.mean(all_ik_times) * 1000:.2f} ms (std: {np.std(all_ik_times) * 1000:.2f} ms)")
    print(f"  Avg speedup: {np.mean(all_ik_times) / np.mean(all_diff_times):.1f}x")

    return all_errors, all_max_errors, all_diff_times, all_ik_times


if __name__ == "__main__":

    evaluate_diffusion(
        model_path="diff_model_vel_2hid_cond.pt",
        num_tests=1,
        device="cuda",
        # ddim_steps=20
    )