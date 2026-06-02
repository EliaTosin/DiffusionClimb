import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import torch
import argparse
import importlib.util
from ink_kin_stance.kinematics import QuadrupedKinematics

# =============================================================================
# Load training modules for model inference
# =============================================================================

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
still_train = _load_module("still_train", os.path.join(SCRIPT_DIR, "still", "diffusion_train.py"))
step_train = _load_module("step_train", os.path.join(SCRIPT_DIR, "step", "diffusion_train.py"))


# =============================================================================
# Pinocchio setup
# =============================================================================

URDF_PATH = os.path.join(SCRIPT_DIR, "aliengo.urdf")
MESH_DIR = os.path.dirname(os.path.abspath(URDF_PATH))

pin_model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
pin_data = pin_model.createData()

try:
    collision_model = pin.buildGeomFromUrdf(
        pin_model, URDF_PATH, pin.GeometryType.COLLISION, package_dirs=[MESH_DIR]
    )
    visual_model = pin.buildGeomFromUrdf(
        pin_model, URDF_PATH, pin.GeometryType.VISUAL, package_dirs=[MESH_DIR]
    )
except Exception as e:
    print(f"Warning: Could not load geometry models: {e}")
    collision_model = pin.GeometryModel()
    visual_model = pin.GeometryModel()

FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
LEG_NAMES = ["FL", "FR", "RL", "RR"]
BODY_HEIGHT = 0.42
THIGH_ANGLE = 0.8
CALF_ANGLE = -1.6
NUM_TRAJ_STEPS = 20
STEP_HEIGHT = 0.05
LEG_IS_RIGHT = {0: False, 1: True, 2: False, 3: True}
MIRROR_LEG = {0: 1, 1: 0, 2: 3, 3: 2}  # FL<->FR, RL<->RR


# =============================================================================
# IK ground truth
# =============================================================================

def generate_combined_ik(kin, q_start, body_start, body_goal,
                         stepping_leg, foot_start_body, foot_goal_body,
                         num_steps=20, step_height=0.05):
    """Generate IK ground truth for combined body shift + leg step.

    At each timestep:
    - Interpolate body position
    - Non-stepping legs: IK to keep feet at their pinned world positions
    - Stepping leg: IK to follow cycloid from start to goal (in body frame)

    Returns: (num_steps, 12) joint angles, final q
    """
    cycloid_waypoints = kin.generate_cycloid_waypoints(
        foot_start_body, foot_goal_body, step_height, num_steps
    )

    trajectory = []
    q_current = q_start.copy()

    for step in range(num_steps):
        alpha = step / (num_steps - 1) if num_steps > 1 else 1.0
        body_pos = body_start + alpha * (body_goal - body_start)

        # Set body position
        q_current = kin.set_body_pose(q_current, body_pos)

        # Non-stepping legs: IK to pinned world positions
        for leg_idx in range(4):
            if leg_idx == stepping_leg:
                continue
            target = kin.feet_world_positions[leg_idx]
            q_current, _, _ = kin.compute_leg_ik(leg_idx, target, q_current)

        # Stepping leg: cycloid waypoint (body frame → world)
        foot_world = kin.body_to_world(cycloid_waypoints[step], q_current)
        q_current, _, _ = kin.compute_leg_ik(stepping_leg, foot_world, q_current)

        angles = kin.get_joint_angles(q_current)
        trajectory.append(angles)

    return np.array(trajectory), q_current


# =============================================================================
# Diffusion trajectory generation
# =============================================================================

def generate_combined_diffusion(t_model, t_diffusion, t_checkpoint,
                                s_model, s_diffusion, s_checkpoint,
                                delta_body, delta_foot, current_joints_12,
                                stepping_leg, device="cuda", ddim_steps = 0):
    """Generate combined trajectory from both diffusion models.

    - Trunk model: all 12 joints for body shifting
    - Step model: 3 joints for the stepping leg (with mirroring for right-side)

    Returns: (num_steps, 12) joint angles
    """
    # Symmetrize input for trunk model: replace stepping leg's joints
    # with the mirror of its counterpart (trunk was trained on symmetric configs
    # and its output for the stepping leg gets overridden anyway)
    sym_joints = current_joints_12.copy()
    mirror = MIRROR_LEG[stepping_leg]
    mirror_joints = current_joints_12[mirror * 3: (mirror + 1) * 3].copy()
    mirror_joints[0] = -mirror_joints[0]  # negate hip for left<->right
    sym_joints[stepping_leg * 3: (stepping_leg + 1) * 3] = mirror_joints

    # Trunk trajectory (all 12 joints) — 6D delta: pos + rpy (no rotation during walk)
    trunk_delta = np.concatenate([delta_body, np.zeros(3)])
    trunk_traj = still_train.generate_trajectory(
        t_model, t_diffusion, t_checkpoint,
        trunk_delta, sym_joints, device=device, ddim_steps=ddim_steps
    )

    # Step trajectory (3 joints for stepping leg, uses actual joints)
    leg_joints = current_joints_12[stepping_leg * 3: (stepping_leg + 1) * 3].copy()
    delta = delta_foot.copy()

    # Mirror for right-side legs
    is_right = LEG_IS_RIGHT[stepping_leg]
    if is_right:
        delta[1] = -delta[1]
        leg_joints[0] = -leg_joints[0]

    step_traj = step_train.generate_trajectory(
        s_model, s_diffusion, s_checkpoint,
        delta, leg_joints, device=device, ddim_steps=ddim_steps
    )

    # Unmirror step output for right-side legs
    if is_right:
        step_traj[:, 0] = -step_traj[:, 0]

    # Combine: trunk for all legs, step overrides the stepping leg
    combined = trunk_traj.copy()
    combined[:, stepping_leg * 3: (stepping_leg + 1) * 3] = step_traj

    return combined


# =============================================================================
# Setup neutral configuration
# =============================================================================

q_init = pin.neutral(pin_model)
q_init[2] = BODY_HEIGHT
for leg in range(4):
    base_idx = 7 + leg * 4
    q_init[base_idx + 0] = 0.0
    q_init[base_idx + 1] = np.sin(THIGH_ANGLE)
    q_init[base_idx + 2] = np.cos(THIGH_ANGLE)
    q_init[base_idx + 3] = CALF_ANGLE

kin = QuadrupedKinematics(pin_model, FOOT_FRAMES, body_frame_name="trunk")
q_neutral = kin.init_stance(q_init)


# =============================================================================
# Main evaluation
# =============================================================================

def evaluate_walk(trunk_model_path, step_model_path, num_walk_steps=8,
                  delta_body=None, delta_foot=None,
                  device="cuda", visualize=True, ddim_steps=0):
    if delta_body is None:
        delta_body = np.array([0.02, 0.0, 0.0])
    if delta_foot is None:
        delta_foot = np.array([-0.08, 0.0, 0.0])

    # Load models
    print(f"Loading trunk model from {trunk_model_path}...")
    t_model, t_diffusion, t_checkpoint = still_train.load_model(trunk_model_path, device=device)
    print(f"  num_steps={t_checkpoint['num_steps']}, num_joints={t_checkpoint['num_joints']}")

    print(f"Loading step model from {step_model_path}...")
    s_model, s_diffusion, s_checkpoint = step_train.load_model(step_model_path, device=device)
    print(f"  num_steps={s_checkpoint['num_steps']}, num_joints={s_checkpoint['num_joints']}")

    # Setup visualization
    viz = None
    if visualize:
        import meshcat.geometry as g
        import meshcat.transformations as tf
        viz = MeshcatVisualizer(pin_model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()
        print("Meshcat viewer opened. Check your browser.")
        time.sleep(1)

        # Show initial pose
        viz.display(q_neutral)
        time.sleep(0.5)

    # Walk state
    body_pos = np.array([0.0, 0.0, BODY_HEIGHT])
    q_current = q_neutral.copy()
    current_joints_12 = kin.get_joint_angles(q_current)

    # Record neutral foot positions in body frame (FK correction target)
    neutral_foot_body = [
        kin.get_foot_position_body(i, q_neutral) for i in range(4)
    ]

    dt = 0.05  # visualization frame time

    # Metrics storage
    all_errors = []
    all_max_errors = []
    all_diff_times = []
    all_ik_times = []

    # Real-time foot position plots
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Foot World Positions")
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

    # Diffusion vs IK comparison plot (updated each walk step)
    JOINT_NAMES = ["hip", "thigh", "calf"]
    fig2, axes2 = plt.subplots(4, 3, figsize=(14, 10))
    fig2.suptitle("Diffusion (solid) vs IK (dashed)")
    cmp_lines = {}
    for leg in range(4):
        for j in range(3):
            ax = axes2[leg, j]
            ax.set_title(f"{LEG_NAMES[leg]} {JOINT_NAMES[j]}")
            ax.set_xlabel("step")
            ax.set_ylabel("rad")
            l_diff, = ax.plot([], [], "b-", label="diffusion")
            l_ik, = ax.plot([], [], "r--", label="IK")
            if leg == 0 and j == 0:
                ax.legend(loc="upper left", fontsize=7)
            cmp_lines[(leg, j)] = (l_diff, l_ik)
    fig2.tight_layout()
    plt.show()

    # Liste per il monitoraggio della stabilità
    walk_history_data = []  # Memorizzerà i dettagli di ogni "passo" del robot

    for walk_step in range(num_walk_steps):
        leg_idx = walk_step % 4
        leg_name = LEG_NAMES[leg_idx]

        kin.update_feet_positions(q_current)

        print(f"\n{'='*60}")
        print(f"Walk step {walk_step + 1}/{num_walk_steps}: {leg_name} stepping")
        print(f"  Body pos: {np.round(body_pos, 4)}")

        # FK correction: observe where the stepping foot actually is vs neutral,
        # adjust delta_foot to compensate for drift
        actual_foot_body = kin.get_foot_position_body(leg_idx, q_current)
        foot_error = neutral_foot_body[leg_idx] - actual_foot_body
        corrected_delta_foot = delta_foot + foot_error
        print(f"  Foot correction {leg_name}: {np.round(foot_error, 4)}")

        # --- Calcolo Stance Correttiva (Anticipatory Shift) ---
        # 1. Piedi che rimarranno a terra (formano il triangolo di supporto)
        support_indices = [i for i in range(4) if i != leg_idx]
        support_feet_xy = [kin.feet_world_positions[i][:2].copy() for i in support_indices]
        support_feet = [kin.feet_world_positions[i] for i in support_indices]

        # 2. Calcolo del baricentro del triangolo (proiezione X, Y)
        centroid_xy = np.mean(support_feet, axis=0)[:2]

        # Inizializziamo il contenitore per questo specifico passo
        current_step_info = {
            'leg_idx': leg_idx,
            'centroid': centroid_xy.copy(),
            'com_start': body_pos[:2].copy(),  # Salviamo dove inizia il corpo
            'com_path': [],
            'foot_path': [],
            'foot_start': kin.feet_world_positions[leg_idx][:2].copy(),
            'support_feet': support_feet_xy  # Solo i 3 piedi fissi
        }

        # 3. Calcolo differenza (delta) tra baricentro target e CoM attuale (body_pos)
        delta_body_xy = centroid_xy - body_pos[:2]

        # 4. Aggiornamento delta_body per il movimento di stance
        delta_body = np.array([delta_body_xy[0], delta_body_xy[1], 0.0])

        print(f"  Delta body: {delta_body}, Delta foot: {delta_foot}")
        body_goal = body_pos + delta_body

        # --- Diffusion ---
        t_start = time.perf_counter()
        diff_traj = generate_combined_diffusion(
            t_model, t_diffusion, t_checkpoint,
            s_model, s_diffusion, s_checkpoint,
            delta_body, corrected_delta_foot, current_joints_12,
            leg_idx, device=device, ddim_steps=ddim_steps
        )
        t_diff = time.perf_counter() - t_start
        print(f"  Diffusion generated in {t_diff * 1000:.1f} ms")

        # Offset trajectory so it starts from current joints (ensures continuity)
        offset = current_joints_12 - diff_traj[0]
        diff_traj += offset

        # --- IK ground truth ---
        foot_start_body = actual_foot_body.copy()
        foot_goal_body = actual_foot_body + corrected_delta_foot
        t_ik_start = time.perf_counter()
        ik_traj, _ = generate_combined_ik(
            kin, q_current, body_pos, body_goal,
            leg_idx, foot_start_body, foot_goal_body,
            num_steps=NUM_TRAJ_STEPS, step_height=STEP_HEIGHT
        )
        t_ik = time.perf_counter() - t_ik_start

        # Compare diffusion vs IK
        errors = np.abs(diff_traj - ik_traj)
        mean_error = errors.mean()
        max_error = errors.max()
        per_joint_error = errors.mean(axis=0)
        all_errors.append(mean_error)
        all_max_errors.append(max_error)
        all_diff_times.append(t_diff)
        all_ik_times.append(t_ik)
        print(f"  Mean error: {mean_error:.6f} rad ({np.rad2deg(mean_error):.4f} deg)")
        print(f"  Max error:  {max_error:.6f} rad ({np.rad2deg(max_error):.4f} deg)")
        print(f"  IK time: {t_ik * 1000:.1f} ms | Speedup: {t_ik/t_diff:.1f}x")

        # Update diffusion vs IK comparison plot
        steps_x = np.arange(NUM_TRAJ_STEPS)
        for leg in range(4):
            for j in range(3):
                ji = leg * 3 + j
                l_diff, l_ik = cmp_lines[(leg, j)]
                l_diff.set_data(steps_x, diff_traj[:, ji])
                l_ik.set_data(steps_x, ik_traj[:, ji])
                ax2 = axes2[leg, j]
                ax2.relim()
                ax2.autoscale_view()
        fig2.suptitle(f"Diffusion vs IK — step {walk_step+1} ({leg_name})")
        fig2.canvas.draw_idle()
        fig2.canvas.flush_events()

        input(f"  [Enter to continue to next step...]")

        # Mark walk step boundary on plots
        for ax in axes:
            ax.axvline(x=frame_count, color="gray", linestyle="--", linewidth=0.8)

        # --- Visualize / advance state ---
        for step in range(NUM_TRAJ_STEPS):
            alpha = step / (NUM_TRAJ_STEPS - 1) if NUM_TRAJ_STEPS > 1 else 1.0
            bp = body_pos + alpha * (body_goal - body_pos)

            # Monitoraggio errore e percorso CoM
            current_step_info['com_path'].append(bp[:2].copy())

            # Recuperiamo la posizione attuale del piede in volo
            pin.forwardKinematics(kin.model, kin.data, q_current)
            pin.updateFramePlacements(kin.model, kin.data)
            f_pos = kin.data.oMf[kin.foot_frame_ids[leg_idx]].translation[:2].copy()
            current_step_info['foot_path'].append(f_pos)

            q_current = kin.set_body_pose(q_current, bp)
            q_current = kin.set_joint_angles(q_current, diff_traj[step])

            # Pin non-stepping feet to their world positions
            for li in range(4):
                if li == leg_idx:
                    continue
                q_current, _, _ = kin.compute_leg_ik(li, kin.feet_world_positions[li], q_current)

            if visualize:
                viz.display(q_current)
                time.sleep(dt)

            # Update foot position plots
            pin.forwardKinematics(kin.model, kin.data, q_current)
            pin.updateFramePlacements(kin.model, kin.data)
            for i in range(4):
                pos = kin.data.oMf[kin.foot_frame_ids[i]].translation
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

        # 1. Aggiorna le posizioni dei piedi nel mondo dopo il movimento
        kin.update_feet_positions(q_current)

        # 2. Ora foot_end prenderà la nuova posizione corretta
        current_step_info['foot_end'] = kin.feet_world_positions[leg_idx][:2].copy()

        # 3. Salviamo il tutto nello storico
        walk_history_data.append(current_step_info)

        # --- Mappa 2D Top-Down (Evoluzione) ---
        plt.figure(figsize=(10, 10))
        num_total_steps = len(walk_history_data)

        for i, data in enumerate(walk_history_data):
            is_last = (i == num_total_steps - 1)
            ls = '-' if is_last else '--'  # Linea continua per l'ultimo, tratteggiata per lo storico
            alpha_val = 1.0 if is_last else 0.2  # Più opaco l'ultimo passo

            # Plot del percorso del corpo (CoM)
            c_path = np.array(data['com_path'])
            plt.plot(c_path[:, 0], c_path[:, 1], color='blue', linestyle=ls, alpha=alpha_val)

            # Plot della traiettoria del piede in volo
            f_path = np.array(data['foot_path'])
            plt.plot(f_path[:, 0], f_path[:, 1], color='orange', linestyle=ls, alpha=alpha_val)

            if is_last:
                # --- CORPO E STABILITÀ ---
                # CoM Iniziale (Cerchio blu) - Mostra da dove partiva il robot
                plt.scatter(data['com_start'][0], data['com_start'][1],
                            marker='o', facecolors='none', edgecolors='blue', s=100,
                            label='CoM Inizio Passo', zorder=5)

                # CoM Finale (X blu) - Dovrebbe cadere sopra il Target Baricentro
                plt.scatter(c_path[-1, 0], c_path[-1, 1],
                            marker='x', color='blue', s=100, label='CoM Finale', zorder=5)

                # Target Baricentro (Croce rossa)
                plt.scatter(data['centroid'][0], data['centroid'][1],
                            marker='+', color='red', s=150, label='Target Baricentro', zorder=5)

                # --- PIEDI ---
                # 3 Piedi fissi (Quadrati Neri)
                supp = np.array(data['support_feet'])
                plt.scatter(supp[:, 0], supp[:, 1], marker='s', color='black', s=100, label='Appoggio Fisso', zorder=4)

                # Piede in Swing: Partenza (Quadrato Rosso) e Target (Quadrato Verde)
                plt.scatter(data['foot_start'][0], data['foot_start'][1],
                            marker='s', color='red', s=100, label='Swing Start', zorder=6)
                plt.scatter(data['foot_end'][0], data['foot_end'][1],
                            marker='s', color='green', s=100, label='Swing Target', zorder=6)

        plt.title("Mappa Stabilità: Storico Tratteggiato e Ultimo Passo in Evidenza")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.5)

        # Pulizia legenda per non duplicare i nomi dello storico
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.tight_layout()
        plt.show()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Walk steps: {num_walk_steps}")
    print(f"Final body pos: {np.round(body_pos, 4)}")
    print(f"Final joints: {np.round(current_joints_12, 3)}")
    print(f"Mean MAE: {np.mean(all_errors):.6f} rad ({np.rad2deg(np.mean(all_errors)):.4f} deg)")
    print(f"Std MAE:  {np.std(all_errors):.6f} rad ({np.rad2deg(np.std(all_errors)):.4f} deg)")
    print(f"Mean Max Error: {np.mean(all_max_errors):.6f} rad ({np.rad2deg(np.mean(all_max_errors)):.4f} deg)")
    print(f"Worst Max Error: {np.max(all_max_errors):.6f} rad ({np.rad2deg(np.max(all_max_errors)):.4f} deg)")
    print(f"\nTiming:")
    print(f"  Diffusion: {np.mean(all_diff_times)*1000:.2f} ms (std: {np.std(all_diff_times)*1000:.2f} ms)")
    print(f"  IK:        {np.mean(all_ik_times)*1000:.2f} ms (std: {np.std(all_ik_times)*1000:.2f} ms)")
    print(f"  Avg speedup: {np.mean(all_ik_times)/np.mean(all_diff_times):.1f}x")

    if visualize:
        print("\nViewer is still open. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":

    delta_body = np.array([0.02, 0.005, 0.00]) # not used since I set the delta_body based on COM
    delta_foot = np.array([0.08, 0.00, 0.00])

    evaluate_walk(
        trunk_model_path ="still/diffusion_model.pt",
        step_model_path  ="step/diffusion_model.pt",
        num_walk_steps   = 200,
        delta_body       = delta_body,
        delta_foot       = delta_foot,
        device           = "cuda",
        ddim_steps       = 0
    )
