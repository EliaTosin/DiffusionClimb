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
from ink_kin_stance.diffusion.inference import load_model, generate_trajectory
from scipy.interpolate import CubicSpline
from ink_kin_stance.kinematics import QuadrupedKinematics

# =============================================================================
# Load training modules for model inference
# =============================================================================

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Pinocchio setup
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
SHOW_PLOTS = True

# =============================================================================
# IK ground truth
# =============================================================================

def generate_combined_ik(kin : QuadrupedKinematics, q_start, body_start, body_goal,
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

    body_positions, body_velocities = kin.compute_spline_target_cmd(body_start, body_goal, num_steps)

    trajectory = []
    q_current = q_start.copy()

    for step in range(num_steps):
        body_pos = body_positions[step]
        body_vel = body_velocities[step]

        # Set body position
        q_current = kin.set_body_pose(q_current, body_pos)

        # Non-stepping legs: IK to pinned world positions
        for leg_idx in range(4):
            if leg_idx == stepping_leg:
                continue
            target = kin.feet_world_positions[leg_idx]
            q_current, _ = kin.compute_leg_clik_body_centric(leg_idx, target, body_vel, q_current)

        # Stepping leg: cycloid waypoint (body frame → world)
        foot_world = kin.body_to_world(cycloid_waypoints[step], q_current)
        q_current, _, _ = kin.compute_leg_ik(stepping_leg, foot_world, q_current)

        angles = kin.get_joint_angles(q_current)
        trajectory.append(angles)

    if SHOW_PLOTS:
        trajectory2 = []
        q_current = q_start.copy()

        for step in range(num_steps):
            body_pos = body_positions[step]
            body_vel = body_velocities[step]

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
            trajectory2.append(angles)

        tj = np.array(trajectory)
        tj2 = np.array(trajectory2)
        # IK comparison plot (updated each walk step)
        JOINT_NAMES = ["hip", "thigh", "calf"]
        LEG_NAMES = ["FL", "FR", "RL", "RR"]
        fig2, axes2 = plt.subplots(4, 3, figsize=(14, 10))
        fig2.suptitle("IK (dashed) vs CLIK (solid)")
        for leg in range(4):
            for j in range(3):
                ax = axes2[leg, j]
                ax.set_title(f"{LEG_NAMES[leg]} {JOINT_NAMES[j]}")
                ax.set_xlabel("step")
                ax.set_ylabel("rad")
                ax.plot(np.arange(NUM_TRAJ_STEPS), tj[:, leg * j], "r", label="CLIK")
                ax.plot(np.arange(NUM_TRAJ_STEPS), tj[:, leg * j], "g--", label="IK CLASSIC")
                if leg == 0 and j == 0:
                    ax.legend(loc="upper left", fontsize=7)
        fig2.tight_layout()
        plt.show()
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
    trunk_traj = generate_trajectory(
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

    step_traj = generate_trajectory(
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

def evaluate_walk(trunk_model_1_path, trunk_model_2_path, step_model_path, num_walk_steps=8,
                  delta_body=None, delta_foot=None,
                  device="cuda", visualize=True, ddim_steps=0):
    if delta_body is None:
        delta_body = np.array([0.02, 0.0, 0.0])
    if delta_foot is None:
        delta_foot = np.array([-0.08, 0.0, 0.0])

    # --- Load Models ---
    print(f"Loading Trunk Model 1 from {trunk_model_1_path}...")
    t1_model, t1_diffusion, t1_checkpoint = load_model(trunk_model_1_path, device=device)

    print(f"Loading Trunk Model 2 from {trunk_model_2_path}...")
    t2_model, t2_diffusion, t2_checkpoint = load_model(trunk_model_2_path, device=device)

    print(f"Loading Step Model from {step_model_path}...")
    s_model, s_diffusion, s_checkpoint = load_model(step_model_path, device=device)

    # --- Setup visualization ---
    viz = None
    if visualize:
        import meshcat.geometry as g
        import meshcat.transformations as tf
        viz = MeshcatVisualizer(pin_model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()
        print("Meshcat viewer opened. Check your browser.")
        time.sleep(1)
        viz.display(q_neutral)
        time.sleep(0.5)

    # --- Walk state ---
    body_pos = np.array([0.0, 0.0, BODY_HEIGHT])
    q_current = q_neutral.copy()
    current_joints_12 = kin.get_joint_angles(q_current)

    neutral_foot_body = [
        kin.get_foot_position_body(i, q_neutral) for i in range(4)
    ]

    dt = 0.05  # visualization frame time
    JOINT_NAMES = ["hip", "thigh", "calf"]

    for walk_step in range(num_walk_steps):
        leg_idx = walk_step % 4
        leg_name = LEG_NAMES[leg_idx]

        kin.update_feet_positions(q_current)

        print(f"\n{'=' * 60}")
        print(f"Walk step {walk_step + 1}/{num_walk_steps}: {leg_name} stepping")

        body_goal = body_pos + delta_body

        # FK correction
        actual_foot_body = kin.get_foot_position_body(leg_idx, q_current)
        foot_error = neutral_foot_body[leg_idx] - actual_foot_body
        corrected_delta_foot = delta_foot + foot_error

        # ====================================================
        # PREDICZIONE 1 (Trunk Model 1)
        # ====================================================
        t_start_1 = time.perf_counter()
        diff_traj_1 = generate_combined_diffusion(
            t1_model, t1_diffusion, t1_checkpoint,
            s_model, s_diffusion, s_checkpoint,
            delta_body, corrected_delta_foot, current_joints_12,
            leg_idx, device=device, ddim_steps=ddim_steps
        )
        t_diff_1 = time.perf_counter() - t_start_1
        diff_traj_1 += (current_joints_12 - diff_traj_1[0])  # Offset

        # ====================================================
        # PREDICZIONE 2 (Trunk Model 2)
        # ====================================================
        t_start_2 = time.perf_counter()
        diff_traj_2 = generate_combined_diffusion(
            t2_model, t2_diffusion, t2_checkpoint,
            s_model, s_diffusion, s_checkpoint,
            delta_body, corrected_delta_foot, current_joints_12,
            leg_idx, device=device, ddim_steps=ddim_steps
        )
        t_diff_2 = time.perf_counter() - t_start_2
        diff_traj_2 += (current_joints_12 - diff_traj_2[0])  # Offset

        # ====================================================
        # IK GROUND TRUTH
        # ====================================================
        foot_start_body = actual_foot_body.copy()
        foot_goal_body = actual_foot_body + corrected_delta_foot
        ik_traj, _ = generate_combined_ik(
            kin, q_current, body_pos, body_goal,
            leg_idx, foot_start_body, foot_goal_body,
            num_steps=NUM_TRAJ_STEPS, step_height=STEP_HEIGHT
        )

        mean_err_1 = np.abs(diff_traj_1 - ik_traj).mean()
        mean_err_2 = np.abs(diff_traj_2 - ik_traj).mean()
        print(f"  Model 1 Error: {mean_err_1:.6f} rad | Time: {t_diff_1 * 1000:.1f} ms")
        print(f"  Model 2 Error: {mean_err_2:.6f} rad | Time: {t_diff_2 * 1000:.1f} ms")

        # ====================================================
        # GRAFICO PER IL PASSO CORRENTE
        # ====================================================
        if SHOW_PLOTS:
            steps_x = np.arange(NUM_TRAJ_STEPS)
            fig_joints, axes_joints = plt.subplots(4, 3, figsize=(14, 10))
            fig_joints.suptitle(f"Step {walk_step + 1} ({leg_name}): Model 1 (Blue) vs Model 2 (Green) vs IK (Red Dashed)")

            for leg in range(4):
                for j in range(3):
                    ji = leg * 3 + j
                    ax = axes_joints[leg, j]
                    ax.set_title(f"{LEG_NAMES[leg]} {JOINT_NAMES[j]}")
                    ax.set_ylabel("rad")

                    # Disegna direttamente tutti i dati calcolati
                    ax.plot(steps_x, diff_traj_1[:, ji], "b-", linewidth=2, label="Model 1")
                    ax.plot(steps_x, diff_traj_2[:, ji], "g-", linewidth=2, alpha=0.7, label="Model 2")
                    ax.plot(steps_x, ik_traj[:, ji], "r--", linewidth=1.5, label="IK Truth")

                    if leg == 0 and j == 0:
                        ax.legend(loc="best", fontsize=8)

            fig_joints.tight_layout()
            plt.show()  # Mostra senza fermare il codice Python!

        # --- Visualize / advance state usando il Modello 1 ---
        for step in range(NUM_TRAJ_STEPS):
            alpha = step / (NUM_TRAJ_STEPS - 1) if NUM_TRAJ_STEPS > 1 else 1.0
            bp = body_pos + alpha * (body_goal - body_pos)

            q_current = kin.set_body_pose(q_current, bp)
            q_current = kin.set_joint_angles(q_current, diff_traj_1[step])

            for li in range(4):
                if li == leg_idx: continue
                q_current, _, _ = kin.compute_leg_ik(li, kin.feet_world_positions[li], q_current)

            if visualize:
                viz.display(q_current)
                time.sleep(dt)

        body_pos = body_goal.copy()
        current_joints_12 = kin.get_joint_angles(q_current)

        # Attendi input dell'utente prima di passare al passo successivo
        input(f"  [Enter to continue to next step...]")

    print(f"\n{'=' * 60}")
    print("WALK COMPLETED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    delta_body = np.array([0.025, 0.0, 0.0])
    delta_foot = np.array([0.1, 0.0, 0.0])

    # Ora passiamo DUE percorsi per i modelli Trunk
    evaluate_walk(
        trunk_model_1_path="still/diffusion_model.pt",  # SOSTITUISCI CON IL TUO PATH
        trunk_model_2_path="still/diff_model_vel2.pt",  # SOSTITUISCI CON IL TUO PATH
        step_model_path="step/diffusion_model.pt",
        num_walk_steps=200,
        delta_body=delta_body,
        delta_foot=delta_foot,
        device="cuda",
        visualize=True,  # Mantieni o metti a False se vuoi fare run veloci solo per i plot
        ddim_steps=0
    )