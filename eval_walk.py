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


# =============================================================================
# Combined kinematics
# =============================================================================

class WalkKinematics:
    """Combined body-shift + leg-step kinematics using Pinocchio."""

    def __init__(self, model, foot_frame_names, body_frame_name="trunk"):
        self.model = model
        self.data = model.createData()
        self.foot_frame_names = foot_frame_names
        self.foot_frame_ids = [model.getFrameId(name) for name in foot_frame_names]
        self.body_frame_id = model.getFrameId(body_frame_name)
        self.has_floating_base = (
            model.njoints > 1
            and model.joints[1].shortname() == "JointModelFreeFlyer"
        )
        self.feet_world_positions = None

    def init_stance(self, q_init):
        pin.forwardKinematics(self.model, self.data, q_init)
        pin.updateFramePlacements(self.model, self.data)
        self.feet_world_positions = [
            self.data.oMf[fid].translation.copy()
            for fid in self.foot_frame_ids
        ]
        return q_init

    def body_to_world(self, pos_body, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMb = self.data.oMf[self.body_frame_id]
        return oMb.act(np.array(pos_body))

    def world_to_body(self, pos_world, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMb = self.data.oMf[self.body_frame_id]
        return oMb.actInv(np.array(pos_world))

    def get_foot_position_body(self, leg_idx, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pos_world = self.data.oMf[self.foot_frame_ids[leg_idx]].translation
        return self.world_to_body(pos_world, q)

    def set_body_pose(self, q, body_translation, body_rotation=None):
        q = q.copy()
        if self.has_floating_base:
            q[0:3] = body_translation
            if body_rotation is not None:
                quat = pin.Quaternion(body_rotation)
                q[3:7] = np.array([quat.x, quat.y, quat.z, quat.w])
        return q

    def compute_leg_ik(self, leg_idx, target_pos_world, q_current,
                       eps=1e-4, max_iter=100, dt=0.1):
        frame_id = self.foot_frame_ids[leg_idx]
        q = q_current.copy()
        v_base = 6 if self.has_floating_base else 0
        leg_v_start = v_base + leg_idx * 3
        leg_v_end = leg_v_start + 3

        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            err = np.array(target_pos_world) - self.data.oMf[frame_id].translation
            if np.linalg.norm(err) < eps:
                return q, True, np.linalg.norm(err)
            J = pin.computeFrameJacobian(
                self.model, self.data, q, frame_id, pin.LOCAL_WORLD_ALIGNED
            )[:3, leg_v_start:leg_v_end]
            damp = 1e-6
            v = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(3), err)
            v_full = np.zeros(self.model.nv)
            v_full[leg_v_start:leg_v_end] = v
            q = pin.integrate(self.model, q, v_full * dt)
        return q, False, np.linalg.norm(err)

    def compute_leg_ik_body_centric(self, leg_idx, target_pos_world, body_cmd_vel_world, q_current,
                                    target_vel_world=None, eps=1e-4, max_iter=100, dt=0.1, kd=10.0):
        """
        Risolve l'IK nel frame locale (Body-Centric).
        - target_pos_world: dove vogliamo il piede (nel mondo)
        - target_vel_world: velocità del piede nel mondo (0 per Stance, v_cicloide per Swing)
        - body_cmd_vel_world: velocità a cui si sta muovendo il trunk
        """
        frame_id = self.foot_frame_ids[leg_idx]
        q = q_current.copy()
        v_base = 6 if self.has_floating_base else 0
        leg_v_start = v_base + leg_idx * 3
        leg_v_end = leg_v_start + 3
        if target_vel_world is None:
            target_vel_world = np.zeros(3)

        for _ in range(max_iter):
            # Aggiorniamo la cinematica UNA SOLA VOLTA per iterazione
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # 1. Estrazione della posa del Body (equivalente a oMb nei tuoi metodi)
            oMb = self.data.oMf[self.body_frame_id]

            # 2. CONVERSIONE POSIZIONI PIEDI DA WORLD A BODY LOCALE
            target_pos_local = oMb.actInv(np.array(target_pos_world))
            current_pos_local = oMb.actInv(self.data.oMf[frame_id].translation)

            # L'errore ora è puro e visto "dagli occhi" del body
            err_local = target_pos_local - current_pos_local

            if np.linalg.norm(err_local) < eps:
                return q, True, np.linalg.norm(err_local)

            # 3. CONVERSIONE VELOCITÀ DA WORLD A BODY LOCALE
            # Ruotiamo le velocità moltiplicandole per la Trasposta della rotazione del body
            R_body_inv = oMb.rotation.T
            foot_vel_local = R_body_inv @ np.array(target_vel_world)
            body_vel_local = R_body_inv @ np.array(body_cmd_vel_world)

            # 4. VELOCITÀ RELATIVA (Effetto Tapis Roulant)
            # La vera velocità target che i giunti devono creare è la differenza tra
            # come si muove il piede e come si muove il tronco.
            target_vel_local = foot_vel_local - body_vel_local

            # Legge CLIK calcolata interamente in locale
            v_task_local = target_vel_local + (kd * err_local)

            # 5. JACOBIANO ROTAZIONALE
            # Lo Jacobiano WORLD_ALIGNED calcola l'influenza dei giunti rispetto agli assi globali.
            # Per usarlo con il nostro v_task_local, dobbiamo ruotare anche lui!
            J_world = pin.computeFrameJacobian(
                self.model, self.data, q, frame_id, pin.LOCAL_WORLD_ALIGNED
            )[:3, leg_v_start:leg_v_end]

            J_local = R_body_inv @ J_world

            # 6. Risoluzione Damped Least Squares
            damp = 1e-6
            v = J_local.T @ np.linalg.solve(J_local @ J_local.T + damp * np.eye(3), v_task_local)

            v_full = np.zeros(self.model.nv)
            v_full[leg_v_start:leg_v_end] = v
            q = pin.integrate(self.model, q, v_full * dt)

        return q, False, np.linalg.norm(err_local)

    def generate_cycloid_waypoints(self, start_pos, target_pos, step_height, num_points):
        start_pos = np.array(start_pos)
        target_pos = np.array(target_pos)
        waypoints = []
        for i in range(num_points):
            phase = i / (num_points - 1) if num_points > 1 else 1.0
            theta = phase * 2 * np.pi
            cycloid_x = (theta - np.sin(theta)) / (2 * np.pi)
            cycloid_z = (1 - np.cos(theta)) / 2
            pos = start_pos + (target_pos - start_pos) * cycloid_x
            pos[2] += step_height * cycloid_z
            waypoints.append(pos.copy())
        return waypoints

    def get_joint_angles(self, q, leg_idx=None):
        """Extract joint angles from pinocchio q. Returns 12 angles or 3 for one leg."""
        if self.has_floating_base:
            q_joints = q[7:]
        else:
            q_joints = q
        angles = []
        i = 0
        while i < len(q_joints):
            angles.append(q_joints[i])
            if i + 3 < len(q_joints):
                thigh_sin = q_joints[i + 1]
                thigh_cos = q_joints[i + 2]
                angles.append(np.arctan2(thigh_sin, thigh_cos))
                angles.append(q_joints[i + 3])
                i += 4
            else:
                break
        angles = np.array(angles)
        if leg_idx is not None:
            return angles[leg_idx * 3: leg_idx * 3 + 3]
        return angles

    def set_joint_angles(self, q, angles_12):
        """Set all 12 joint angles in pinocchio q."""
        q = q.copy()
        for leg in range(4):
            base_idx = 7 + leg * 4
            joint_idx = leg * 3
            q[base_idx + 0] = angles_12[joint_idx]
            q[base_idx + 1] = np.sin(angles_12[joint_idx + 1])
            q[base_idx + 2] = np.cos(angles_12[joint_idx + 1])
            q[base_idx + 3] = angles_12[joint_idx + 2]
        return q

    def update_feet_positions(self, q):
        """Update stored feet world positions from current q."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        self.feet_world_positions = [
            self.data.oMf[fid].translation.copy()
            for fid in self.foot_frame_ids
        ]

    def compute_spline_target_cmd(self, body_start, body_goal, num_steps, plot=False):

        # 1. Definisci SOLO i keyframe (inizio e fine)
        key_alphas = [0.0, 1.0]
        key_positions = [body_start, body_goal]

        # 2. Crea la spline forzando la velocità a 0 agli estremi (bc_type='clamped')
        cubic_spline = CubicSpline(key_alphas, key_positions, bc_type='clamped')

        # 3. Crea il vettore di "tempo" (alphas) per i tuoi step
        alphas = np.linspace(0, 1.0, num_steps)

        # 4. Interroga la spline per ottenere posizioni e velocità morbide
        body_positions = cubic_spline(alphas)
        body_velocities = cubic_spline(alphas, 1)  # Derivata prima = velocità

        if plot:
            # --- Creazione dei Grafici ---
            # Creiamo una figura con 3 righe e 1 colonna, condividendo l'asse X
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            labels = ['X', 'Y', 'Z']
            colors_pos = ['blue', 'green', 'red']
            colors_vel = ['cyan', 'lime', 'orange']

            for i in range(3):
                ax1 = axes[i]

                # Plot Posizione sul primo asse Y (sinistra)
                line1 = ax1.plot(alphas, body_positions[:, i], color=colors_pos[i], linewidth=2.5, label=f'Posizione {labels[i]}')
                ax1.set_ylabel(f'Posizione {labels[i]}', color=colors_pos[i], fontsize=12)
                ax1.tick_params(axis='y', labelcolor=colors_pos[i])
                ax1.grid(True, alpha=0.3)

                # Crea un secondo asse Y (destra) per la Velocità, che condivide l'asse X
                ax2 = ax1.twinx()
                line2 = ax2.plot(alphas, body_velocities[:, i], color=colors_vel[i], linewidth=2.5, linestyle='--', label=f'Velocità {labels[i]}')
                ax2.set_ylabel(f'Velocità {labels[i]}', color=colors_vel[i], fontsize=12)
                ax2.tick_params(axis='y', labelcolor=colors_vel[i])

                # Uniamo le legende in un unico riquadro in alto a sinistra
                lines = line1 + line2
                ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')

            # Impostazioni generali della figura
            axes[-1].set_xlabel('Alpha (Tempo normalizzato)', fontsize=12)
            fig.suptitle('Evoluzione delle coordinate 3D: Posizione vs Velocità', fontsize=16)

            plt.tight_layout()
            plt.show()

        return body_positions, body_velocities


# =============================================================================
# IK ground truth
# =============================================================================

def generate_combined_ik(kin : WalkKinematics, q_start, body_start, body_goal,
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
    trajectory_classic = []
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
            q_current, _, _ = kin.compute_leg_ik_body_centric(leg_idx, target, body_vel, q_current)
            q_current2, _, _ = kin.compute_leg_ik(leg_idx, target, q_current)

        # Stepping leg: cycloid waypoint (body frame → world)
        foot_world = kin.body_to_world(cycloid_waypoints[step], q_current)
        q_current, _, _ = kin.compute_leg_ik(stepping_leg, foot_world, q_current)
        q_current2, _, _ = kin.compute_leg_ik(stepping_leg, foot_world, q_current)

        angles = kin.get_joint_angles(q_current)
        trajectory.append(angles)

        angles2 = kin.get_joint_angles(q_current2)
        trajectory_classic.append(angles2)

    return np.array(trajectory), q_current, np.array(trajectory_classic)


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

kin = WalkKinematics(pin_model, FOOT_FRAMES, body_frame_name="trunk")
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
    t_model, t_diffusion, t_checkpoint = load_model(trunk_model_path, device=device)
    print(f"  num_steps={t_checkpoint['num_steps']}, num_joints={t_checkpoint['num_joints']}")

    print(f"Loading step model from {step_model_path}...")
    s_model, s_diffusion, s_checkpoint = load_model(step_model_path, device=device)
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
            l_ik_classic, = ax.plot([], [], "g-", label="IK Classic", alpha=0.5)
            if leg == 0 and j == 0:
                ax.legend(loc="upper left", fontsize=7)
            cmp_lines[(leg, j)] = (l_diff, l_ik, l_ik_classic)
    fig2.tight_layout()
    plt.show()

    for walk_step in range(num_walk_steps):
        leg_idx = walk_step % 4
        leg_name = LEG_NAMES[leg_idx]

        kin.update_feet_positions(q_current)

        print(f"\n{'='*60}")
        print(f"Walk step {walk_step + 1}/{num_walk_steps}: {leg_name} stepping")
        print(f"  Body pos: {np.round(body_pos, 4)}")
        print(f"  Delta body: {delta_body}, Delta foot: {delta_foot}")

        body_goal = body_pos + delta_body

        # FK correction: observe where the stepping foot actually is vs neutral,
        # adjust delta_foot to compensate for drift
        actual_foot_body = kin.get_foot_position_body(leg_idx, q_current)
        foot_error = neutral_foot_body[leg_idx] - actual_foot_body
        corrected_delta_foot = delta_foot + foot_error
        print(f"  Foot correction {leg_name}: {np.round(foot_error, 4)}")

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
        ik_traj, _, ik_traj_classic = generate_combined_ik(
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
                l_diff, l_ik, l_ik_classic = cmp_lines[(leg, j)]
                l_diff.set_data(steps_x, diff_traj[:, ji])
                l_ik.set_data(steps_x, ik_traj[:, ji])
                l_ik_classic.set_data(steps_x, ik_traj_classic[:, ji])
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

    delta_body = np.array([0.02, 0.005, 0.0])
    delta_foot = np.array([0.08, 0.02, 0.0])

    evaluate_walk(
        trunk_model_path ="still/diffusion_model.pt",
        step_model_path  ="step/diffusion_model.pt",
        num_walk_steps   = 200,
        delta_body       = delta_body,
        delta_foot       = delta_foot,
        device           = "cuda",
        ddim_steps       = 0
    )
