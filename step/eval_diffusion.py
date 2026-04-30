import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time
import os
import random
import torch
import matplotlib.pyplot as plt

# Import diffusion model utilities
from diffusion_train import load_model, generate_trajectory, GaussianDiffusion

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
LEG_NAMES = ["FL", "FR", "RL", "RR"]
LEG_IS_RIGHT = {0: False, 1: True, 2: False, 3: True}
BODY_HEIGHT = 0.42
THIGH_ANGLE = 0.8
CALF_ANGLE = -1.6
NUM_STEPS = 20
STEP_HEIGHT = 0.05
CHOSEN_LEG = 0  # Front Left (model trained on this leg)


class QuadrupedLegStepper:
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

    def init_stance(self, q_init=None):
        if q_init is None:
            q_init = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, q_init)
        pin.updateFramePlacements(self.model, self.data)
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

    def compute_leg_ik(self, leg_idx, target_pos_world, q_current, eps=1e-4, max_iter=100, dt=0.1):
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
            return angles[leg_idx * 3 : leg_idx * 3 + 3]
        return angles


# Setup stepper and neutral configuration
stepper = QuadrupedLegStepper(model, FOOT_FRAMES, body_frame_name="trunk")

q_init = pin.neutral(model)
q_init[2] = BODY_HEIGHT
for leg in range(4):
    base_idx = 7 + leg * 4
    q_init[base_idx + 0] = 0.0
    q_init[base_idx + 1] = np.sin(THIGH_ANGLE)
    q_init[base_idx + 2] = np.cos(THIGH_ANGLE)
    q_init[base_idx + 3] = CALF_ANGLE

q_neutral = stepper.init_stance(q_init)
foot_init_body_per_leg = [stepper.get_foot_position_body(i, q_neutral) for i in range(4)]

def plot_trajectories(diff_trajectory, ik_trajectory, delta_pos):
    # Creiamo un asse X normalizzato [0, 1] per far combaciare le lunghezze
    x_diff = np.linspace(0, 1, len(diff_trajectory))
    x_ik = np.linspace(0, 1, len(ik_trajectory))

    # Creazione della figura con 3 subplots sovrapposti
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    joint_names = ['Giunto 1', 'Giunto 2', 'Giunto 3']

    for i in range(3):
        # Plot traiettoria 'diff' (linea continua)
        axes[i].plot(x_diff, diff_trajectory[:, i], label="diff", color='royalblue', linewidth=2)

        # Plot traiettoria 'ik' (linea tratteggiata)
        axes[i].plot(x_ik, ik_trajectory[:, i], label="ik", color='darkorange', linestyle='--', linewidth=2)

        # Personalizzazione del singolo subplot
        axes[i].set_title(f'Confronto {joint_names[i]}', fontsize=12)
        axes[i].set_ylabel('Posizione')
        axes[i].grid(True, linestyle=':', alpha=0.7)
        axes[i].legend()

    # Etichetta comune per l'asse X
    axes[2].set_xlabel('Progresso Traiettoria (Normalizzato 0-1)')
    plt.suptitle(f"Delta pos -- X: {delta_pos[0]:.4f}, Y: {delta_pos[1]:.4f}, Z: {delta_pos[2]:.4f}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Necessario per evitare che il titolo si sovrapponga ai grafici
    plt.show()


def evaluate_diffusion(model_path="diffusion_model.pt", num_tests=10, device="cuda", visualize=True, plot_traj=False, ddim_steps=0):
    """
    Evaluate step diffusion model (trained on FL) against IK ground truth.
    Iterates through all 4 legs, using mirroring for right-side legs.
    """
    # Load diffusion model
    print(f"Loading model from {model_path}...")
    diff_model, diffusion, checkpoint = load_model(model_path, device=device)
    num_steps = checkpoint["num_steps"]
    num_joints = checkpoint["num_joints"]
    print(f"Model loaded: num_steps={num_steps}, num_joints={num_joints}")

    # Setup visualization
    if visualize:
        import meshcat.geometry as g
        import meshcat.transformations as tf
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()

    # Metrics storage
    all_errors = []
    all_max_errors = []
    all_diff_times = []
    all_ik_times = []

    for test_idx in range(num_tests):
        chosen_leg = test_idx % 4
        is_right = LEG_IS_RIGHT[chosen_leg]
        foot_init_body = foot_init_body_per_leg[chosen_leg]
        leg_name = LEG_NAMES[chosen_leg]

        # Generate random start and goal foot positions in body frame
        start_foot_body = foot_init_body + np.array([
            random.uniform(-0.05, 0.05),
            random.uniform(-0.03, 0.03),
            random.uniform(-0.05, 0.05),
        ])
        goal_foot_body = foot_init_body + np.array([
            random.uniform(-0.05, 0.10),
            random.uniform(-0.03, 0.03),
            random.uniform(-0.05, 0.05),
        ])

        print(f"\n{'='*60}")
        print(f"Test {test_idx + 1}/{num_tests} — {leg_name} (mirror={'yes' if is_right else 'no'})")
        print(f"Start foot (body): {start_foot_body}")
        print(f"Goal foot (body):  {goal_foot_body}")

        # Generate ground truth with IK (cycloid trajectory)
        # This also gives us the current_joints for diffusion conditioning
        t_ik_start = time.perf_counter()
        q_current = q_neutral.copy()

        # Place foot at start position
        start_world = stepper.body_to_world(start_foot_body, q_current)
        q_current, _, _ = stepper.compute_leg_ik(chosen_leg, start_world, q_current)

        # Capture initial joint state as first trajectory point
        ik_trajectory = []
        initial_angles = stepper.get_joint_angles(q_current, leg_idx=chosen_leg)
        ik_trajectory.append(initial_angles)

        # Generate cycloid waypoints (skip first since we already have initial state)
        waypoints = stepper.generate_cycloid_waypoints(
            start_foot_body, goal_foot_body, STEP_HEIGHT, NUM_STEPS
        )

        for waypoint in waypoints[1:]:  # Skip first waypoint (same as start)
            target_world = stepper.body_to_world(waypoint, q_current)
            q_new, _, _ = stepper.compute_leg_ik(chosen_leg, target_world, q_current)
            q_current = q_new
            angles = stepper.get_joint_angles(q_new, leg_idx=chosen_leg)
            ik_trajectory.append(angles)

        ik_trajectory = np.array(ik_trajectory)
        t_ik = time.perf_counter() - t_ik_start

        # Generate trajectory with diffusion model
        # Mirror inputs for right-side legs so the FL-trained model can be reused
        delta_pos = goal_foot_body - start_foot_body
        current_joints = initial_angles.copy()

        if is_right:
            delta_pos = delta_pos.copy()
            delta_pos[1] = -delta_pos[1]
            current_joints[0] = -current_joints[0]

        t_start = time.perf_counter()
        diff_trajectory = generate_trajectory(
            diff_model, diffusion, checkpoint,
            delta_pos, current_joints,
            device=device, ddim_steps=ddim_steps,
        )

        # Unmirror output for right-side legs
        if is_right:
            diff_trajectory[:, 0] = -diff_trajectory[:, 0]

        t_diff = time.perf_counter() - t_start

        if plot_traj:
            plot_trajectories(diff_trajectory, ik_trajectory, delta_pos)

        # # Compute errors
        # errors = np.abs(diff_trajectory - ik_trajectory)
        # mean_error = errors.mean()
        # max_error = errors.max()
        # per_joint_error = errors.mean(axis=0)
        #
        # all_errors.append(mean_error)
        # all_max_errors.append(max_error)
        # all_diff_times.append(t_diff)
        # all_ik_times.append(t_ik)
        #
        # print(f"\nResults:")
        # print(f"  Mean Absolute Error: {mean_error:.6f} rad ({np.rad2deg(mean_error):.4f} deg)")
        # print(f"  Max Absolute Error:  {max_error:.6f} rad ({np.rad2deg(max_error):.4f} deg)")
        # print(f"  Per-joint MAE (hip, thigh, calf): {np.round(per_joint_error, 4)}")
        # print(f"  Diffusion time: {t_diff*1000:.2f} ms | IK time: {t_ik*1000:.2f} ms | Speedup: {t_ik/t_diff:.1f}x")

        # Visualize comparison
        if visualize:
            import meshcat.geometry as g
            import meshcat.transformations as tf
            print("\nVisualizing: IK (ground truth) then Diffusion...")
            dt = 0.05

            # Show start and goal foot targets
            start_world = stepper.body_to_world(start_foot_body, q_neutral)
            goal_world = stepper.body_to_world(goal_foot_body, q_neutral)
            viz.viewer["foot_start"].set_object(
                g.Sphere(0.02),
                g.MeshLambertMaterial(color=0x00FF00, opacity=0.7)
            )
            viz.viewer["foot_start"].set_transform(tf.translation_matrix(start_world))
            viz.viewer["foot_goal"].set_object(
                g.Sphere(0.02),
                g.MeshLambertMaterial(color=0xFF0000, opacity=0.7)
            )
            viz.viewer["foot_goal"].set_transform(tf.translation_matrix(goal_world))

            # Show IK trajectory
            print("  Playing IK trajectory...")
            q_display = q_neutral.copy()
            start_world_ik = stepper.body_to_world(start_foot_body, q_display)
            q_display, _, _ = stepper.compute_leg_ik(chosen_leg, start_world_ik, q_display)
            waypoints_viz = stepper.generate_cycloid_waypoints(
                start_foot_body, goal_foot_body, STEP_HEIGHT, NUM_STEPS
            )
            for waypoint in waypoints_viz:
                target_world = stepper.body_to_world(waypoint, q_display)
                q_display, _, _ = stepper.compute_leg_ik(chosen_leg, target_world, q_display)
                viz.display(q_display)
                time.sleep(dt)

            time.sleep(0.5)

            # Show diffusion trajectory
            print("  Playing Diffusion trajectory...")
            q_display = q_neutral.copy()
            for step in range(num_steps):
                print(f"Step {step}/{NUM_STEPS}: {len(diff_trajectory)}--{diff_trajectory[step]}")
                diff_angles = diff_trajectory[step]
                # Set leg joints from diffusion output
                base_idx = 7 + chosen_leg * 4
                q_display[base_idx + 0] = diff_angles[0]  # hip
                q_display[base_idx + 1] = np.sin(diff_angles[1])  # thigh sin
                q_display[base_idx + 2] = np.cos(diff_angles[1])  # thigh cos
                q_display[base_idx + 3] = diff_angles[2]  # calf

                viz.display(q_display)
                time.sleep(dt)

            time.sleep(0.5)

    # # Summary statistics
    # print(f"\n{'='*60}")
    # print("SUMMARY")
    # print(f"{'='*60}")
    # print(f"Total tests: {num_tests}")
    # print(f"Mean MAE: {np.mean(all_errors):.6f} rad ({np.rad2deg(np.mean(all_errors)):.4f} deg)")
    # print(f"Std MAE:  {np.std(all_errors):.6f} rad ({np.rad2deg(np.std(all_errors)):.4f} deg)")
    # print(f"Mean Max Error: {np.mean(all_max_errors):.6f} rad ({np.rad2deg(np.mean(all_max_errors)):.4f} deg)")
    # print(f"Worst Max Error: {np.max(all_max_errors):.6f} rad ({np.rad2deg(np.max(all_max_errors)):.4f} deg)")
    # print(f"\nTiming:")
    # print(f"  Diffusion: {np.mean(all_diff_times)*1000:.2f} ms (std: {np.std(all_diff_times)*1000:.2f} ms)")
    # print(f"  IK:        {np.mean(all_ik_times)*1000:.2f} ms (std: {np.std(all_ik_times)*1000:.2f} ms)")
    # print(f"  Avg speedup: {np.mean(all_ik_times)/np.mean(all_diff_times):.1f}x")

    return all_errors, all_max_errors, all_diff_times, all_ik_times


if __name__ == "__main__":

    evaluate_diffusion(
        model_path="diffusion_model.pt",
        num_tests=100,
        device="cuda",
        ddim_steps=0,
    )
