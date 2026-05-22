import pinocchio as pin
import numpy as np

from .constants import (
    URDF_PATH, FOOT_FRAMES, BODY_HEIGHT, THIGH_ANGLE, CALF_ANGLE,
    HIP_LOWER, HIP_UPPER, CALF_LOWER, CALF_UPPER,
)
from scipy.interpolate import CubicSpline


def build_neutral_q(model):
    """Build the standard neutral standing configuration for the Aliengo."""
    q = pin.neutral(model)
    q[2] = BODY_HEIGHT
    for leg in range(4):
        base_idx = 7 + leg * 4
        q[base_idx + 0] = 0.0
        q[base_idx + 1] = np.sin(THIGH_ANGLE)
        q[base_idx + 2] = np.cos(THIGH_ANGLE)
        q[base_idx + 3] = CALF_ANGLE
    return q


def rotation_rpy(roll, pitch, yaw):
    """Build a 3x3 rotation matrix from roll-pitch-yaw angles."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def check_joint_limits(angles_12):
    """Return True if all 12 joint angles are within URDF limits."""
    for leg in range(4):
        hip = angles_12[leg * 3]
        calf = angles_12[leg * 3 + 2]
        if hip < HIP_LOWER or hip > HIP_UPPER:
            return False
        if calf < CALF_LOWER or calf > CALF_UPPER:
            return False
    return True


def build_pinocchio_model(urdf_path=None):
    """Build a Pinocchio model + visual/collision geometry from the Aliengo URDF."""
    if urdf_path is None:
        urdf_path = URDF_PATH
    mesh_dir = __import__("os").path.dirname(__import__("os").path.abspath(urdf_path))

    pin_model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    pin_data = pin_model.createData()

    try:
        collision_model = pin.buildGeomFromUrdf(
            pin_model, urdf_path, pin.GeometryType.COLLISION, package_dirs=[mesh_dir],
        )
        visual_model = pin.buildGeomFromUrdf(
            pin_model, urdf_path, pin.GeometryType.VISUAL, package_dirs=[mesh_dir],
        )
    except Exception:
        collision_model = pin.GeometryModel()
        visual_model = pin.GeometryModel()

    return pin_model, pin_data, collision_model, visual_model


class QuadrupedKinematics:
    """Unified kinematics helper for the Aliengo quadruped.

    Merges the functionality of QuadrupedStance, QuadrupedLegStepper and
    WalkKinematics into a single class.
    """

    def __init__(self, model, foot_frame_names=None, body_frame_name="trunk"):
        if foot_frame_names is None:
            foot_frame_names = FOOT_FRAMES
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

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_stance(self, q_init):
        """Perform FK on *q_init* and store the resulting foot world positions."""
        pin.forwardKinematics(self.model, self.data, q_init)
        pin.updateFramePlacements(self.model, self.data)
        self.feet_world_positions = [
            self.data.oMf[fid].translation.copy()
            for fid in self.foot_frame_ids
        ]
        return q_init

    def update_feet_positions(self, q):
        """Re-compute and store feet world positions from *q*."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        self.feet_world_positions = [
            self.data.oMf[fid].translation.copy()
            for fid in self.foot_frame_ids
        ]

    # ------------------------------------------------------------------
    # Body-frame helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Body pose
    # ------------------------------------------------------------------

    def set_body_pose(self, q, body_translation, body_rotation=None):
        """Set the floating-base translation (and optional rotation matrix)."""
        q = q.copy()
        if self.has_floating_base:
            q[0:3] = body_translation
            if body_rotation is not None:
                quat = pin.Quaternion(body_rotation)
                q[3:7] = np.array([quat.x, quat.y, quat.z, quat.w])
        return q

    # ------------------------------------------------------------------
    # Joint angle helpers
    # ------------------------------------------------------------------

    def get_joint_angles(self, q, leg_idx=None):
        """Extract 12 joint angles (or 3 for a single leg) from Pinocchio q.

        Handles the sin/cos encoding of thigh angles in the FreeFlyer model.
        """
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
        """Write 12 joint angles into a Pinocchio q vector."""
        q = q.copy()
        for leg in range(4):
            base_idx = 7 + leg * 4
            ji = leg * 3
            q[base_idx + 0] = angles_12[ji]
            q[base_idx + 1] = np.sin(angles_12[ji + 1])
            q[base_idx + 2] = np.cos(angles_12[ji + 1])
            q[base_idx + 3] = angles_12[ji + 2]
        return q

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    def compute_leg_ik(self, leg_idx, target_pos_world, q_current,
                       eps=1e-4, max_iter=100, dt=0.1):
        """Solve IK for a single leg to reach *target_pos_world*.

        Returns (q, converged, residual_norm).
        """
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
                self.model, self.data, q, frame_id, pin.LOCAL_WORLD_ALIGNED,
            )[:3, leg_v_start:leg_v_end]

            damp = 1e-6
            v = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(3), err)

            v_full = np.zeros(self.model.nv)
            v_full[leg_v_start:leg_v_end] = v
            q = pin.integrate(self.model, q, v_full * dt)

        return q, False, np.linalg.norm(err)

    def solve_stance(self, body_translation, body_rotation=None, q_init=None):
        """Solve IK for all 4 legs to keep feet at their stored world positions.

        Returns (q, converged).
        """
        body_translation = np.array(body_translation)
        if body_rotation is None:
            body_rotation = np.eye(3)
        q = self.set_body_pose(q_init, body_translation, body_rotation)

        for leg_idx in range(len(self.foot_frame_ids)):
            target_pos = self.feet_world_positions[leg_idx]
            q, converged, _ = self.compute_leg_ik(leg_idx, target_pos, q)
            if not converged:
                return q, False
        return q, True

    # ------------------------------------------------------------------
    # Cycloid trajectory generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_cycloid_waypoints(start_pos, target_pos, step_height, num_points):
        """Generate cycloid-arc waypoints between two positions."""
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

        return body_positions, body_velocities

    def compute_leg_clik_body_centric(self, leg_idx, target_pos_world, body_cmd_vel_world, q_current,
                                      target_vel_world=None, dt=0.05, kd=10.0, max_joint_vel=5.0):
        """
        Risolve il CLIK nel frame locale (Body-Centric) per inseguire una traiettoria dinamica.
        Esegue UN SOLO step di integrazione basato sul tempo (dt).
        """
        frame_id = self.foot_frame_ids[leg_idx]
        q = q_current.copy()

        # Indici dei giunti
        v_base = 6 if self.has_floating_base else 0
        leg_v_start = v_base + leg_idx * 3
        leg_v_end = leg_v_start + 3

        # Sicurezza sui tipi di dato (meglio di moltiplicare per 0.)
        if target_vel_world is None:
            target_vel_world = np.zeros(3)

        # 1. Aggiornamento Cinematica (Fatto una sola volta!)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # 2. Estrazione della posa del Body
        oMb = self.data.oMf[self.body_frame_id]

        # 3. CONVERSIONE POSIZIONI DA WORLD A BODY LOCALE
        target_pos_local = oMb.actInv(np.array(target_pos_world))
        current_pos_local = oMb.actInv(self.data.oMf[frame_id].translation)

        # Errore puro visto dal body
        err_local = target_pos_local - current_pos_local

        # 4. CONVERSIONE VELOCITÀ DA WORLD A BODY LOCALE
        R_body_inv = oMb.rotation.T
        foot_vel_local = R_body_inv @ np.array(target_vel_world)
        body_vel_local = R_body_inv @ np.array(body_cmd_vel_world)

        # 5. VELOCITÀ RELATIVA (Tapis Roulant)
        target_vel_local = foot_vel_local - body_vel_local

        # 6. Legge CLIK (Feedforward + Proporzionale sull'errore)
        v_task_local = target_vel_local + (kd * err_local)

        # 7. JACOBIANO ROTAZIONALE
        J_world = pin.computeFrameJacobian(
            self.model, self.data, q, frame_id, pin.LOCAL_WORLD_ALIGNED
        )[:3, leg_v_start:leg_v_end]

        J_local = R_body_inv @ J_world

        # 8. Risoluzione Damped Least Squares
        damp = 1e-6
        v_joint = J_local.T @ np.linalg.solve(J_local @ J_local.T + damp * np.eye(3), v_task_local)

        # 9. Sicurezza (Clipping velocità motori)
        v_joint = np.clip(v_joint, -max_joint_vel, max_joint_vel)

        # 10. INTEGRAZIONE NEL TEMPO
        v_full = np.zeros(self.model.nv)
        v_full[leg_v_start:leg_v_end] = v_joint
        q_new = pin.integrate(self.model, q, v_full * dt)

        # Ritorno la nuova posa e la norma dell'errore (per il logging)
        return q_new, np.linalg.norm(err_local)


    def solve_stance_vel(self, body_translation, body_rotation=None, q_init=None, body_vel=None):
        """Solve IK for all 4 legs to keep feet at their stored world positions.

        Returns (q, converged).
        """
        body_translation = np.array(body_translation)
        if body_rotation is None:
            body_rotation = np.eye(3)
        q = self.set_body_pose(q_init, body_translation, body_rotation)
        # q = self.set_body_pose(q_init, body_translation) # No rotation dataset (to consider?)

        for leg_idx in range(len(self.foot_frame_ids)):
            target_pos = self.feet_world_positions[leg_idx]
            q, _ = self.compute_leg_clik_body_centric(
                leg_idx, target_pos, body_vel, q
            )
        return q, True
