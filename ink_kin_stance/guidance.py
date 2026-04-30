import numpy as np
import torch
import pinocchio as pin

from .constants import FOOT_FRAMES


def ddim_sample_guided(
    diffusion,
    model,
    condition,
    shape,
    ddim_steps=50,
    eta=0.0,
    guidance_fn=None,
    guidance_scale=1.0,
    guidance_start=0.4,
):
    """DDIM sampling with optional stability guidance in x0-prediction space.

    guidance_fn    : callable(x0_pred_norm) -> grad tensor, same shape as x0_pred.
    guidance_scale : overall step size for guidance.
    guidance_start : fraction of denoising steps before which guidance is disabled.

    Returns trajectory tensor of *shape*.
    """
    device = condition.device
    x = torch.randn(shape, device=device)

    step_size = diffusion.num_timesteps // ddim_steps
    timesteps = list(reversed(list(range(0, diffusion.num_timesteps, step_size))))

    for i, t in enumerate(timesteps):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        with torch.no_grad():
            predicted_noise = model(x, t_batch, condition)

        alpha_t = diffusion.alphas_cumprod[t]
        alpha_prev = (
            diffusion.alphas_cumprod[timesteps[i + 1]]
            if i + 1 < len(timesteps)
            else torch.tensor(1.0, device=device)
        )

        x0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

        if guidance_fn is not None:
            frac = i / max(len(timesteps) - 1, 1)
            if frac >= guidance_start:
                strength = guidance_scale * (
                    (frac - guidance_start) / (1.0 - guidance_start + 1e-8)
                ) ** 1.5
                with torch.no_grad():
                    grad = guidance_fn(x0_pred)
                grad_mag = torch.norm(grad).item()
                if grad_mag > 1e-8:
                    x0_pred = x0_pred - strength * grad / grad_mag

        eps = (x - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(
            torch.clamp(1.0 - alpha_t, min=1e-8)
        )
        sigma = eta * torch.sqrt(
            (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
        )
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_prev - sigma ** 2, min=0.0)) * eps

        x = torch.sqrt(alpha_prev) * x0_pred + dir_xt
        if sigma > 0 and i + 1 < len(timesteps):
            x = x + sigma * torch.randn_like(x)

    return x


class StabilityGuidance:
    """Callable returning dJ/dx (normalised space) for guided DDIM sampling.

    Pinocchio joint ordering must match the diffusion model ordering:
        leg-major: FL[hip,thigh,calf], FR, RL, RR.
    """

    def __init__(
        self,
        pin_model,
        pin_data,
        body_pos,
        body_quat_xyzw,
        planted_foot_positions,
        traj_mean,
        traj_std,
        full_12_joints,
        stepping_leg=-1,
        margin=0.020,
        lambda_com=3.0,
        lambda_slip=1.0,
        protect_start=1,
        protect_end=4,
    ):
        self.model = pin_model
        self.data = pin_data

        self.body_pos = np.array(body_pos, dtype=np.float64)
        self.body_quat_xyzw = np.array(body_quat_xyzw, dtype=np.float64)
        self.planted_positions = np.array(planted_foot_positions, dtype=np.float64)

        self.traj_mean = np.array(traj_mean, dtype=np.float64)
        self.traj_std = np.array(traj_std, dtype=np.float64)
        self.full_12_joints = np.array(full_12_joints, dtype=np.float64)

        self.stepping_leg = stepping_leg
        self.margin = margin
        self.lambda_com = lambda_com
        self.lambda_slip = lambda_slip

        self._support_2d = self._sort_ccw(self.planted_positions[:, :2])
        self._initial_foot_positions = self.planted_positions.copy()

        self._protect_start = protect_start
        self._protect_end = protect_end
        self._taper_mask = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, x0_pred_norm):
        device = x0_pred_norm.device
        n_joints = x0_pred_norm.shape[2]
        num_steps = x0_pred_norm.shape[1]
        x_np = x0_pred_norm.detach().cpu().numpy()[0]

        if self._taper_mask is None or len(self._taper_mask) != num_steps:
            self._taper_mask = self._make_taper_mask(num_steps)

        x_phys = x_np * self.traj_std + self.traj_mean

        is_step_model = (n_joints == 3)
        grad_phys = np.zeros_like(x_phys)

        for step in range(num_steps):
            if self._taper_mask[step] == 0.0:
                continue

            joints_12 = self._get_joints_12(x_phys[step], is_step_model)
            q = self._build_q(joints_12)

            pin.centerOfMass(self.model, self.data, q)
            Jcom = pin.jacobianCenterOfMass(self.model, self.data, q)

            com = self.data.com[0]
            zmp_2d = com[:2]

            ssm, nearest_i = self._compute_ssm(zmp_2d)
            if ssm < self.margin:
                dJ_dzmp = self._ssm_gradient(nearest_i)
                J_xy = Jcom[:2, 6:]
                grad_full_12 = dJ_dzmp @ J_xy

                if is_step_model:
                    s = self.stepping_leg * 3
                    grad_phys[step] += grad_full_12[s: s + 3]
                else:
                    grad_phys[step] += grad_full_12

            if self.lambda_slip > 0:
                grad_phys[step] += self._slip_gradient(
                    joints_12, q, Jcom, is_step_model
                )

        grad_phys *= self._taper_mask[:, np.newaxis]
        grad_norm = grad_phys * self.traj_std

        return torch.tensor(
            grad_norm[np.newaxis], dtype=torch.float32, device=device,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_taper_mask(self, num_steps):
        mask = np.ones(num_steps, dtype=np.float64)
        ps = self._protect_start
        pe = self._protect_end
        ramp = 2

        mask[:ps] = 0.0
        for k in range(ramp):
            idx = ps + k
            if idx < num_steps:
                mask[idx] = (k + 1) / (ramp + 1)

        mask[num_steps - pe:] = 0.0
        for k in range(ramp):
            idx = num_steps - pe - 1 - k
            if idx >= 0:
                mask[idx] = (k + 1) / (ramp + 1)

        np.clip(mask, 0.0, 1.0, out=mask)
        return mask

    def _get_joints_12(self, optimised_joints, is_step_model):
        if is_step_model:
            joints_12 = self.full_12_joints.copy()
            s = self.stepping_leg * 3
            joints_12[s: s + 3] = optimised_joints
        else:
            joints_12 = optimised_joints.copy()
        return joints_12

    def _build_q(self, joints_12):
        q = pin.neutral(self.model)
        q[0:3] = self.body_pos
        q[3:7] = self.body_quat_xyzw
        for leg in range(4):
            base_idx = 7 + leg * 4
            ji = leg * 3
            q[base_idx + 0] = joints_12[ji]
            q[base_idx + 1] = np.sin(joints_12[ji + 1])
            q[base_idx + 2] = np.cos(joints_12[ji + 1])
            q[base_idx + 3] = joints_12[ji + 2]
        return q

    def _sort_ccw(self, pts_2d):
        c = pts_2d.mean(axis=0)
        angles = np.arctan2(pts_2d[:, 1] - c[1], pts_2d[:, 0] - c[0])
        return pts_2d[np.argsort(angles)]

    def _compute_ssm(self, zmp_2d):
        feet = self._support_2d
        n = len(feet)
        ssm = np.inf
        nearest = 0
        for i in range(n):
            p1, p2 = feet[i], feet[(i + 1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            d = np.dot(normal, zmp_2d - p1) / (np.linalg.norm(normal) + 1e-10)
            if d < ssm:
                ssm = d
                nearest = i
        return ssm, nearest

    def _ssm_gradient(self, nearest_i):
        feet = self._support_2d
        p1 = feet[nearest_i]
        p2 = feet[(nearest_i + 1) % len(feet)]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        normal_unit = normal / (np.linalg.norm(normal) + 1e-10)
        return -self.lambda_com * normal_unit

    def _slip_gradient(self, joints_12, q, Jcom, is_step_model):
        n_joints = 3 if is_step_model else 12
        grad = np.zeros(n_joints)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        if is_step_model:
            planted_legs = [i for i in range(4) if i != self.stepping_leg]
        else:
            planted_legs = list(range(4))

        for k, leg_i in enumerate(planted_legs):
            frame_id = self.model.getFrameId(FOOT_FRAMES[leg_i])
            foot_pos = self.data.oMf[frame_id].translation

            delta = foot_pos - self._initial_foot_positions[k]
            if np.linalg.norm(delta) < 1e-6:
                continue

            J_foot = pin.computeFrameJacobian(
                self.model, self.data, q, frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3, :]

            if is_step_model:
                # Planted leg joints are not in the optimised set — slip
                # gradient is zero by construction for the stepping-leg DOFs.
                pass
            else:
                J_foot_joints = J_foot[:, 6:]
                grad_joint = 2.0 * self.lambda_slip * delta @ J_foot_joints
                grad += grad_joint

        return grad
