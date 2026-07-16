import numpy
import torch
from still.diffusion_train import load_model, generate_trajectory
import numpy as np
import time
import random
import os
from ink_kin_stance.kinematics import QuadrupedKinematics, rotation_rpy

class StillDiffusionHelper:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model from {model_path}...")
        self.model, self.diffusion, self.checkpoint = load_model(model_path, device=self.device)
        self.model.eval()

        self.delta_mean = torch.tensor(self.checkpoint["delta_mean"], dtype=torch.float32).to(self.device)
        self.delta_std = torch.tensor(self.checkpoint["delta_std"], dtype=torch.float32).to(self.device)
        self.traj_mean = torch.tensor(self.checkpoint["traj_mean"], dtype=torch.float32).to(self.device)
        self.traj_std = torch.tensor(self.checkpoint["traj_std"], dtype=torch.float32).to(self.device)

    def generate_trajectory_multienv(self, delta : torch.Tensor, current_joints : torch.Tensor, ddim_steps=0):
        """
        Generate a foot trajectory given delta position and current joint values for multiple envs.

        delta: (num_envs, 6) tensor - [dx, dy, dz, droll, dpitch, dyaw]
        current_joints: (num_envs, 12) tensor - current joint angles
        ddim_steps: number of DDIM denoising steps (default 50, set to 0 for full DDPM)

        Returns: (num_envs, num_steps, num_joints) tensor trajectory
        """

        delta_norm = (delta - self.delta_mean) / self.delta_std
        current_joints_norm = (current_joints - self.traj_mean) / self.traj_std

        condition = torch.concatenate([delta_norm, current_joints_norm], dim=1)

        # Sample trajectory
        shape = (condition.shape[0], self.checkpoint["num_steps"], self.checkpoint["num_joints"])
        if ddim_steps > 0:
            trajectory = self.diffusion.ddim_sample(self.model, condition, shape, ddim_steps=ddim_steps)
        else:
            trajectory = self.diffusion.sample(self.model, condition, shape)

        # Denormalize
        trajectory = trajectory * self.traj_std + self.traj_mean

        return trajectory


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Se usi multi-GPU
    # Forziamo PyTorch ad essere completamente deterministico (può rallentare leggermente l'esecuzione)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed impostato globalmente a: {seed}")


def run_benchmark(func, n_runs=10):
    """
    Esegue una funzione n volte.
    """
    times = []

    # Warm-up (per inizializzare la GPU o la cache)
    func()

    for _ in range(n_runs):
        t_start = time.perf_counter()
        func()
        times.append(time.perf_counter() - t_start)

    times_arr = np.array(times)
    return {
        'mean' : np.mean(times_arr),
        'std' : np.std(times_arr)
    }

def start_with_pin(checkpoint):
    import pinocchio as pin
    import os

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
    BODY_HEIGHT = 0.42
    THIGH_ANGLE = 0.8
    CALF_ANGLE = -1.6
    # Ranges matching generate_still_data.py
    X_RANGE, Y_RANGE, Z_RANGE = 0.10, 0.05, 0.1
    # ROLL_RANGE, PITCH_RANGE, YAW_RANGE = 0.25, 0.25, 0.3
    ROLL_RANGE, PITCH_RANGE, YAW_RANGE = 0.0, 0.0, 0.0
    num_steps = checkpoint["num_steps"]

    stance = QuadrupedKinematics(model, FOOT_FRAMES)

    q_init = pin.neutral(model)
    q_init[2] = BODY_HEIGHT
    for leg in range(4):
        base_idx = 7 + leg * 4
        q_init[base_idx + 0] = 0.0
        q_init[base_idx + 1] = np.sin(THIGH_ANGLE)
        q_init[base_idx + 2] = np.cos(THIGH_ANGLE)
        q_init[base_idx + 3] = CALF_ANGLE

    q_neutral = stance.init_stance(q_init)
    # Save neutral feet so perturbation doesn't accumulate across tests
    feet_neutral = [fp.copy() for fp in stance.feet_world_positions]

    # Perturb foot positions from neutral (matching data gen)
    # perturbed_feet = []
    # for i in range(4):
    #     fp = feet_neutral[i].copy()
    #     fp[0] += random.uniform(-0.08, 0.08)
    #     fp[1] += random.uniform(-0.05, 0.05)
    #     fp[2] += random.uniform(-0.04, 0.04)
    #     perturbed_feet.append(fp)
    # stance.feet_world_positions = perturbed_feet

    # Random start position + RPY
    start_pos = np.array([
        0,
        0,
        BODY_HEIGHT
    ])
    start_rpy = np.array([
        random.uniform(-ROLL_RANGE, ROLL_RANGE),
        random.uniform(-PITCH_RANGE, PITCH_RANGE),
        random.uniform(-YAW_RANGE, YAW_RANGE),
    ])

    # Random goal position + RPY
    goal_pos = start_pos + np.array([
        0.1,
        0,
        0
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

    print(f"Start pos: {start_pos}, rpy: {np.rad2deg(start_rpy).round(1)} deg")
    print(f"Goal  pos: {goal_pos}, rpy: {np.rad2deg(goal_rpy).round(1)} deg")

    # Compute delta for diffusion conditioning (pos + rpy)
    delta = np.concatenate([goal_pos - start_pos, goal_rpy - start_rpy])

    print(f"Delta pos: {delta}")
    print(f"Current joints:  {current_joints}")

    ik_trajectory = []
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
        angles = stance.get_joint_angles(q_new)
        ik_trajectory.append(angles)

    ik_trajectory = np.array(ik_trajectory)

    return delta, current_joints, ik_trajectory

def test_diff(model_path, perform_multi=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, diffusion, checkpoint = load_model(model_path, device=device)

    delta, current_joints, ik_trajectory = start_with_pin(checkpoint)

    # conversione in tensori e duplica per 4000 envs
    delta_tensor = torch.tensor(numpy.array(delta), dtype=torch.float32).to(device)
    current_joints_tensor = torch.tensor(numpy.array(current_joints), dtype=torch.float32).to(device)
    NUM_ENV = 2 # set here the num_envs for the multienv run
    delta_tensor = delta_tensor.repeat(NUM_ENV, 1)
    current_joints_tensor = current_joints_tensor.repeat(NUM_ENV, 1)

    helper = StillDiffusionHelper(model_path, device=device)

    if perform_multi:
        N_RUNS = 20

        # 1. Test single run
        bench_single = lambda: generate_trajectory(
            model, diffusion, checkpoint, delta, current_joints,
            device=device, ddim_steps=0
        )

        # 2. Test multienv run
        bench_multi = lambda: helper.generate_trajectory_multienv(
            delta_tensor, current_joints_tensor
        )

        stats_single = run_benchmark(bench_single, n_runs=N_RUNS)
        stats_multi = run_benchmark(bench_multi, n_runs=N_RUNS)

        print(f"\nRisultati Single: {stats_single['mean']:.4f}s ± {stats_single['std']:.4f}s")
        print(f"Risultati Multi:  {stats_multi['mean']:.4f}s ± {stats_multi['std']:.4f}s")
    else:
        t_start = time.perf_counter()
        single_traj = generate_trajectory(
            model, diffusion, checkpoint,
            delta, current_joints,
            device=device, ddim_steps=0
        )
        t_diff = time.perf_counter() - t_start
        print("Time for pred single diff trajectory: ", t_diff)

        t_start = time.perf_counter()
        multi_traj = helper.generate_trajectory_multienv(
            delta_tensor, current_joints_tensor,
        )
        t_diff = time.perf_counter() - t_start
        print("Time for pred multi diff trajectory: ", t_diff)

        from eval_diffusion import plot_trajectories
        plot_trajectories(
            ik_trajectory=ik_trajectory,
            diff_trajectory=single_traj,
            helper_trajectory=multi_traj[0, :].cpu().numpy(),
            delta=delta
        )
        # numpy.save("ik_trajectory.npy", ik_trajectory)


    return

if __name__ == "__main__":

    test_diff(
        model_path="/home/etosin/Documents/ink_kin_stance/still/diffusion_model.pt",
        perform_multi=False
    )