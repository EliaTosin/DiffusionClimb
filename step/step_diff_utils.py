import numpy
import torch
from ink_kin_stance.diffusion.inference import load_model, generate_trajectory
import numpy as np
import time

class StepDiffusionHelper:
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

    def generate_trajectory_multienv(self, delta_pos : torch.Tensor, current_joints : torch.Tensor, ddim_steps=0):
        """
        Generate a foot trajectory given delta position and current joint values for multiple envs.

        delta_pos: (num_envs, 3) tensor - [goal_x - start_x, goal_y - start_y, goal_z - start_z]
        current_joints: (num_envs, 3) tensor - current joint angles
        ddim_steps: number of DDIM denoising steps (default 50, set to 0 for full DDPM)

        Returns: (num_envs, num_steps, num_joints) tensor trajectory
        """

        delta_pos_norm = (delta_pos - self.delta_mean) / self.delta_std
        current_joints_norm = (current_joints - self.traj_mean) / self.traj_std

        condition = torch.concatenate([delta_pos_norm, current_joints_norm], dim=1)

        # Sample trajectory
        shape = (condition.shape[0], self.checkpoint["num_steps"], self.checkpoint["num_joints"])
        if ddim_steps > 0:
            trajectory = self.diffusion.ddim_sample(self.model, condition, shape, ddim_steps=ddim_steps)
        else:
            trajectory = self.diffusion.sample(self.model, condition, shape)

        # Denormalize
        trajectory = trajectory * self.traj_std + self.traj_mean

        return trajectory


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

def start_with_pin():
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
    LEG_NAMES = ["FL", "FR", "RL", "RR"]
    LEG_IS_RIGHT = {0: False, 1: True, 2: False, 3: True}
    BODY_HEIGHT = 0.42
    THIGH_ANGLE = 0.8
    CALF_ANGLE = -1.6
    # BODY_HEIGHT = 0.4
    # THIGH_ANGLE = 0.9
    # CALF_ANGLE = -1.8
    NUM_STEPS = 20
    STEP_HEIGHT = 0.05
    CHOSEN_LEG = 0  # Front Left (model trained on this leg)

    from step.eval_diffusion import QuadrupedLegStepper
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

    chosen_leg = CHOSEN_LEG
    is_right = LEG_IS_RIGHT[chosen_leg]
    foot_init_body = foot_init_body_per_leg[chosen_leg]
    leg_name = LEG_NAMES[chosen_leg]

    # Generate start and goal foot positions in body frame
    start_foot_body = foot_init_body
    goal_foot_body = foot_init_body + np.array([
        0.20,
        0,
        0
    ])

    print(f"Start foot (body): {start_foot_body}")
    print(f"Goal foot (body):  {goal_foot_body}")

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
    delta_pos = goal_foot_body - start_foot_body
    current_joints = initial_angles.copy()

    print(f"Delta pos: {delta_pos}")
    print(f"Current joints:  {current_joints}")

    return delta_pos, current_joints, ik_trajectory

def test_diff(model_path, perform_multi=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, diffusion, checkpoint = load_model(model_path, device=device)

    delta_pos, current_joints, ik_trajectory = start_with_pin()

    # conversione in tensori e duplica per 4000 envs
    delta_pos_tensor = torch.tensor(numpy.array(delta_pos), dtype=torch.float32).to(device)
    current_joints_tensor = torch.tensor(numpy.array(current_joints), dtype=torch.float32).to(device)
    NUM_ENV = 2 # set here the num_envs for the multienv run
    delta_pos_tensor = delta_pos_tensor.repeat(NUM_ENV, 1)
    current_joints_tensor = current_joints_tensor.repeat(NUM_ENV, 1)

    helper = StepDiffusionHelper(model_path, device=device)

    if perform_multi:
        N_RUNS = 20

        # 1. Test single run
        bench_single = lambda: generate_trajectory(
            model, diffusion, checkpoint, delta_pos, current_joints,
            device=device, ddim_steps=0
        )

        # 2. Test multienv run
        bench_multi = lambda: helper.generate_trajectory_multienv(
            delta_pos_tensor, current_joints_tensor
        )

        stats_single = run_benchmark(bench_single, n_runs=N_RUNS)
        stats_multi = run_benchmark(bench_multi, n_runs=N_RUNS)

        print(f"\nRisultati Single: {stats_single['mean']:.4f}s ± {stats_single['std']:.4f}s")
        print(f"Risultati Multi:  {stats_multi['mean']:.4f}s ± {stats_multi['std']:.4f}s")
    else:
        t_start = time.perf_counter()
        single_traj = generate_trajectory(
            model, diffusion, checkpoint,
            delta_pos, current_joints,
            device=device, ddim_steps=0
        )
        t_diff = time.perf_counter() - t_start
        print("Time for pred single diff trajectory: ", t_diff)

        t_start = time.perf_counter()
        multi_traj = helper.generate_trajectory_multienv(
            delta_pos_tensor, current_joints_tensor,
        )
        t_diff = time.perf_counter() - t_start
        print("Time for pred multi diff trajectory: ", t_diff)

        from eval_diffusion import plot_trajectories
        plot_trajectories(ik_trajectory=ik_trajectory, diff_trajectory=single_traj, delta_pos=delta_pos)

        plot_trajectories(ik_trajectory=ik_trajectory, diff_trajectory=multi_traj[0, :].cpu().numpy(), delta_pos=delta_pos)

        plot_trajectories(ik_trajectory=single_traj, diff_trajectory=multi_traj[0, :].cpu().numpy(), delta_pos=delta_pos)

        # numpy.save("ik_trajectory.npy", ik_trajectory)


    return

if __name__ == "__main__":

    test_diff(
        model_path="/home/etosin/Documents/ink_kin_stance/step/diffusion_model.pt",
        perform_multi=False
    )