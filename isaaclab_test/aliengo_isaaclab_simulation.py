def get_isaaclab_diffusion_simulation():
    """Get Isaaclab diffusion simulation."""
    import sys
    import os
    import select
    import numpy as np
    import tty
    import termios

    class AliengoIsaaclabSimulation:
        def __init__(
                self,
                aliengo_isaaclab_diffusion
        ):
            self._physics_rate = 200
            self._physics_dt = 1 / self._physics_rate


            self._aliengo_diff = aliengo_isaaclab_diffusion

            # Goal position
            self._goal_pos = None
            self._trajectory_generated = False

        def setup(self):
            """Initialize after world is ready."""
            self._aliengo_diff.env.reset()
            self._aliengo_diff.initialize()

        def set_goal(self, delta_pos: np.ndarray):
            """Set a new delta position and generate trunk trajectory."""
            self._goal_pos = delta_pos  # Store delta for reference

            # Get current joint positions in model order
            artic_positions = self._aliengo_diff.get_joint_positions()
            current_joints = self._aliengo_diff._articulation_to_model_order(artic_positions)

            print(f"Generating trunk trajectory with delta_pos={delta_pos}")

            delta = np.concatenate([delta_pos, np.zeros_like(delta_pos)], axis=0)

            trajectory = self._aliengo_diff.generate_trajectory(delta, current_joints)
            self._aliengo_diff.set_trajectory(trajectory)
            print(f"Trunk trajectory: {trajectory.shape[0]} steps, {trajectory.shape[1]} joints")

        def set_step_goal(self, leg_idx: int, delta_foot_pos: np.ndarray):
            """Set a foot delta for a specific leg and generate step trajectory."""
            leg_names = ["FL", "FR", "RL", "RR"]
            # Get current leg joints from model order
            artic_positions = self._aliengo_diff.get_joint_positions()
            current_joints = self._aliengo_diff._articulation_to_model_order(artic_positions)
            leg_indices = self._aliengo_diff.LEG_MODEL_INDICES[leg_idx]
            current_leg = current_joints[leg_indices]

            print(f"Generating {leg_names[leg_idx]} step trajectory with delta_foot={delta_foot_pos}")

            trajectory = self._aliengo_diff.generate_leg_step_trajectory(leg_idx, delta_foot_pos, current_leg)
            self._aliengo_diff.set_leg_step_trajectory(leg_idx, trajectory, delta_foot_pos)
            print(f"{leg_names[leg_idx]} step trajectory: {trajectory.shape[0]} steps, {trajectory.shape[1]} joints")

        def on_physics_step(self, step_size):
            """Physics callback - execute trajectory."""
            self._aliengo_diff.forward(step_size)

        def wait_for_trajectories(self, simulation_app):
            """Step simulation until all active trajectories are done."""
            while simulation_app.is_running():
                self._aliengo_diff.env.step(self._aliengo_diff.target_pos)
                trunk_done = self._aliengo_diff._current_trajectory is None
                steps_done = all(t is None for t in self._aliengo_diff._step_trajectories.values())
                if trunk_done and steps_done:
                    break

        def _read_key_nonblocking(self):
            """Check for a keypress without blocking. Returns the key or None."""
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
            return None

        def _execute_advance(self, current_leg):
            """Advance: trunk forward + step current leg."""
            delta_pos = np.array([0.02, 0.0, 0.0])
            delta_foot = np.array([-0.08, 0.0, 0.0])
            self.set_goal(delta_pos)
            self.set_step_goal(current_leg, delta_foot)

        def _execute_reset(self):
            """Reset robot by directly setting pose and joints to starting state."""
            # Cancel any active trajectories and clear anchors
            self._aliengo_diff._current_trajectory = None
            self._aliengo_diff._held_positions = None
            for i in range(4):
                self._aliengo_diff._step_trajectories[i] = None

            # Directly set world pose and joint positions
            # self._aliengo_diff.robot.set_world_pose(self._start_pos, self._start_rot)
            # self._aliengo_diff.robot.set_joint_positions(self._aliengo_diff.DEFAULT_JOINT_POS)
            #
            # # Zero out velocities
            # self._aliengo_diff.robot.set_joint_velocities(np.zeros(12))
            self._aliengo_diff.env.reset()

            print("Reset to starting configuration.")

        def _trajectories_active(self):
            """Check if any trajectories are still executing."""
            trunk_active = self._aliengo_diff._current_trajectory is not None
            steps_active = any(t is not None for t in self._aliengo_diff._step_trajectories.values())
            return trunk_active or steps_active

        def run(self, simulation_app):
            """Main simulation loop with interactive control."""
            self.setup()

            # Let simulation settle
            for _ in range(100):
                # self._aliengo_diff.env.step(self._aliengo_diff.DEFAULT_JOINT_POS_TENSOR)
                self._aliengo_diff.env.sim.step(render=True)

            # Capture starting state
            self._start_pos, self._start_rot = self._aliengo_diff.get_world_pose()
            self._start_foot_body = {
                i: self._aliengo_diff.get_foot_body_position(i) for i in range(4)
            }

            current_leg = 0
            waiting_for_key = True

            print("\n=== Interactive Control ===")
            print("  Enter  : advance (trunk + step current leg)")
            print("  r      : reset to starting configuration")
            print("  q      : quit")
            print(f"  Current leg: {self._aliengo_diff.LEG_NAMES[current_leg]}")
            print("==========================\n")

            # Set terminal to raw mode so we get keypresses immediately
            # old_settings = termios.tcgetattr(sys.stdin)
            try:
                # tty.setcbreak(sys.stdin.fileno())
                i = 0
                while simulation_app.is_running():
                    self._aliengo_diff.env.sim.step(render=True)

                    if waiting_for_key:
                        # key = self._read_key_nonblocking()
                        key = input()
                        if not key == "q" or not key == "r":
                            key = "ENTER"
                        if key is None:
                            continue

                        if key == "q":
                            break
                        elif key == "r":
                            self._execute_reset()
                            # Let physics settle after direct reset
                            for _ in range(50):
                                # self._world.step(render=True)
                                # self._aliengo_diff.env.step(self._aliengo_diff.DEFAULT_JOINT_POS_TENSOR)
                                self._aliengo_diff.env.sim.step(render=True)
                            self._aliengo_diff.print_foot_status()
                            print(f"\n[leg={self._aliengo_diff.LEG_NAMES[current_leg]}] Enter/r/q >")
                        else:
                            # Enter or any other key = advance
                            self._aliengo_diff.print_foot_status()
                            print(f"\nAdvancing with leg {self._aliengo_diff.LEG_NAMES[current_leg]}...")
                            self._execute_advance(current_leg)
                            waiting_for_key = False
                            i = 0

                    else:
                        # Waiting for trajectories to finish
                        if not self._trajectories_active():
                            current_leg = (current_leg + 1) % 4
                            waiting_for_key = True
                            self._aliengo_diff.print_foot_status()
                            print(f"\n[leg={self._aliengo_diff.LEG_NAMES[current_leg]}] Enter/r/q >")
                        else:
                            print(f"Traj still active - Step {i}")
                            self.on_physics_step(self._physics_dt)

            finally:
                # Restore terminal settings
                # termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                print("Quitting...")

            simulation_app.close()

    return AliengoIsaaclabSimulation