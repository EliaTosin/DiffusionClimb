import os
import numpy as np

# Project root (parent of this package directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Robot geometry
URDF_FILENAME = "aliengo.urdf"
URDF_PATH = os.path.join(PROJECT_ROOT, URDF_FILENAME)
BODY_HEIGHT = 0.42
THIGH_ANGLE = 0.8
CALF_ANGLE = -1.6
NUM_TRAJ_STEPS = 20
STEP_HEIGHT = 0.05

# Frame names
FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
LEG_NAMES = ["FL", "FR", "RL", "RR"]

# Leg topology
LEG_IS_RIGHT = {0: False, 1: True, 2: False, 3: True}
MIRROR_LEG = {0: 1, 1: 0, 2: 3, 3: 2}
CHOSEN_LEG = 0  # FL, used for step model training

# Per-leg model order indices: leg_idx -> [hip, thigh, calf] in 12-joint model order
LEG_MODEL_INDICES = {
    0: [0, 1, 2],    # FL
    1: [3, 4, 5],    # FR
    2: [6, 7, 8],    # RL
    3: [9, 10, 11],  # RR
}

# Joint limits from URDF
HIP_LOWER = -1.2217
HIP_UPPER = 1.2217
CALF_LOWER = -2.7751
CALF_UPPER = -0.6458

# Data generation ranges (metres / radians)
X_RANGE = 0.15
Y_RANGE = 0.05
Z_RANGE = 0.1
ROLL_RANGE = 0.25
PITCH_RANGE = 0.25
YAW_RANGE = 0.3
IDENTITY_FRACTION = 0.005

# Joint names for display
JOINT_NAMES = ["hip", "thigh", "calf"]

# Isaac Sim joint orderings
MODEL_JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

ARTICULATION_JOINT_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]

# Default model-to-articulation mapping
MODEL_TO_ARTICULATION = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
ARTICULATION_TO_MODEL = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
