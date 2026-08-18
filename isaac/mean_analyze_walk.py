import numpy as np

def analyze_log(log_path="walk_log.npz"):
    data = np.load(log_path)
    num_steps = int(data["num_steps"])
    print(f"Caricati dati per {num_steps} passi completi.\n")

    all_pd_err = []
    all_ik_err = []
    all_smoothness = []
    all_drift = []

    for i in range(num_steps):
        prefix = f"step{i}_"
        targets = data[prefix + "targets"]
        actuals = data[prefix + "actuals"]
        ik_joints = data[prefix + "ik_joints"]
        foot_pos = data[prefix + "foot_pos"]
        stepping_leg = int(data[prefix + "stepping_leg"])

        # 1. PD Error
        pd_err = np.abs(actuals - targets).mean()
        all_pd_err.append(pd_err)

        # 2. IK Error
        if not np.any(np.isnan(ik_joints)):
            ik_err = np.abs(targets - ik_joints).mean()
            all_ik_err.append(ik_err)

        # 3. Smoothness
        if len(targets) > 1:
            sm = np.abs(np.diff(targets, axis=0)).mean()
            all_smoothness.append(sm)

        # 4. Foot Drift (sulle 3 zampe in Stance)
        stance_legs = [l for l in range(4) if l != stepping_leg]
        for leg in stance_legs:
            d = np.linalg.norm(foot_pos[-1, leg] - foot_pos[0, leg]) * 1000  # mm
            all_drift.append(d)

    print("=" * 50)
    print("   RISULTATI GLOBALI PER LA TABELLA DELLA TESI")
    print("=" * 50)
    print(f"PD Tracking Error (MAE) : {np.mean(all_pd_err):.4f} ± {np.std(all_pd_err):.4f} rad")
    print(f"Diffusion vs IK Error    : {np.mean(all_ik_err):.4f} ± {np.std(all_ik_err):.4f} rad")
    print(f"Smoothness (Delta q)     : {np.mean(all_smoothness):.4f} rad/frame")
    print(f"Stance Foot Drift (Mean) : {np.mean(all_drift):.2f} ± {np.std(all_drift):.2f} mm")
    print("=" * 50)

if __name__ == "__main__":
    analyze_log()
    analyze_log("walk_log_seed42.npz")
