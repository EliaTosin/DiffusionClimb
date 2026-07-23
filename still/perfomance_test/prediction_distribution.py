import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from still.diffusion_train import load_model
from ink_kin_stance.diffusion.dataset import load_and_split_data, load_and_split_data_history


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_error_distributions(model, diffusion, test_loader, device="cuda",
                                num_samples=1000, dataset=None, has_retroaction=False, ddim_steps=5):
    model.eval()
    all_gt = []
    all_pred_ddim = []

    collected_samples = 0
    print(f"Estrazione distribuzioni su {num_samples} campioni (DDIM-{ddim_steps})...")

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch in test_loader:
            if collected_samples >= num_samples:
                break

            batch_size = batch["trajectory"].shape[0]
            take_n = min(batch_size, num_samples - collected_samples)

            delta = batch["delta"][:take_n].to(device, non_blocking=True)
            current_joints = batch["current_joints"][:take_n].to(device, non_blocking=True)
            gt_trajectory = batch["trajectory"][:take_n].to(device, non_blocking=True)

            condition = torch.cat([delta, current_joints], dim=-1)
            if has_retroaction:
                prev_joints = batch["prev_joints"][:take_n].to(device, non_blocking=True)
                prev_actions = batch["prev_actions"][:take_n].to(device, non_blocking=True)
                condition = torch.cat([delta, current_joints, prev_joints, prev_actions], dim=-1)

            shape = gt_trajectory.shape
            pred_trajectory_ddim = diffusion.ddim_sample(model=model, shape=shape, condition=condition, ddim_steps=ddim_steps)

            if dataset is not None and hasattr(dataset, 'traj_mean') and hasattr(dataset, 'traj_std'):
                t_mean = torch.tensor(dataset.traj_mean, device=device)
                t_std = torch.tensor(dataset.traj_std, device=device)
                gt_trajectory = (gt_trajectory * t_std) + t_mean
                pred_trajectory_ddim = (pred_trajectory_ddim * t_std) + t_mean

            all_gt.append(gt_trajectory.cpu())
            all_pred_ddim.append(pred_trajectory_ddim.cpu())
            collected_samples += take_n

    gt_final = torch.cat(all_gt, dim=0)  # [N, 20, 12]
    ddim_final = torch.cat(all_pred_ddim, dim=0)  # [N, 20, 12]

    err_abs = torch.abs(gt_final - ddim_final)

    # 1. Peak Error per ogni traiettoria -> [N]
    peak_errors_rad, _ = torch.max(err_abs.view(gt_final.shape[0], -1), dim=1)
    peak_errors_deg = (peak_errors_rad * (180.0 / np.pi)).numpy()

    # 2. Worst Final Joint Error per ogni traiettoria -> [N]
    final_gt = gt_final[:, -1, :]
    final_ddim = ddim_final[:, -1, :]
    err_final = torch.abs(final_gt - final_ddim)
    final_worst_rad, _ = torch.max(err_final, dim=1)
    final_worst_deg = (final_worst_rad * (180.0 / np.pi)).numpy()

    return peak_errors_deg, final_worst_deg


def plot_error_distributions(data_dict, output_pdf="still_error_distribution.pdf"):
    # Impostazioni stilistiche con FONT IN GRASSETTO e marcati
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'grid.linestyle': ':',
        'grid.alpha': 0.65
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    colors = {
        "diff_model_vel_retroaction.pt": "#2980b9",
        "diff_model_vel_retroaction_Wloss.pt": "#e74c3c"
    }

    labels_map = {
        "diff_model_vel_retroaction.pt": "Diff Retroaction",
        "diff_model_vel_retroaction_Wloss.pt": "Diff Retroaction Wloss"
    }

    # Definizione esatta di 36 Bins tra 0 e 18 gradi (passo = 0.5°)
    max_deg = 18.0
    num_bins = 36
    bins = np.linspace(0, max_deg, num_bins + 1)
    x_ticks = np.arange(0, max_deg + 1, 1.0)

    # --- PLOT 1: Peak Trajectory Error Distribution ---
    for model_name, res in data_dict.items():
        color = colors.get(model_name, "#34495e")
        label = labels_map.get(model_name, model_name)
        data = res["peak"]
        mean_val = np.mean(data)

        # Istogramma
        axes[0].hist(data, bins=bins, alpha=0.45, color=color, label=label, density=True, edgecolor='none')

        # Linea della Media (senza $ \mathbf{} $ per uniformare il font)
        axes[0].axvline(mean_val, color=color, linestyle='--', linewidth=2.2,
                        label=f"{label} Mean: {mean_val:.2f}°")

    axes[0].set_title('Mean Peak Error (deg) Distribution', fontweight='bold', pad=10)
    axes[0].set_xlabel('Mean Peak Error (deg)', fontweight='bold')
    axes[0].set_ylabel('Probability Density', fontweight='bold')
    axes[0].set_xticks(x_ticks)
    axes[0].set_xlim(0, max_deg)
    axes[0].grid(True, which='both')

    # Legenda: applica il grassetto uniforme a TUTTI gli elementi
    axes[0].legend(frameon=True, facecolor='#fcfcfc', edgecolor='#ccc',
                   fontsize=9.5, prop={'weight': 'bold', 'size': 9.5})

    # --- PLOT 2: Worst Final Joint Error Distribution ---
    for model_name, res in data_dict.items():
        color = colors.get(model_name, "#34495e")
        label = labels_map.get(model_name, model_name)
        data = res["final"]
        mean_val = np.mean(data)

        # Istogramma
        axes[1].hist(data, bins=bins, alpha=0.45, color=color, label=label, density=True, edgecolor='none')

        # Linea della Media (senza $ \mathbf{} $ per uniformare il font)
        axes[1].axvline(mean_val, color=color, linestyle='--', linewidth=2.2,
                        label=f"{label} Mean: {mean_val:.2f}°")

    axes[1].set_title('Worst Final Joint Error (deg) Distribution', fontweight='bold', pad=10)
    axes[1].set_xlabel('Worst Final Error per Trajectory (deg)', fontweight='bold')
    axes[1].set_ylabel('Probability Density', fontweight='bold')
    axes[1].set_xticks(x_ticks)
    axes[1].set_xlim(0, max_deg)
    axes[1].grid(True, which='both')

    # Legenda: applica il grassetto uniforme a TUTTI gli elementi
    axes[1].legend(frameon=True, facecolor='#fcfcfc', edgecolor='#ccc',
                   fontsize=9.5, prop={'weight': 'bold', 'size': 9.5})

    # Rende in GRASSETTO i tick numerici degli assi X e Y per tutti e due i subplots
    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')
        for tick in ax.get_yticklabels():
            tick.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    print(f"Grafico delle distribuzioni salvato in: {output_pdf}")
    plt.show()


if __name__ == "__main__":
    set_seed(42)
    device = "cuda"
    num_samples_to_test = 200000
    ddim_steps_to_use = 5

    models_to_compare = [
        "models/diff_model_vel_retroaction_Wloss.pt",
        "models/diff_model_vel_retroaction.pt"
    ]

    results = {}

    for path in models_to_compare:
        if not os.path.exists(path):
            print(f"Modello {path} non trovato, salto...")
            continue

        file_name = os.path.basename(path)
        print(f"Elaborazione: {file_name}")

        diff_model, diffusion, checkpoint = load_model(path, device=device)
        has_retroaction = (diff_model.condition_mlp[0].in_features == 42)

        if has_retroaction:
            _, _, test_dataset, _, _ = load_and_split_data_history("trajectory_log_vel.zarr", train_ratio=0.8, val_ratio=0.1)
        else:
            _, _, test_dataset, _, _ = load_and_split_data("trajectory_log.zarr", train_ratio=0.8, val_ratio=0.1)

        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        peak_deg, final_deg = extract_error_distributions(
            model=diff_model,
            diffusion=diffusion,
            test_loader=test_loader,
            device=device,
            num_samples=num_samples_to_test,
            dataset=test_dataset,
            has_retroaction=has_retroaction,
            ddim_steps=ddim_steps_to_use
        )

        results[file_name] = {
            "peak": peak_deg,
            "final": final_deg
        }

    if results:
        plot_error_distributions(results, output_pdf="still_error_distribution.pdf")