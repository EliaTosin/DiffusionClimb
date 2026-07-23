import torch
from torch.utils.data import DataLoader
from still.diffusion_train import load_model
from ink_kin_stance.diffusion.dataset import load_and_split_data, load_and_split_data_history
import numpy as np


def evaluate_two_models(model_A, diffusion_A, has_retroaction_A,
                        model_B, diffusion_B, has_retroaction_B,
                        test_loader, device="cuda", num_samples=10, dataset=None, ddim_steps=50):
    """
    Valuta contemporaneamente due modelli sul dataset di test per un confronto diretto.

    Returns:
        gt_trajectories, pred_A_ddpm, pred_A_ddim, pred_B_ddpm, pred_B_ddim, deltas, curr_joints
    """
    model_A.eval()
    model_B.eval()

    all_gt = []
    all_pred_A_ddpm = []
    all_pred_A_ddim = []
    all_pred_B_ddpm = []
    all_pred_B_ddim = []
    all_deltas = []
    all_curr_j = []

    collected_samples = 0
    print(f"Inizio generazione comparativa per {num_samples} traiettorie (DDIM-{ddim_steps})...")

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch in test_loader:
            if collected_samples >= num_samples:
                break

            batch_size = batch["trajectory"].shape[0]
            take_n = min(batch_size, num_samples - collected_samples)

            # Estrazione input comuni
            delta = batch["delta"][:take_n].to(device, non_blocking=True)
            current_joints = batch["current_joints"][:take_n].to(device, non_blocking=True)
            gt_trajectory = batch["trajectory"][:take_n].to(device, non_blocking=True)

            # --- Condizionamento Modello A ---
            condition_A = torch.cat([delta, current_joints], dim=-1)
            if has_retroaction_A:
                prev_joints = batch["prev_joints"][:take_n].to(device, non_blocking=True)
                prev_actions = batch["prev_actions"][:take_n].to(device, non_blocking=True)
                condition_A = torch.cat([delta, current_joints, prev_joints, prev_actions], dim=-1)

            # --- Condizionamento Modello B ---
            condition_B = torch.cat([delta, current_joints], dim=-1)
            if has_retroaction_B:
                prev_joints = batch["prev_joints"][:take_n].to(device, non_blocking=True)
                prev_actions = batch["prev_actions"][:take_n].to(device, non_blocking=True)
                condition_B = torch.cat([delta, current_joints, prev_joints, prev_actions], dim=-1)

            shape = gt_trajectory.shape

            # --- Inferenza Modello A ---
            pred_A_ddpm = diffusion_A.sample(model=model_A, shape=shape, condition=condition_A)
            pred_A_ddim = diffusion_A.ddim_sample(model=model_A, shape=shape, condition=condition_A, ddim_steps=ddim_steps)

            # --- Inferenza Modello B ---
            pred_B_ddpm = diffusion_B.sample(model=model_B, shape=shape, condition=condition_B)
            pred_B_ddim = diffusion_B.ddim_sample(model=model_B, shape=shape, condition=condition_B, ddim_steps=ddim_steps)

            # --- De-normalizzazione ---
            if dataset is not None and hasattr(dataset, 'traj_mean') and hasattr(dataset, 'traj_std'):
                t_mean = torch.tensor(dataset.traj_mean, device=device)
                t_std = torch.tensor(dataset.traj_std, device=device)
                d_mean = torch.tensor(dataset.delta_mean, device=device)
                d_std = torch.tensor(dataset.delta_std, device=device)

                gt_trajectory = (gt_trajectory * t_std) + t_mean
                pred_A_ddpm = (pred_A_ddpm * t_std) + t_mean
                pred_A_ddim = (pred_A_ddim * t_std) + t_mean
                pred_B_ddpm = (pred_B_ddpm * t_std) + t_mean
                pred_B_ddim = (pred_B_ddim * t_std) + t_mean
                delta = (delta * d_std) + d_mean
                current_joints = (current_joints * t_std) + t_mean

            all_gt.append(gt_trajectory.cpu())
            all_pred_A_ddpm.append(pred_A_ddpm.cpu())
            all_pred_A_ddim.append(pred_A_ddim.cpu())
            all_pred_B_ddpm.append(pred_B_ddpm.cpu())
            all_pred_B_ddim.append(pred_B_ddim.cpu())
            all_deltas.append(delta.cpu())
            all_curr_j.append(current_joints.cpu())

            collected_samples += take_n
            print(f"Generato {collected_samples}/{num_samples} campioni di test.")

    return (
        torch.cat(all_gt, dim=0),
        torch.cat(all_pred_A_ddpm, dim=0),
        torch.cat(all_pred_A_ddim, dim=0),
        torch.cat(all_pred_B_ddpm, dim=0),
        torch.cat(all_pred_B_ddim, dim=0),
        torch.cat(all_deltas, dim=0),
        torch.cat(all_curr_j, dim=0)
    )


if __name__ == "__main__":
    from prediction_distribution import set_seed
    set_seed(42)

    device = "cuda"
    ddim_steps_eval = 5  # Cambia a piacimento (es. 5 per testare la configurazione d'uso)
    num_samples_to_compare = 10  # Numero di traiettorie da plottare

    # 1. Caricamento Modello A (es. Retroazione - Vincitore)
    model_path_A = "models/diff_model_vel_retroaction.pt"
    print(f"Loading Model A from {model_path_A}...")
    model_A, diffusion_A, checkpoint_A = load_model(model_path_A, device=device)
    # has_retroaction_A = checkpoint_A.get("retroaction", False)

    # 2. Caricamento Modello B (es. Wloss - Secondo classificato)
    model_path_B = "models/diff_model_vel_retroaction_Wloss.pt"
    print(f"Loading Model B from {model_path_B}...")
    model_B, diffusion_B, checkpoint_B = load_model(model_path_B, device=device)
    # has_retroaction_B = checkpoint_B.get("retroaction", False)

    # 3. Scelta e caricamento del Dataset (usa la retroazione se almeno uno dei due la richiede)
    use_history_dataset = True
    if use_history_dataset:
        print("Using historical dataset loader...")
        _, _, test_dataset, _, _ = load_and_split_data_history("trajectory_log_vel.zarr", train_ratio=0.8, val_ratio=0.1)
    else:
        print("Using standard dataset loader...")
        _, _, test_dataset, _, _ = load_and_split_data("trajectory_log.zarr", train_ratio=0.8, val_ratio=0.1)

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 4. Esecuzione del test comparativo
    gt, pred_A_ddpm, pred_A_ddim, pred_B_ddpm, pred_B_ddim, deltas, curr_j = evaluate_two_models(
        model_A=model_A, diffusion_A=diffusion_A, has_retroaction_A=True,
        model_B=model_B, diffusion_B=diffusion_B, has_retroaction_B=True,
        test_loader=test_loader, device=device, num_samples=num_samples_to_compare, dataset=test_dataset,
        ddim_steps=ddim_steps_eval
    )

    # 5. Visualizzazione interattiva dei confronti
    # Usiamo plot_trajectories modificando gli ingressi per confrontare Model A DDIM vs Model B DDIM
    from still.eval_diffusion import plot_trajectories

    print("\nVisualizzazione dei grafici comparativi...")
    for i in range(gt.shape[0]):
        print(f"\n--- Plotting traiettoria {i + 1}/{num_samples_to_compare} ---")
        plot_trajectories(
            ik_trajectory=gt[i, :],  # Ground Truth (Solido)
            diff_trajectory=pred_A_ddim[i, :],  # Modello A (Tratteggiato / Es. Retroaction)
            delta=deltas[i, :],  # Target
            helper_trajectory=pred_B_ddim[i, :],  # Modello B (Puntinato / Es. Wloss)
        )