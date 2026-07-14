import torch
from torch.utils.data import DataLoader
from still.diffusion_train import load_model
from ink_kin_stance.diffusion.dataset import load_and_split_data, load_and_split_data_history

#
# def evaluate_trajectories(model, diffusion, test_loader, device="cuda",
#                           num_samples=10, dataset=None, has_retroaction=False):
#     """
#     Valuta il modello sul dataset di test e ritorna traiettorie e condizioni.
#
#     Args:
#         model: Il modello di diffusione allenato.
#         diffusion: L'oggetto GaussianDiffusion.
#         test_loader: DataLoader del test set.
#         device: Dispositivo su cui eseguire i calcoli.
#         num_samples: Numero esatto di traiettorie da generare (es. 10).
#         dataset: (Opzionale) Il dataset, usato per de-normalizzare.
#
#     Returns:
#         gt_trajectories, pred_trajectories, deltas, current_joints, prev_joints, prev_actions
#     """
#     model.eval()
#
#     # Liste per accumulare i risultati
#     all_gt = []
#     all_pred = []
#     all_deltas = []
#     all_curr_j = []
#     all_prev_j = []
#     all_prev_a = []
#
#     collected_samples = 0
#     print(f"Inizio generazione per {num_samples} traiettorie...")
#
#     with torch.no_grad(), torch.amp.autocast("cuda"):
#         for batch in test_loader:
#             if collected_samples >= num_samples:
#                 break
#
#             # Calcoliamo quanti elementi prendere da questo batch
#             batch_size = batch["trajectory"].shape[0]
#             take_n = min(batch_size, num_samples - collected_samples)
#
#             # 1. Estrazione e Slicing immediato (risparmia calcolo durante la diffusione)
#             delta = batch["delta"][:take_n].to(device, non_blocking=True)
#             current_joints = batch["current_joints"][:take_n].to(device, non_blocking=True)
#             gt_trajectory = batch["trajectory"][:take_n].to(device, non_blocking=True)
#             # 2. Creazione della condizione
#             condition = torch.cat([delta, current_joints], dim=-1)
#             if has_retroaction:
#                 prev_joints = batch["prev_joints"][:take_n].to(device, non_blocking=True)
#                 prev_actions = batch["prev_actions"][:take_n].to(device, non_blocking=True)
#                 condition = torch.cat([delta, current_joints, prev_joints, prev_actions], dim=-1)
#
#             # 3. Campionamento dal modello
#             shape = gt_trajectory.shape
#             pred_trajectory = diffusion.sample(model=model, shape=shape, condition=condition)
#             pred_trajectory_ddim = diffusion.ddim_sample(model=model, shape=shape, condition=condition, ddim_steps=10)
#
#             # 4. De-normalizzazione (se il dataset è fornito)
#             if dataset is not None and hasattr(dataset, 'traj_mean') and hasattr(dataset, 'traj_std'):
#                 t_mean = torch.tensor(dataset.traj_mean, device=device)
#                 t_std = torch.tensor(dataset.traj_std, device=device)
#                 d_mean = torch.tensor(dataset.delta_mean, device=device)
#                 d_std = torch.tensor(dataset.delta_std, device=device)
#
#                 # De-normalizziamo le traiettorie e gli stati dei giunti (stessa scala)
#                 gt_trajectory = (gt_trajectory * t_std) + t_mean
#                 pred_trajectory = (pred_trajectory * t_std) + t_mean
#                 current_joints = (current_joints * t_std) + t_mean
#                 if has_retroaction:
#                     prev_joints = (prev_joints * t_std) + t_mean
#                     prev_actions = (prev_actions * t_std) + t_mean
#
#                 # De-normalizziamo il delta (scala diversa)
#                 delta = (delta * d_std) + d_mean
#
#             # Salvataggio su CPU
#             all_gt.append(gt_trajectory.cpu())
#             all_pred.append(pred_trajectory.cpu())
#             all_deltas.append(delta.cpu())
#             all_curr_j.append(current_joints.cpu())
#             if has_retroaction:
#                 all_prev_j.append(prev_joints.cpu())
#                 all_prev_a.append(prev_actions.cpu())
#
#             collected_samples += take_n
#             print(f"Generato {collected_samples}/{num_samples} campioni.")
#
#     # 5. Concatenazione finale per restituire singoli tensori puliti
#     if not has_retroaction:
#         all_prev_j = all_curr_j
#         all_prev_a = all_curr_j
#     return (
#         torch.cat(all_gt, dim=0),
#         torch.cat(all_pred, dim=0),
#         torch.cat(all_deltas, dim=0),
#         torch.cat(all_curr_j, dim=0),
#         torch.cat(all_prev_j, dim=0),
#         torch.cat(all_prev_a, dim=0)
#     )


import torch
import numpy as np


def evaluate_trajectories(model, diffusion, test_loader, device="cuda",
                          num_samples=10, dataset=None, has_retroaction=False, ddim_steps=50):
    """
    Valuta il modello sul dataset di test, calcola i tempi di inferenza di DDPM e DDIM,
    e ritorna traiettorie (sia DDPM che DDIM) e condizioni.

    Args:
        model: Il modello di diffusione allenato.
        diffusion: L'oggetto GaussianDiffusion.
        test_loader: DataLoader del test set.
        device: Dispositivo su cui eseguire i calcoli.
        num_samples: Numero esatto di traiettorie da generare (es. 10).
        dataset: (Opzionale) Il dataset, usato per de-normalizzare.
        has_retroaction: Se includere giunti e azioni precedenti nella condizione.
        ddim_steps: Numero di step per il campionatore DDIM.

    Returns:
        gt_trajectories, pred_ddpm, pred_ddim, deltas, current_joints, prev_joints, prev_actions
    """
    model.eval()

    # Liste per accumulare i risultati (sdoppiate per DDPM e DDIM)
    all_gt = []
    all_pred_ddpm = []
    all_pred_ddim = []
    all_deltas = []
    all_curr_j = []
    all_prev_j = []
    all_prev_a = []

    # Liste per tracciare i tempi di inferenza di ogni batch
    ddpm_times = []
    ddim_times = []

    collected_samples = 0
    print(f"Inizio generazione per {num_samples} traiettorie (DDPM vs DDIM-{ddim_steps})...")

    # Inizializzazione degli eventi CUDA per il profiling preciso del tempo
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch in test_loader:
            if collected_samples >= num_samples:
                break

            batch_size = batch["trajectory"].shape[0]
            take_n = min(batch_size, num_samples - collected_samples)

            # 1. Estrazione e Slicing immediato
            delta = batch["delta"][:take_n].to(device, non_blocking=True)
            current_joints = batch["current_joints"][:take_n].to(device, non_blocking=True)
            gt_trajectory = batch["trajectory"][:take_n].to(device, non_blocking=True)

            # 2. Creazione della condizione
            condition = torch.cat([delta, current_joints], dim=-1)
            if has_retroaction:
                prev_joints = batch["prev_joints"][:take_n].to(device, non_blocking=True)
                prev_actions = batch["prev_actions"][:take_n].to(device, non_blocking=True)
                condition = torch.cat([delta, current_joints, prev_joints, prev_actions], dim=-1)

            shape = gt_trajectory.shape

            # 3. Campionamento DDPM puro + Profiling Temporale
            start_event.record()
            pred_trajectory_ddpm = diffusion.sample(model=model, shape=shape, condition=condition)
            end_event.record()
            torch.cuda.synchronize()  # Attende che la GPU finisca per misurare il tempo reale
            ddpm_times.append(start_event.elapsed_time(end_event))  # Tempo in millisecondi

            # 4. Campionamento DDIM + Profiling Temporale
            start_event.record()
            pred_trajectory_ddim = diffusion.ddim_sample(model=model, shape=shape, condition=condition, ddim_steps=ddim_steps)
            end_event.record()
            torch.cuda.synchronize()
            ddim_times.append(start_event.elapsed_time(end_event))  # Tempo in millisecondi

            # 5. De-normalizzazione (se il dataset è fornito)
            if dataset is not None and hasattr(dataset, 'traj_mean') and hasattr(dataset, 'traj_std'):
                t_mean = torch.tensor(dataset.traj_mean, device=device)
                t_std = torch.tensor(dataset.traj_std, device=device)
                d_mean = torch.tensor(dataset.delta_mean, device=device)
                d_std = torch.tensor(dataset.delta_std, device=device)

                gt_trajectory = (gt_trajectory * t_std) + t_mean
                pred_trajectory_ddpm = (pred_trajectory_ddpm * t_std) + t_mean
                pred_trajectory_ddim = (pred_trajectory_ddim * t_std) + t_mean
                current_joints = (current_joints * t_std) + t_mean
                if has_retroaction:
                    prev_joints = (prev_joints * t_std) + t_mean
                    prev_actions = (prev_actions * t_std) + t_mean

                delta = (delta * d_std) + d_mean

            # Salvataggio su CPU
            all_gt.append(gt_trajectory.cpu())
            all_pred_ddpm.append(pred_trajectory_ddpm.cpu())
            all_pred_ddim.append(pred_trajectory_ddim.cpu())
            all_deltas.append(delta.cpu())
            all_curr_j.append(current_joints.cpu())
            if has_retroaction:
                all_prev_j.append(prev_joints.cpu())
                all_prev_a.append(prev_actions.cpu())

            collected_samples += take_n
            print(f"Generato {collected_samples}/{num_samples} campioni.")

    # Stampa dei risultati di benchmark temporale
    mean_ddpm_ms = np.mean(ddpm_times)
    mean_ddim_ms = np.mean(ddim_times)
    speedup = mean_ddpm_ms / mean_ddim_ms

    print("\n" + "=" * 50)
    print(" BENCHMARK INFERENCE TIME (Media per batch)")
    print("=" * 50)
    print(f"DDPM (Full): {mean_ddpm_ms:.2f} ms")
    print(f"DDIM ({ddim_steps} steps): {mean_ddim_ms:.2f} ms")
    print(f"Fattore di Accelerazione (Speedup): {speedup:.2f}x più veloce!")
    print("=" * 50 + "\n")

    # 6. Concatenazione finale
    if not has_retroaction:
        all_prev_j = all_curr_j
        all_prev_a = all_curr_j

    # Assumiamo che all_gt, all_pred_ddpm e all_pred_ddim siano liste di tensori CPU
    # prima di essere concatenati, o esegui questo calcolo sui tensori finali già concatenati:

    gt_final = torch.cat(all_gt, dim=0)  # Shape: [N_samples, 20, 12]
    ddpm_final = torch.cat(all_pred_ddpm, dim=0)  # Shape: [N_samples, 20, 12]
    ddim_final = torch.cat(all_pred_ddim, dim=0)  # Shape: [N_samples, 20, 12]

    # 1. Calcolo dell'errore assoluto punto per punto per ogni traiettoria
    # Risultato ha shape [N_samples, 20, 12]
    err_abs_ddpm = torch.abs(gt_final - ddpm_final)
    err_abs_ddim = torch.abs(gt_final - ddim_final)

    # 2. Trova l'errore massimo (Worst Case) ALL'INTERNO di ogni singola traiettoria
    # Appiattiamo le dimensioni [20, 12] -> [240] per trovare il picco assoluto di quel sample
    worst_cases_ddpm, _ = torch.max(err_abs_ddpm.view(gt_final.shape[0], -1), dim=1)  # Shape: [N_samples]
    worst_cases_ddim, _ = torch.max(err_abs_ddim.view(gt_final.shape[0], -1), dim=1)  # Shape: [N_samples]

    # 3. Calcola la media e la deviazione standard dei Worst Cases tra tutte le traiettorie
    mean_worst_ddpm = torch.mean(worst_cases_ddpm).item()
    std_worst_ddpm = torch.std(worst_cases_ddpm).item()

    mean_worst_ddim = torch.mean(worst_cases_ddim).item()
    std_worst_ddim = torch.std(worst_cases_ddim).item()

    # 4. Trova anche il "Worst-of-the-Worst" assoluto di tutto il dataset di test
    absolute_max_ddpm = torch.max(worst_cases_ddpm).item()
    absolute_max_ddim = torch.max(worst_cases_ddim).item()

    print("\n" + "=" * 50)
    print(" WORST-CASE TRAJECTORY ERROR ANALYSIS (Radianti)")
    print("=" * 50)
    print(f"DDPM - Errore di picco medio: {mean_worst_ddpm:.4f} ± {std_worst_ddpm:.4f} rad")
    print(f"DDPM - Massimo picco assoluto registrato: {absolute_max_ddpm:.4f} rad")
    print("-" * 50)
    print(f"DDIM - Errore di picco medio: {mean_worst_ddim:.4f} ± {std_worst_ddim:.4f} rad")
    print(f"DDIM - Massimo picco assoluto registrato: {absolute_max_ddim:.4f} rad")
    print("=" * 50 + "\n")

    # Estraiamo solo l'ultimo passo temporale di ogni traiettoria (passo 20, indice -1)
    # Shape risultante: [N_samples, 12] (errore sui 12 giunti alla fine del movimento)
    final_gt = gt_final[:, -1, :]
    final_ddpm = ddpm_final[:, -1, :]
    final_ddim = ddim_final[:, -1, :]

    # 1. Calcolo dell'errore assoluto finale per ogni giunto di ogni traiettoria
    err_final_ddpm = torch.abs(final_gt - final_ddpm)
    err_final_ddim = torch.abs(final_gt - final_ddim)

    # 2. Per ogni traiettoria, prendiamo il giunto che ha sgarbato di più alla fine (Worst Giunto)
    worst_final_ddpm, _ = torch.max(err_final_ddpm, dim=1)  # Shape: [N_samples]
    worst_final_ddim, _ = torch.max(err_final_ddim, dim=1)  # Shape: [N_samples]

    # 3. Media e Massimo dell'errore finale tra tutte le traiettorie
    mean_final_err_ddpm = torch.mean(err_final_ddpm).item()  # Media globale su tutti i giunti alla fine
    max_final_err_ddpm = torch.max(worst_final_ddpm).item()  # Il peggior errore finale registrato

    mean_final_err_ddim = torch.mean(err_final_ddim).item()
    max_final_err_ddim = torch.max(worst_final_ddim).item()

    print("\n" + "=" * 50)
    print(" FINAL POSITION ACCURACY ANALYSIS (Passo Temporale Finale)")
    print("=" * 50)
    print(f"DDPM - Errore finale medio (tutti i giunti): {mean_final_err_ddpm:.4f} rad")
    print(f"DDPM - Peggior errore di arrivo assoluto:   {max_final_err_ddpm:.4f} rad")
    print("-" * 50)
    print(f"DDIM - Errore finale medio (tutti i giunti): {mean_final_err_ddim:.4f} rad")
    print(f"DDIM - Peggior errore di arrivo assoluto:   {max_final_err_ddim:.4f} rad")
    print("=" * 50 + "\n")

    return (
        torch.cat(all_gt, dim=0),
        torch.cat(all_pred_ddpm, dim=0),
        torch.cat(all_pred_ddim, dim=0),
        torch.cat(all_deltas, dim=0),
        torch.cat(all_curr_j, dim=0),
        torch.cat(all_prev_j, dim=0),
        torch.cat(all_prev_a, dim=0)
    )

if __name__ == "__main__":
    model_path = "/home/etosin/Documents/diff_stance/step/diffusion_model_retroaction_BEST.pt"
    device = "cuda"

    print(f"Loading model from {model_path}...")
    diff_model, diffusion, checkpoint = load_model(model_path, device=device)

    keys = list(checkpoint.keys())
    keys.remove('model_state_dict')
    print("*"*100)
    for k in keys:
        print(k, ": ", checkpoint[k])
    print("*"*100)

    # 2. Ricrea il test_loader (se non lo ritorni dalla funzione train)
    has_retroaction = checkpoint.get("retroaction", False)
    if has_retroaction:
        _, _, test_dataset, _, _ = load_and_split_data_history("trajectory_log.zarr", train_ratio=0.8, val_ratio=0.1)

    else:
        _, _, test_dataset, _, _ = load_and_split_data("trajectory_log.zarr", train_ratio=0.8, val_ratio=0.1)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)  # Batch size più piccolo per il sampling

    # 3. Ottieni le traiettorie
    gt_trajectories, pred_trajectories_ddpm, pred_trajectories_ddim, deltas, curr_joints, prev_joints, prev_actions = evaluate_trajectories(
        model=diff_model,
        diffusion=diffusion,
        test_loader=test_loader,
        device="cuda",
        dataset=test_dataset,  # Passiamo il dataset per la denormalizzazione
        has_retroaction=has_retroaction,
    )
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    print(f"Shape Ground Truth: {gt_trajectories.shape}")
    print(f"Shape Predizioni DDPM: {pred_trajectories_ddpm.shape}")
    print(f"Shape Predizioni DDIM: {pred_trajectories_ddim.shape}")

    # 4. Calcolo di una metrica diretta (es. MSE sulle traiettorie finali/denormalizzate)
    mse_error = torch.nn.functional.mse_loss(pred_trajectories_ddpm, gt_trajectories)
    print(f"Mean Squared Error DDPM (sulle traiettorie reali): {mse_error.item()}")

    mse_error = torch.nn.functional.mse_loss(pred_trajectories_ddim, gt_trajectories)
    print(f"Mean Squared Error DDIM (sulle traiettorie reali): {mse_error.item()}")

    from eval_diffusion import plot_trajectories
    for i in range(gt_trajectories.shape[0]):
        plot_trajectories(
            ik_trajectory=gt_trajectories[i, :],
            diff_trajectory=pred_trajectories_ddim[i, :],
            delta_pos=deltas[i, :],
            # helper_trajectory=pred_trajectories_ddim[i, :],
        )
        if i > 10:
            break