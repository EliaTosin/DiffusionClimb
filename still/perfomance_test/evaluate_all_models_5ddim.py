import os
import glob
import csv
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from still.diffusion_train import load_model
from ink_kin_stance.diffusion.dataset import load_and_split_data, load_and_split_data_history

"""
Evaluate all models by performing 1000 samples at 5 DDIM steps, producing a stat file and then ranking 
them by giving a linear weight to each column.
"""


def evaluate_trajectories(model, diffusion, test_loader, device="cuda",
                          num_samples=1000, dataset=None, has_retroaction=False, ddim_steps=5):
    model.eval()
    all_gt = []
    all_pred_ddim = []
    ddim_times = []

    collected_samples = 0
    print(f"Inizio generazione per {num_samples} traiettorie (DDIM-{ddim_steps})...")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

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

            # Campionamento DDIM + Profiling Temporale
            start_event.record()
            pred_trajectory_ddim = diffusion.ddim_sample(model=model, shape=shape, condition=condition, ddim_steps=ddim_steps)
            end_event.record()
            torch.cuda.synchronize()
            ddim_times.append(start_event.elapsed_time(end_event))

            # De-normalizzazione
            if dataset is not None and hasattr(dataset, 'traj_mean') and hasattr(dataset, 'traj_std'):
                t_mean = torch.tensor(dataset.traj_mean, device=device)
                t_std = torch.tensor(dataset.traj_std, device=device)
                gt_trajectory = (gt_trajectory * t_std) + t_mean
                pred_trajectory_ddim = (pred_trajectory_ddim * t_std) + t_mean

            all_gt.append(gt_trajectory.cpu())
            all_pred_ddim.append(pred_trajectory_ddim.cpu())

            collected_samples += take_n
            print(f"Generato {collected_samples}/{num_samples} campioni.")

    # --- CALCOLO METRICHE ---
    mean_ddim_ms = np.mean(ddim_times)

    gt_final = torch.cat(all_gt, dim=0)
    ddim_final = torch.cat(all_pred_ddim, dim=0)

    # 1. MSE Generale
    mse_ddim = torch.nn.functional.mse_loss(ddim_final, gt_final).item()

    # 2. Errore Assoluto Medio (Tutti i punti)
    err_abs_ddim = torch.abs(gt_final - ddim_final)
    mean_abs_err_rad = torch.mean(err_abs_ddim).item()
    mean_abs_err_deg = mean_abs_err_rad * (180.0 / np.pi)

    # 3. Peak Error (Peggior errore per traiettoria)
    worst_cases_ddim, _ = torch.max(err_abs_ddim.view(gt_final.shape[0], -1), dim=1)
    mean_peak_err_rad = torch.mean(worst_cases_ddim).item()
    mean_peak_err_deg = mean_peak_err_rad * (180.0 / np.pi)

    # 4. Final Position Error (Errore sull'ultimo step temporale)
    final_gt = gt_final[:, -1, :]
    final_ddim = ddim_final[:, -1, :]
    err_final_ddim = torch.abs(final_gt - final_ddim)

    mean_final_err_rad = torch.mean(err_final_ddim).item()
    mean_final_err_deg = mean_final_err_rad * (180.0 / np.pi)

    worst_final_ddim, _ = torch.max(err_final_ddim, dim=1)
    max_final_err_rad = torch.max(worst_final_ddim).item()
    max_final_err_deg = max_final_err_rad * (180.0 / np.pi)

    metrics = {
        "Inference Time (ms)": round(mean_ddim_ms, 2),
        "MSE (rad^2)": mse_ddim,
        "Mean Error (rad)": mean_abs_err_rad,
        "Mean Error (deg)": mean_abs_err_deg,
        "Mean Peak Error (rad)": mean_peak_err_rad,
        "Mean Peak Error (deg)": mean_peak_err_deg,
        "Mean Final Error (rad)": mean_final_err_rad,
        "Mean Final Error (deg)": mean_final_err_deg,
        "Max Final Error (rad)": max_final_err_rad,
        "Max Final Error (deg)": max_final_err_deg
    }

    return metrics


if __name__ == "__main__":
    device = "cuda"
    num_samples_to_test = 1000
    ddim_steps_to_use = 5

    # Cartelle in cui cercare i modelli
    folders_to_search = ["models"]
    model_paths = []

    for folder in folders_to_search:
        if os.path.exists(folder):
            model_paths.extend(glob.glob(f"{folder}/*.pt"))

    if not model_paths:
        print("Nessun modello trovato nelle cartelle specificate.")
        exit()

    print(f"Trovati {len(model_paths)} modelli. Inizio valutazione...")

    csv_filename = "perfomance_test/still_models_evaluation.csv"
    csv_headers = [
        "Model File", "Folder", "Inference Time (ms)", "MSE (rad^2)",
        "Mean Error (rad)", "Mean Error (deg)",
        "Mean Peak Error (rad)", "Mean Peak Error (deg)",
        "Mean Final Error (rad)", "Mean Final Error (deg)",
        "Max Final Error (rad)", "Max Final Error (deg)"
    ]

    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        writer.writeheader()

        for path in model_paths:
            folder_name = os.path.dirname(path)
            file_name = os.path.basename(path)
            print("\n" + "=" * 80)
            print(f"Valutazione modello: {file_name} (da {folder_name})")
            print("=" * 80)

            try:
                diff_model, diffusion, checkpoint = load_model(path, device=device)
                if diff_model.condition_mlp[0].in_features == 42:
                    has_retroaction = True
                elif diff_model.condition_mlp[0].in_features == 18:
                    has_retroaction = False
                else:
                    print("Detected model with condition shape", diff_model.condition_mlp[0].in_features)

                # Caricamento Dataset basato sul flag del modello
                if has_retroaction:
                    _, _, test_dataset, _, _ = load_and_split_data_history("trajectory_log_vel.zarr", train_ratio=0.8, val_ratio=0.1)
                else:
                    _, _, test_dataset, _, _ = load_and_split_data("trajectory_log.zarr", train_ratio=0.8, val_ratio=0.1)

                test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

                # Calcolo metriche
                metrics = evaluate_trajectories(
                    model=diff_model,
                    diffusion=diffusion,
                    test_loader=test_loader,
                    device=device,
                    num_samples=num_samples_to_test,
                    dataset=test_dataset,
                    has_retroaction=has_retroaction,
                    ddim_steps=ddim_steps_to_use
                )

                # Scrittura riga nel CSV
                row_data = {"Model File": file_name, "Folder": folder_name}
                row_data.update(metrics)
                writer.writerow(row_data)
                csv_file.flush()  # Salva immediatamente su disco

                print(f"Metriche salvate con successo per {file_name}.")

            except Exception as e:
                print(f"Errore durante la valutazione del modello {file_name}: {e}")

    print(f"\nValutazione completata! Tutti i risultati sono stati salvati in '{csv_filename}'.")

    if not os.path.exists(csv_filename):
        print(f"Errore: il file '{csv_filename}' non esiste.")
        exit()

    df = pd.read_csv(csv_filename)

    # Se ci sono più di 10 modelli, prendiamo i top 10 per MSE generico come base di partenza.
    if len(df) > 10:
        print(f"Trovati {len(df)} modelli. Seleziono i migliori 10 basati sull'MSE per il ranking a 10 posizioni.\n")
        df = df.sort_values(by="MSE (rad^2)").head(10).reset_index(drop=True)
    elif len(df) < 10:
        print(f"Attenzione: ci sono solo {len(df)} modelli nel CSV. Il punteggio massimo sarà pari a {len(df)} punti.\n")

    n_models = len(df)

    # LISTA CORRETTA: Rimosse le metriche duplicate in radianti per evitare il doppio conteggio
    ranking_columns = [
        "Inference Time (ms)",
        "MSE (rad^2)",
        "Mean Error (deg)",
        "Mean Peak Error (deg)",
        "Mean Final Error (deg)",
        "Max Final Error (deg)"
    ]

    # Inizializziamo un DataFrame per i punteggi
    df_points = pd.DataFrame()
    df_points["Model File"] = df["Model File"]
    df_points["Folder"] = df["Folder"]

    # Calcolo dei punti per ciascuna categoria (più basso = migliore -> più punti)
    for col in ranking_columns:
        ranks = df[col].rank(ascending=True, method="min")
        points = n_models - ranks + 1
        df_points[f"{col} Points"] = points

    # Calcolo del punteggio totale sommando le colonne di punti
    points_cols = [c for c in df_points.columns if "Points" in c]
    df_points["Total Points"] = df_points[points_cols].sum(axis=1)

    # Ordiniamo la classifica finale
    df_ranking_final = df_points.sort_values(by="Total Points", ascending=False).reset_index(drop=True)

    # Visualizzazione dei risultati
    print("=" * 100)
    print(" CLASSIFICA FINALE DEI MODELLI (Ranking Corretto - Senza Duplicati)")
    print("=" * 100)
    print(df_ranking_final[["Model File", "Folder", "Total Points"] + points_cols].to_string(index=True))
    print("=" * 100)

    vincitore = df_ranking_final.iloc[0]
    print(f"\n🏆 IL MODELLO VINCITORE È: {vincitore['Model File']} (da {vincitore['Folder']}) "
          f"con {vincitore['Total Points']} punti totali!")