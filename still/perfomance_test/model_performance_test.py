import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from still.diffusion_train import load_model
from ink_kin_stance.diffusion.dataset import load_and_split_data, load_and_split_data_history

"""
script to produce the inference_tradeoff_metrics.csv file which represent the table in (results - training setup)
"""


def run_benchmark(model, diffusion, test_loader, device="cuda",
                  num_samples=1000, dataset=None, has_retroaction=False):
    model.eval()

    # Configurazioni da testare: (Tipo, Numero di Steps)
    configs = [
        ("ddpm", 500),
        ("ddim", 200),
        ("ddim", 100),
        ("ddim", 50),
        ("ddim", 25),
        ("ddim", 10),
        ("ddim", 5),
        ("ddim", 2),
        ("ddim", 1)
    ]

    results = []

    for sampler_type, steps in configs:
        print(f"\n--- Inizio test: {sampler_type.upper()} con {steps} steps ---")

        all_gt = []
        all_pred = []
        inference_times = []
        collected_samples = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            for batch in tqdm(test_loader, desc=f"Valutazione {sampler_type.upper()}-{steps}"):
                if collected_samples >= num_samples:
                    break

                # Slicing e spostamento su GPU
                delta = batch["delta"].to(device, non_blocking=True)
                current_joints = batch["current_joints"].to(device, non_blocking=True)
                gt_trajectory = batch["trajectory"].to(device, non_blocking=True)

                # Creazione condizione
                condition = torch.cat([delta, current_joints], dim=-1)
                if has_retroaction:
                    prev_joints = batch["prev_joints"].to(device, non_blocking=True)
                    prev_actions = batch["prev_actions"].to(device, non_blocking=True)
                    condition = torch.cat([delta, current_joints, prev_joints, prev_actions], dim=-1)

                shape = gt_trajectory.shape

                # Generazione con Profiling Temporale Preciso
                start_event.record()
                if sampler_type == "ddpm":
                    pred_trajectory = diffusion.sample(model=model, shape=shape, condition=condition)
                else:
                    pred_trajectory = diffusion.ddim_sample(model=model, shape=shape, condition=condition, ddim_steps=steps)
                end_event.record()

                torch.cuda.synchronize()
                inference_times.append(start_event.elapsed_time(end_event))

                # De-normalizzazione per calcolo metriche reali (in radianti)
                if dataset is not None and hasattr(dataset, 'traj_mean') and hasattr(dataset, 'traj_std'):
                    t_mean = torch.tensor(dataset.traj_mean, device=device)
                    t_std = torch.tensor(dataset.traj_std, device=device)
                    gt_trajectory = (gt_trajectory * t_std) + t_mean
                    pred_trajectory = (pred_trajectory * t_std) + t_mean

                all_gt.append(gt_trajectory.cpu())
                all_pred.append(pred_trajectory.cpu())

                collected_samples += 1

        # Concatenazione dei risultati per la configurazione attuale
        gt_final = torch.cat(all_gt, dim=0)[:num_samples]
        pred_final = torch.cat(all_pred, dim=0)[:num_samples]

        # 1. Calcolo Tempi
        mean_time_ms = np.mean(inference_times[:num_samples])
        std_time_ms = np.std(inference_times[:num_samples])
        update_rate_hz = 1000.0 / mean_time_ms if mean_time_ms > 0 else 0

        # 2. Calcolo MSE Globale
        mse_error = torch.nn.functional.mse_loss(pred_final, gt_final).item()

        # 3. Calcolo Worst-Case Peak Error (Errore massimo all'interno della traiettoria)
        err_abs = torch.abs(gt_final - pred_final)
        worst_cases, _ = torch.max(err_abs.view(gt_final.shape[0], -1), dim=1)
        mean_worst_peak = torch.mean(worst_cases).item()

        # 4. Calcolo Final Position Error (Errore all'ultimo step temporale)
        err_final = torch.abs(gt_final[:, -1, :] - pred_final[:, -1, :])
        worst_final, _ = torch.max(err_final, dim=1)
        mean_final_err = torch.mean(worst_final).item()

        # Salvataggio nel dizionario dei risultati
        results.append({
            "Sampler": sampler_type.upper(),
            "Steps": steps,
            "Inference Time (ms)": round(mean_time_ms, 2),
            "Time Std (ms)": round(std_time_ms, 2),
            "Update Rate (Hz)": round(update_rate_hz, 1),
            "MSE (rad^2)": round(mse_error, 6),
            "Mean Peak Error (rad)": round(mean_worst_peak, 4),
            "Mean Final Error (rad)": round(mean_final_err, 4)
        })

        print(f"Risultati salvati per {sampler_type.upper()}-{steps}: {mean_time_ms:.2f} ms | MSE: {mse_error:.6f}")

        # Pulisce la cache CUDA per evitare sbalzi termici/memoria tra un test e l'altro
        torch.cuda.empty_cache()

    return pd.DataFrame(results)


if __name__ == "__main__":
    model_path = "diff_model_vel_retroaction_BEST.pt"
    device = "cuda"

    print(f"Loading model from {model_path}...")
    diff_model, diffusion, checkpoint = load_model(model_path, device=device)

    # Verifica presenza retroazione nel modello
    has_retroaction = checkpoint.get("retroaction", False)

    # Caricamento del dataset in base al tipo di modello
    if has_retroaction:
        _, _, test_dataset, _, _ = load_and_split_data_history("trajectory_log_vel.zarr", train_ratio=0.8, val_ratio=0.1)
    else:
        _, _, test_dataset, _, _ = load_and_split_data("trajectory_log.zarr", train_ratio=0.8, val_ratio=0.1)

    # IMPORTANTE: batch_size=1 per misurare accuratamente la latenza per singola query del robot
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("\nInizio della campagna di Benchmark. Batch Size forzato a 1 per misurare la Single-Inference Latency.")

    # Esecuzione del Benchmark su 500 campioni
    df_results = run_benchmark(
        model=diff_model,
        diffusion=diffusion,
        test_loader=test_loader,
        device=device,
        dataset=test_dataset,
        has_retroaction=has_retroaction,
        num_samples=1000  # Modifica questo valore se vuoi testare più campioni
    )

    # Salvataggio in CSV e stampa a schermo
    csv_filename = "perfomance_test/inference_tradeoff_metrics.csv"
    df_results.to_csv(csv_filename, index=False)

    print("\n" + "=" * 80)
    print(" BENCHMARK COMPLETATO - RISULTATI SALVATI IN:", csv_filename)
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)