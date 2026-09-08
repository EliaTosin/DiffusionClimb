import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Carica i dati
df = pd.read_csv("perfomance_test/inference_tradeoff_metrics.csv")

# 2. Dati DDIM
df_ddim = df[df["Sampler"] == "DDIM"].sort_values(by="Steps", ascending=False)
steps = df_ddim["Steps"].to_numpy()
mse_ddim = df_ddim["MSE (rad^2)"].to_numpy()
time_ddim = df_ddim["Inference Time (ms)"].to_numpy()

# 3. Dati baseline DDPM
df_ddpm = df[df["Sampler"] == "DDPM"]
ddpm_mse_baseline = df_ddpm["MSE (rad^2)"].values[0]
ddpm_time_baseline = df_ddpm["Inference Time (ms)"].values[0]

# --- Configurazione tipografica accademica ---
plt.rcParams.update({
    "font.family": "serif",
    "font.weight": "bold",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

x_labels = [str(s) for s in steps]
x_indices = np.arange(len(steps))
color_mse = "#2C3E50"   # Blu scuro / ardesia
color_time = "#16A085"  # Verde petrolio

# Range bloccati per garantire allineamento perfetto tra slide
y1_lim = (0, mse_ddim[-2])
y2_lim = (0, max(time_ddim) * 1.15)


def build_tradeoff_plot(show_baseline=False, show_ddim_mse=False, filename=None):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, color="#E0E0E0")

    # Asse X e Asse Y1 (MSE)
    ax1.set_xlabel("DDIM Sampling Steps", fontweight="bold", labelpad=10)
    ax1.set_ylabel("Mean Squared Error (MSE) [$rad^2$]", color=color_mse, fontweight="bold")
    ax1.set_ylim(y1_lim)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels)
    ax1.tick_params(axis='y', labelcolor=color_mse)

    # Asse Y2 (Inference Latency)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Single-Inference Latency [ms]", color=color_time, fontweight="bold")
    ax2.set_ylim(y2_lim)
    ax2.tick_params(axis='y', labelcolor=color_time)

    # --- ELEMENTO 1: Sempre presente (Inference Time) ---
    line_time = ax2.plot(x_indices, time_ddim, marker='s', linestyle=":", linewidth=2,
                         color=color_time, label="DDIM Inference Time", zorder=3)

    # --- ELEMENTO 2: Baseline DDPM ---
    line_ddpm_base = None
    if show_baseline:
        line_ddpm_base = ax1.axhline(y=ddpm_mse_baseline, color="red", linestyle="--",
                                     linewidth=1.5, label="DDPM Baseline (500 steps)", zorder=2)
        ax1.text(5.5, ddpm_mse_baseline * 1.10, "DDPM Baseline \n (500 steps)",
                 color="red", fontweight="bold", fontsize=10, va="bottom", ha="left", zorder=4)

    # --- ELEMENTO 3: Performance DDIM (MSE) & Spot Ottimale ---
    line_mse = None
    if show_ddim_mse:
        line_mse = ax1.plot(x_indices, mse_ddim, marker='o', linewidth=2.5,
                            color=color_mse, label="DDIM Trajectory MSE", zorder=3)

        opt_idx = 5
        ax1.scatter(opt_idx, mse_ddim[opt_idx], color="orange", s=150, facecolors='none',
                    edgecolors='orange', linewidths=3, zorder=5)

    # Configurazione dinamica della legenda
    if show_ddim_mse and show_baseline:
        opt_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                                markeredgecolor='orange', markersize=10, markeredgewidth=2)
        final_lines = [line_mse[0], line_time[0], opt_marker]
        final_labels = [
            "DDIM Trajectory MSE (Left)",
            "DDIM Inference Time (Right)",
            "Optimal Selected Spot (5 Steps)"
        ]
        ax1.legend(final_lines, final_labels, loc="upper center",
                   fancybox=True, shadow=False, ncol=1, frameon=True,
                   facecolor="white", edgecolor="#D0D0D0")
    elif show_baseline:
        ax1.legend([line_time[0], line_ddpm_base],
                   ["DDIM Inference Time (Right)", "DDPM Baseline MSE (Left)"],
                   loc="upper center", fancybox=True, shadow=False,
                   frameon=True, facecolor="white", edgecolor="#D0D0D0")
    else:
        ax1.legend([line_time[0]], ["DDIM Inference Time (Right)"],
                   loc="upper center", fancybox=True, shadow=False,
                   frameon=True, facecolor="white", edgecolor="#D0D0D0")

    # Grassetto su tutti i tick
    for label in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
        label.set_weight('bold')

    plt.title("DDIM Sampling Trade-Off: Precision vs Latency", pad=15)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches="tight")
        print(f"Grafico salvato in: {filename}")
    plt.show()
    plt.close()


# --- GENERAZIONE DEI 3 GRAFICI ---

# 1. Solo Inference Time
build_tradeoff_plot(
    show_baseline=False,
    show_ddim_mse=False,
    filename="perfomance_test/ddim_tradeoff_step1_time.pdf"
)

# 2. Inference Time + Baseline DDPM
build_tradeoff_plot(
    show_baseline=True,
    show_ddim_mse=False,
    filename="perfomance_test/ddim_tradeoff_step2_baseline.pdf"
)

# 3. Completo: Inference Time + Baseline DDPM + DDIM MSE (con punto di ottimo)
build_tradeoff_plot(
    show_baseline=True,
    show_ddim_mse=True,
    filename="perfomance_test/ddim_tradeoff_step3_full.pdf"
)