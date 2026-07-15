import matplotlib.pyplot as plt
import numpy as np

"""
script to produce the ddim_tradeoff_plot.pdf file (results - training setup)
"""

import pandas as pd
import numpy as np

# 1. Carica il file CSV generato dal benchmark
df = pd.read_csv("perfomance_test/inference_tradeoff_metrics.csv")

# 2. Estrai i dati per la curva DDIM (escludendo la riga del DDPM)
df_ddim = df[df["Sampler"] == "DDIM"].sort_values(by="Steps", ascending=False)
steps = df_ddim["Steps"].to_numpy()
mse_ddim = df_ddim["MSE (rad^2)"].to_numpy()
time_ddim = df_ddim["Inference Time (ms)"].to_numpy()

# 3. Estrai i dati per la baseline DDPM (la riga dove Sampler è DDPM)
df_ddpm = df[df["Sampler"] == "DDPM"]
ddpm_mse_baseline = df_ddpm["MSE (rad^2)"].values[0]
ddpm_time_baseline = df_ddpm["Inference Time (ms)"].values[0]

# --- CONFIGURAZIONE GRAFICA ACCADEMICA ---
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

fig, ax1 = plt.subplots(figsize=(8, 5))
grid_color = "#E0E0E0"
ax1.grid(True, which="both", linestyle="--", linewidth=0.5, color=grid_color)

# Per rendere leggibile l'asse X con step discreti e non lineari, usiamo le stringhe
x_labels = [str(s) for s in steps]
x_indices = np.arange(len(steps))

# --- ASSE Y1: MSE LOSS (Scala Logaritmica per evidenziare l'esplosione dell'errore) ---
color_mse = "#2C3E50"  # Grigio scuro/Blu accademico
ax1.set_xlabel("DDIM Sampling Steps", fontweight="bold", labelpad=10)
ax1.set_ylabel("Mean Squared Error (MSE) [$rad^2$]", color=color_mse, fontweight="bold")
# ax1.set_yscale("log")  # La scala logaritmica è fondamentale qui!
ax1.set_ylim(0, mse_ddim[-2])

# Plot della curva MSE
line_mse = ax1.plot(x_indices, mse_ddim, marker='o', linewidth=2.5, color=color_mse,
                    label="DDIM Trajectory MSE", zorder=3)

# Linea Orizzontale per la Baseline DDPM
line_ddpm_base = ax1.axhline(y=ddpm_mse_baseline, color="red", linestyle="--", linewidth=1.5,
                             label="DDPM Baseline (500 steps)", zorder=2)
# --- SCRITTA SOPRA LA LINEA BASELINE ---
# Posiziona il testo vicino all'inizio dell'asse X (es. coordinata x = 0.2)
# e leggermente sopra il valore di MSE del DDPM (moltiplicato per 1.15 visto che l'asse è logaritmico)
ax1.text(5.5, ddpm_mse_baseline * 1.10, "DDPM Baseline \n (500 steps)",
         color="red", fontweight="bold", fontsize=10,
         va="bottom", ha="left", zorder=4)

ax1.tick_params(axis='y', labelcolor=color_mse)
ax1.set_xticks(x_indices)
ax1.set_xticklabels(x_labels)

# Evidenziamo con un cerchio colorato l'ottimo a 5 step (indice 5 nella lista)
opt_idx = 5
ax1.scatter(opt_idx, mse_ddim[opt_idx], color="orange", s=150, facecolors='none',
            edgecolors='orange', linewidths=3, zorder=5, label="Selected Optimal (5 steps)")

# --- ASSE Y2: INFERENCE TIME ---
ax2 = ax1.twinx()
color_time = "#16A085"  # Verde petrolio
ax2.set_ylabel("Single-Inference Latency [ms]", color=color_time, fontweight="bold")

# Plot della curva dei tempi (tratteggiata per non confondere con l'MSE)
line_time = ax2.plot(x_indices, time_ddim, marker='s', linestyle=":", linewidth=2, color=color_time,
                     label="DDIM Inference Time", zorder=3)
ax2.tick_params(axis='y', labelcolor=color_time)

# --- COORDINAMENTO LEGENDA UNICA (Esclusa Baseline e ottimizzata a 1 colonna) ---
# Uniamo solo la curva MSE (blu) e la curva Tempo (verde)
lines = line_mse + line_time

# Creiamo l'indicatore grafico per l'ottimo a 5 step
opt_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                        markeredgecolor='orange', markersize=10, markeredgewidth=2)

# Uniamo i marker e i testi per la legenda definitiva
final_lines = [line_mse[0], line_time[0], opt_marker]
final_labels = [
    "DDIM Trajectory MSE (Left)",
    "DDIM Inference Time (Right)",
    "Optimal Selected Spot (5 Steps)"
]

# Posizioniamo la legenda in alto a destra, dentro il grafico, incolonnata (ncol=1)
ax1.legend(final_lines, final_labels, loc="upper center",
           fancybox=True, shadow=False, ncol=1, frameon=True, facecolor="white", edgecolor="#D0D0D0")

# Forza il font in bold su tutti i ticks degli assi
for label in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
    label.set_weight('bold')

plt.title("DDIM Sampling Trade-Off: Precision vs Latency", pad=15)
plt.tight_layout()

# Salvataggio in formato PDF vettoriale per LaTeX
plt.savefig("perfomance_test/ddim_tradeoff_plot.pdf", bbox_inches="tight")
print("Grafico salvato con successo in 'ddim_tradeoff_plot.pdf'!")