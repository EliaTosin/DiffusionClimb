import os
import matplotlib.pyplot as plt
import pandas as pd

"""
script to produce the loss_convergence.pdf file (results - training setup) from the downloaded tensorboard log
"""

# --- CONFIGURAZIONE NOMI FILE ---
# Sostituisci questi nomi con i nomi reali dei tuoi file CSV dentro csv_plots (attuale)
FOLDER_PATH = "./"
FILE_STEP_TRAIN = "step_runs_diffusion_NUOVO_Loss_train.csv"
FILE_STEP_VAL = "step_runs_diffusion_NUOVO_Loss_val.csv"
FILE_STILL_TRAIN = "still_runs_diffusion_retroaction_Loss_train.csv"
FILE_STILL_VAL = "still_runs_diffusion_retroaction_Loss_val.csv"

# --- CONFIGURAZIONE STILE ACCADEMICO ---
plt.rcParams.update({
    "font.family": "serif",
    "font.weight": "bold",      # Forza il grassetto globalmente dove possibile
    "font.size": 14,            # Alzato il font base
    "axes.labelsize": 16,       # Più grande per Mean Squared Error e Training Epochs
    "axes.labelweight": "bold", # Grassetto per i titoli degli assi
    "axes.titlesize": 18,       # Più grande per il titolo principale (Loss Evolution)
    "axes.titleweight": "bold",# Grassetto per il titolo principale
    "xtick.labelsize": 14,      # Numeri sull'asse X più grandi
    "ytick.labelsize": 14,      # Numeri sull'asse Y più grandi
    "legend.fontsize": 13,      # Testo della legenda più visibile
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


def load_tensorboard_csv(file_name):
    path = os.path.join(FOLDER_PATH, file_name)
    # TensorBoard esporta solitamente con colonne: Wall time, Step, Value
    df = pd.read_csv(path)
    # Rinominiamo per comodità (Step corrisponde all'epoca se loggato a fine epoca)
    df = df.rename(columns={"Step": "Epoch", "Value": "Loss"})
    return df


try:
    # Caricamento dati
    df_step_train = load_tensorboard_csv(FILE_STEP_TRAIN)
    df_step_val = load_tensorboard_csv(FILE_STEP_VAL)
    df_still_train = load_tensorboard_csv(FILE_STILL_TRAIN)
    df_still_val = load_tensorboard_csv(FILE_STILL_VAL)

    # Creazione della figura
    plt.figure(figsize=(10, 5.5))

    # --- PLOT CURVE STEP (DYNAMIC LOCOMOTION) ---
    # Usiamo linea continua per il Train e tratteggiata per la Validation
    plt.plot(df_step_train["Epoch"], df_step_train["Loss"],
             label="Step Task (Train)", color="#009688", linewidth=2, linestyle="-")
    plt.plot(df_step_val["Epoch"], df_step_val["Loss"],
             label="Step Task (Val)", color="#80cbc4", linewidth=1.8, linestyle="--")

    # --- PLOT CURVE STILL (STATIC POSTURAL) ---
    plt.plot(df_still_train["Epoch"], df_still_train["Loss"],
             label="Still Task (Train)", color="#7b1fa2", linewidth=2, linestyle="-")
    plt.plot(df_still_val["Epoch"], df_still_val["Loss"],
             label="Still Task (Val)", color="#ce93d8", linewidth=1.8, linestyle="--")

    # --- ABBELLIMENTI E ASSI ---
    plt.xlabel("Training Epochs")
    plt.ylabel("Mean Squared Error (MSE) Loss")
    plt.gca().tick_params(axis='both', labelsize=14, labelfontfamily='serif')
    # Forza il bold sui ticks in modo esplicito
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_weight('bold')
    plt.title("Loss Evolution")

    # Limiti assi (regola se vuoi zoomare o escludere i primissimi step esplosivi)
    # Imposta un piccolo margine del 3% a destra e sinistra in automatico, oppure manuale:
    max_epoch = max(df_step_train["Epoch"].max(), df_still_train["Epoch"].max())
    plt.xlim(-2, max_epoch + 3)  # Estende leggermente la griglia oltre lo 0 e oltre il 150

    # Se vuoi allontanare un filo anche i punti dal soffitto (0.025):
    plt.ylim(-0.001, 0.026)

    # Griglia e Legenda
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", prop={'weight': 'bold'})

    # Ottimizzazione spazi
    plt.tight_layout()

    # Salvataggio in PDF Vettoriale (Perfetto per LaTeX)
    output_pdf = os.path.join(FOLDER_PATH, "loss_convergence.pdf")

    plt.savefig(output_pdf, bbox_inches="tight", dpi=300)

    print(f"Grafico salvato con successo in:\n {output_pdf}")
    plt.show()

except FileNotFoundError as e:
    print(f"Errore: Verificare i nomi dei file CSV. Dettaglio: {e}")
except Exception as e:
    print(f"Si è verificato un errore durante il plotting: {e}")