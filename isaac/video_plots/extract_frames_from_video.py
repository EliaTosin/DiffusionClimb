import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def extract_gait_keyframes(
    video_path,
    start_frame,
    end_frame,
    num_frames=8,
    output_dir="gait_frames",
    figure_name="gait_sequence_compact.png",
):
    """
    Estrae num_frames fotogrammi distribuiti uniformemente tra start_frame ed end_frame.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il video {video_path}")
        return

    # Lettura delle proprietà del video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps if fps > 0 else 0

    print(
        f"Info Video -> Frame totali: {total_frames} | FPS: {fps:.2f} | Durata: {duration_sec:.2f}s"
    )

    # Limita end_frame al massimo disponibile per evitare letture fuori range
    end_frame = min(end_frame, total_frames - 1)

    # Calcola gli indici dei frame distribuiti uniformemente
    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    extracted_frames = []

    print(
        f"Estrazione di {num_frames} frame compresi tra {start_frame} e {end_frame}..."
    )

    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # OpenCV legge in BGR, convertiamo in RGB per Matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            extracted_frames.append(frame_rgb)

            # Salva anche il frame singolo in alta qualità
            single_frame_path = os.path.join(
                output_dir, f"frame_{i + 1}_idx{idx}.png"
            )
            cv2.imwrite(
                single_frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            )
        else:
            print(f"Attenzione: Impossibile leggere il frame all'indice {idx}")

    cap.release()

    # --- CREAZIONE DELLA FIGURA UNICA (STRIP ORIZZONTALE) ---
    num_rows = 2
    fig, axes = plt.subplots(
        num_rows, num_frames // num_rows, figsize=(10, 3.5), dpi=300
    )
    for i, (ax, img) in enumerate(zip(axes.flatten(), extracted_frames)):
        # 2. CROP DELL'IMMAGINE (Rimuove sfondo nero sopra e sotto + bordi laterali inutili)
        h, w, _ = img.shape
        crop_img = img[
            int(h * 0.15) : int(h * 0.85), int(w * 0.05) : int(w * 0.95)
        ]

        ax.imshow(crop_img)

        # 3. TITOLO PIÙ PICCOLO E VICINO
        ax.set_title(f"$t_{{{i + 1}}}$", fontsize=10, fontweight="bold", pad=2)
        ax.axis("off")

    # 4. CONTROLLO MANUALE DELLO SPAZIAMENTO (Elimina i vuoti bianchi)
    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.90,
        bottom=0.01,
        wspace=0.05,
        hspace=0.15,
    )
    if not figure_name.endswith(".png"):
        figure_name = figure_name + ".png"

    # plt.savefig(figure_name, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()


if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    VIDEO_PATH = "/home/etosin/Videos/Tesi/canva/canva_isometric_TRIM_LAG_12s.mp4"

    START_FRAME = 10
    END_FRAME = 2000

    extract_gait_keyframes(
        video_path=VIDEO_PATH,
        start_frame=START_FRAME,
        end_frame=END_FRAME,
        num_frames=8,
        output_dir="./gait_sequence_results_ISOMETRIC",
        figure_name="planar_sequence.png",
    )