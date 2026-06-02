import torch
import os


def aggiorna_checkpoint_retrocompatibile(input_path, output_path, chiave, valore):
    """
    Apre un file .pt, aggiunge/modifica una chiave nella radice del dizionario e salva un nuovo file.
    """
    if not os.path.exists(input_path):
        print(f"❌ Errore: Il file {input_path} non esiste.")
        return

    print(f"📥 Lettura del checkpoint: {input_path}")

    # weights_only=False è CRITICO qui, perché i tuoi file hanno i metadati Numpy (traj_mean, ecc.)
    try:
        checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ Errore durante il caricamento: {e}")
        return

    # Verifichiamo se è un dizionario (lo standard dei tuoi salvataggi)
    if not isinstance(checkpoint, dict):
        print("❌ Errore: Il file non contiene un dizionario alla radice.")
        return

    # Controllo pre-esistente
    if chiave in checkpoint:
        print(f"❌ Attenzione: La chiave '{chiave}' esiste già con valore '{checkpoint[chiave]}'")
        # return

    # Aggiunta/Modifica della chiave
    checkpoint[chiave] = valore
    print(f"✅ Inserita chiave '{chiave}' = {valore}")

    # Salvataggio del nuovo file
    torch.save(checkpoint, output_path)
    print(f"💾 Nuovo checkpoint salvato con successo in: {output_path}")


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURAZIONE
    # =========================================================================

    # 1. Il tuo file vecchio (quello senza Dropout)
    FILE_VECCHIO = "diff_model_vel_2hid_cond.pt"

    # 2. Come vuoi chiamare il file aggiornato (ti consiglio di non sovrascrivere l'originale!)
    FILE_NUOVO = f'{FILE_VECCHIO.split(".")[0]}_compatibile.pt'

    # =========================================================================
    # ESECUZIONE
    # =========================================================================

    # Se nella tua nuova classe hai aggiunto il parametro "use_dropout"
    # e lo vuoi impostare a False per i modelli vecchi:
    aggiorna_checkpoint_retrocompatibile(FILE_VECCHIO, FILE_NUOVO, chiave="dropout", valore=0.1)

    # Se ti serve anche il valore numerico del dropout_rate,
    # puoi decommentare la riga sotto e fare una seconda passata sul file nuovo:
    # aggiorna_checkpoint_retrocompatibile(FILE_NUOVO, FILE_NUOVO, chiave="dropout_rate", valore=0.0)