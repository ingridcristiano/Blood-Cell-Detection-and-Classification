import cv2
import os
import pandas as pd
import numpy as np


def visualizza_confronto_completo():
    # --- 1. DEFINIZIONE DEI PERCORSI RELATIVI ---
    cartella_output_prima = 'output'
    cartella_train_originali = os.path.join('archive', 'train', 'img')
    file_csv_ml = os.path.join('csv', 'features_cellule_corretto_da_ML.csv')

    print("=========================================================")
    print(" 🎬 AVVIO ISPETTORE VISIVO: C++ vs LABEL SPREADING (ML)")
    print("=========================================================\n")

    # --- 2. CONTROLLO FILE E CARTELLE ---
    if not os.path.exists(file_csv_ml):
        print(f"[ERRORE] File CSV non trovato: {file_csv_ml}")
        return
    if not os.path.exists(cartella_output_prima):
        print(f"[ERRORE] Cartella output non trovata: {cartella_output_prima}")
        return
    if not os.path.exists(cartella_train_originali):
        print(f"[ERRORE] Cartella immagini originali non trovata: {cartella_train_originali}")
        return

    # --- 3. LETTURA DEL DATASET CORRETTO DALL'IA ---
    df = pd.read_csv(file_csv_ml)
    immagini_uniche = df['ImageName'].unique()

    if len(immagini_uniche) == 0:
        print("[AVVISO] Il file CSV è vuoto.")
        return

    # Mappa Colori per OpenCV (Attenzione: OpenCV usa BGR, non RGB!)
    colori_bgr = {
        'GlobuloRosso': (0, 0, 255),  # Rosso puro
        'GlobuloBianco': (255, 0, 0),  # Blu puro
        'Piastrina': (0, 255, 0),  # Verde puro
        'Rumore': (150, 150, 150)  # Grigio
    }

    print(f"✅ Trovate {len(immagini_uniche)} immagini nel CSV.")
    print("--------------------------------------------------")
    print(" 🕹️  COMANDI DEL VISUALIZZATORE:")
    print(" - Premi [QUALSIASI TASTO] per passare all'immagine successiva.")
    print(" - Premi [Q] oppure [ESC] per chiudere definitivamente.")
    print("--------------------------------------------------\n")

    # --- 4. CICLO DI SCORRIMENTO IMMAGINI ---
    for i, nome_immagine in enumerate(immagini_uniche, 1):
        percorso_sx = os.path.join(cartella_output_prima, nome_immagine)
        percorso_dx = os.path.join(cartella_train_originali, nome_immagine)

        # Caricamento delle due immagini
        img_sx = cv2.imread(percorso_sx)
        img_dx = cv2.imread(percorso_dx)

        # Se manca una delle due, la saltiamo con un avviso
        if img_sx is None or img_dx is None:
            print(f"  [AVVISO] Impossibile caricare {nome_immagine} (manca in output o in archive). Salto...")
            continue

        # --- 5. DISEGNO DEI RETTANGOLI DELL'IA (SULL'IMMAGINE DESTRA) ---
        dati_immagine = df[df['ImageName'] == nome_immagine]
        for _, row in dati_immagine.iterrows():
            # Coordinate dal C++
            x, y = int(row['BoxX']), int(row['BoxY'])
            w, h = int(row['BoxW']), int(row['BoxH'])

            # Etichetta decisa dal Label Spreading
            etichetta_ia = str(row['CellType_Predetto_ML'])
            colore = colori_bgr.get(etichetta_ia, (255, 255, 255))  # Default bianco

            # Disegno rettangolo e testo sull'immagine originale
            cv2.rectangle(img_dx, (x, y), (x + w, y + h), colore, 2)
            cv2.putText(img_dx, etichetta_ia[:3], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colore, 2)

        # --- 6. AGGIUNTA DEI TITOLI IN ALTO ---
        # Creiamo un box nero in alto per far risaltare il testo
        cv2.rectangle(img_sx, (0, 0), (600, 40), (0, 0, 0), -1)
        cv2.putText(img_sx, "1. PRIMA (Output C++ / Medico)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                    2)

        cv2.rectangle(img_dx, (0, 0), (600, 40), (0, 0, 0), -1)
        cv2.putText(img_dx, "2. DOPO (Correzione Label Spreading ML)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        # --- 7. UNIONE E RIDIMENSIONAMENTO ---
        # Affianchiamo le due immagini orizzontalmente in una sola finestra
        img_combinata = cv2.hconcat([img_sx, img_dx])

        # Ridimensionamento intelligente per gli schermi classici (Max 1800x900)
        altezza, larghezza = img_combinata.shape[:2]
        max_w, max_h = 1800, 900
        if altezza > max_h or larghezza > max_w:
            scala = min(max_w / larghezza, max_h / altezza)
            img_combinata = cv2.resize(img_combinata, (int(larghezza * scala), int(altezza * scala)))

        # --- 8. VISUALIZZAZIONE ---
        titolo_finestra = f"Confronto Finale [{i}/{len(immagini_uniche)}] - {nome_immagine}"
        cv2.imshow("Ispettore IA", img_combinata)

        tasto = cv2.waitKey(0) & 0xFF

        if tasto == 27 or tasto == ord('q'):
            print("\n🛑 Visualizzazione interrotta dall'utente.")
            break

    cv2.destroyAllWindows()
    print("\n✅ Fine del set di immagini.")


if __name__ == "__main__":
    visualizza_confronto_completo()