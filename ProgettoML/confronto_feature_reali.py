import pandas as pd
import json
import os


def calcola_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


def valida_dati():
    print("1. Caricamento del dataset C++ (features_cellule.csv)...")
    try:
        df = pd.read_csv('features_cellule.csv', sep=None, engine='python')
        df.columns = df.columns.str.strip()
        df['ImageName'] = df['ImageName'].astype(str).str.strip()
    except Exception as e:
        print(f"ERRORE: Impossibile leggere il CSV: {e}")
        return

    df['GroundTruth_Label'] = 'Rumore'
    df['IoU_Score'] = 0.0

    # --- PERCORSO DELLA CARTELLA AGGIORNATO ---
    # La 'r' prima delle virgolette serve per gestire i backslash di Windows
    cartella_ann = r'C:\Progetti\Template C++\ProgettoML\train\ann'
    soglia_iou = 0.3

    print(f"-> Sto cercando i file JSON esattamente qui: {cartella_ann}")

    if not os.path.exists(cartella_ann):
        print("\n[ALLARME] La cartella non esiste in questa posizione! Controlla di aver scritto bene il percorso.")
        return

    print("2. Incrocio geometrico con i file JSON del medico...")
    immagini_uniche = df['ImageName'].unique()

    veri_positivi = 0
    falsi_positivi = 0

    for img_name in immagini_uniche:
        json_path = os.path.join(cartella_ann, img_name + ".json")

        if not os.path.exists(json_path):
            print(f"  [Warning] JSON non trovato: {json_path}")
            continue

        with open(json_path, 'r') as f:
            dati_ann = json.load(f)

        righe_immagine = df[df['ImageName'] == img_name].index

        for idx in righe_immagine:
            cpp_box = [
                df.at[idx, 'BoxX'],
                df.at[idx, 'BoxY'],
                df.at[idx, 'BoxX'] + df.at[idx, 'BoxW'],
                df.at[idx, 'BoxY'] + df.at[idx, 'BoxH']
            ]

            miglior_iou = 0.0
            miglior_label_medico = 'Rumore'

            for obj in dati_ann.get('objects', []):
                pts = obj['points']['exterior']
                medico_box = [pts[0][0], pts[0][1], pts[1][0], pts[1][1]]

                iou = calcola_iou(cpp_box, medico_box)

                if iou > miglior_iou:
                    miglior_iou = iou
                    cls = obj['classTitle']
                    if cls == 'WBC':
                        miglior_label_medico = 'GlobuloBianco'
                    elif cls == 'RBC':
                        miglior_label_medico = 'GlobuloRosso'
                    elif cls == 'Platelets':
                        miglior_label_medico = 'Piastrina'
                    else:
                        miglior_label_medico = cls

            df.at[idx, 'IoU_Score'] = miglior_iou
            if miglior_iou >= soglia_iou:
                df.at[idx, 'GroundTruth_Label'] = miglior_label_medico
                veri_positivi += 1
            else:
                falsi_positivi += 1

    print("\n=======================================================")
    print("--- RISULTATO (GROUND TRUTH) ---")
    print(f"✅ Cellule validate e CONFERMATE : {veri_positivi}")
    print(f"❌ Macchie SCARTATE come rumore: {falsi_positivi}")
    print("=======================================================\n")

    df.to_csv('features_cellule_VALIDATE.csv', index=False)


if __name__ == '__main__':
    valida_dati()