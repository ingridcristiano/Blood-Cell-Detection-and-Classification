import pandas as pd
import json
import os


def calcola_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def valida_tutto():
    cartella_script = os.path.dirname(os.path.abspath(__file__))

    # Capisce da solo se sei dentro "csv" o nella cartella principale
    if os.path.basename(cartella_script) == "csv":
        base_dir = os.path.dirname(cartella_script)
        cartella_csv = cartella_script
    else:
        base_dir = cartella_script
        cartella_csv = os.path.join(cartella_script, 'csv')

    soglia_iou = 0.35

    # Facciamo il giro due volte: prima per il TRAIN, poi per il TEST
    fasi = ['train', 'test']

    for fase in fasi:
        print(f"\n=======================================================")
        print(f"🔄 AVVIO VALIDAZIONE PER IL DATASET: {fase.upper()}")
        print(f"=======================================================")

        percorso_csv_input = os.path.join(cartella_csv, f'features_cellule_{fase}.csv')

        # Rispettiamo i nomi esatti che il modello si aspetterà
        if fase == 'train':
            percorso_csv_output = os.path.join(cartella_csv, 'features_cellule_VALIDATE.csv')
        else:
            percorso_csv_output = os.path.join(cartella_csv, 'features_cellule_test_VALIDATE.csv')

        cartella_ann = os.path.join(base_dir, 'archive', fase, 'ann')

        if not os.path.exists(percorso_csv_input):
            print(f"[ERRORE] Manca il file C++ di base: {percorso_csv_input}")
            continue

        if not os.path.exists(cartella_ann):
            print(f"[ERRORE] Manca la cartella JSON: {cartella_ann}")
            continue

        df = pd.read_csv(percorso_csv_input, sep=None, engine='python')
        df.columns = df.columns.str.strip()
        df['ImageName'] = df['ImageName'].astype(str).str.strip()

        df['GroundTruth_Label'] = 'Sconosciuto'
        df['IoU_Score'] = 0.0

        immagini_uniche = df['ImageName'].unique()
        veri_positivi = 0
        rumore_confermato = 0
        sconosciuti = 0

        for img_name in immagini_uniche:
            json_path = os.path.join(cartella_ann, img_name + ".json")
            if not os.path.exists(json_path): continue

            with open(json_path, 'r') as f:
                dati_ann = json.load(f)

            righe_immagine = df[df['ImageName'] == img_name].index

            for idx in righe_immagine:
                cpp_box = [df.at[idx, 'BoxX'], df.at[idx, 'BoxY'],
                           df.at[idx, 'BoxX'] + df.at[idx, 'BoxW'], df.at[idx, 'BoxY'] + df.at[idx, 'BoxH']]
                area_oggetto = df.at[idx, 'Area']

                miglior_iou = 0.0
                miglior_label_medico = 'Sconosciuto'

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
                    if area_oggetto < 150:
                        df.at[idx, 'GroundTruth_Label'] = 'Rumore'
                        rumore_confermato += 1
                    else:
                        df.at[idx, 'GroundTruth_Label'] = 'Sconosciuto'
                        sconosciuti += 1

        print(f"✅ Ancore Certe : {veri_positivi}")
        print(f"🗑️ Rumore Certo : {rumore_confermato}")
        print(f"❓ Sconosciuti  : {sconosciuti}")

        df.to_csv(percorso_csv_output, index=False)
        print(f"-> File generato: {percorso_csv_output}\n")


if __name__ == '__main__':
    valida_tutto()