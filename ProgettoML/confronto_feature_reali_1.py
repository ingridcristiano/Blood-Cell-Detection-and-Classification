import pandas as pd
import json
import os


def calcola_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def valida_dati(fase):
    # fase può essere 'train' o 'test'
    percorso_csv_input = os.path.join('csv', f'features_cellule_{fase}.csv')
    percorso_csv_output = os.path.join('csv', f'features_cellule_{fase}_VALIDATE.csv')
    cartella_ann = os.path.join('..', 'ProgettoIPA', 'archive', fase, 'ann')

    # Se il test di output originale si chiama "features_cellule_test_VALIDATE.csv"
    if fase == 'train':
        percorso_csv_output = os.path.join('csv', 'features_cellule_VALIDATE.csv')

    print(f"\n[{fase.upper()}] Validazione in corso da: {percorso_csv_input}")
    try:
        df = pd.read_csv(percorso_csv_input, sep=None, engine='python')
        df.columns = df.columns.str.strip()
        df['ImageName'] = df['ImageName'].astype(str).str.strip()
    except Exception as e:
        print(f"ERRORE: Impossibile leggere il CSV: {e}")
        return

    df['GroundTruth_Label'] = 'Rumore'
    df['IoU_Score'] = 0.0

    if not os.path.exists(cartella_ann):
        print(f"  [ALLARME] Cartella JSON {fase} non trovata: {cartella_ann}")
        return

    veri_positivi, falsi_positivi = 0, 0

    for img_name in df['ImageName'].unique():
        json_path = os.path.join(cartella_ann, img_name + ".json")
        if not os.path.exists(json_path): continue

        with open(json_path, 'r') as f:
            dati_ann = json.load(f)

        righe_immagine = df[df['ImageName'] == img_name].index
        for idx in righe_immagine:
            cpp_box = [df.at[idx, 'BoxX'], df.at[idx, 'BoxY'], df.at[idx, 'BoxX'] + df.at[idx, 'BoxW'],
                       df.at[idx, 'BoxY'] + df.at[idx, 'BoxH']]
            miglior_iou, miglior_label_medico = 0.0, 'Rumore'

            for obj in dati_ann.get('objects', []):
                pts = obj['points']['exterior']
                medico_box = [pts[0][0], pts[0][1], pts[1][0], pts[1][1]]
                iou = calcola_iou(cpp_box, medico_box)
                if iou > miglior_iou:
                    miglior_iou = iou
                    cls = obj['classTitle']
                    miglior_label_medico = 'GlobuloBianco' if cls == 'WBC' else 'GlobuloRosso' if cls == 'RBC' else 'Piastrina' if cls == 'Platelets' else cls

            df.at[idx, 'IoU_Score'] = miglior_iou
            if miglior_iou >= 0.01:
                df.at[idx, 'GroundTruth_Label'] = miglior_label_medico
                veri_positivi += 1
            else:
                falsi_positivi += 1

    print(f"  ✅ Cellule confermate: {veri_positivi} | ❌ Macchie scartate (Rumore): {falsi_positivi}")
    df.to_csv(percorso_csv_output, index=False)


if __name__ == '__main__':
    valida_dati('train')
    valida_dati('test')