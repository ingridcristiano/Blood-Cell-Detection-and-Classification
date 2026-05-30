import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# =========================================================================
# CONFIGURAZIONE DINAMICA DEI PERCORSI
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARTELLA_CSV = os.path.join(BASE_DIR, 'csv')
CARTELLA_JSON = os.path.join(BASE_DIR, 'archive', 'train', 'ann')

FEATURES = [
    'Area', 'Perimeter', 'Circularity', 'AspectRatio',
    'MeanBlue', 'MeanGreen', 'MeanRed', 'MeanValue', 'MinValue', 'MaxValue',
    'MeanSaturation', 'MinSat', 'MaxSat', 'TextureValue', 'TextureSat', 'TextureLaplacian',
    'Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7'
]

# Il Rumore torna a essere la classe 3 (Lo scarto spietato)
mappa_classi = {'GlobuloBianco': 0, 'GlobuloRosso': 1, 'Piastrina': 2, 'Rumore': 3}


# =========================================================================
# FUNZIONI DI SUPPORTO
# =========================================================================
def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    return interArea / float(
        ((boxA[2] - boxA[0]) * (boxA[3] - boxA[1])) + ((boxB[2] - boxB[0]) * (boxB[3] - boxB[1])) - interArea)


def carica_e_valida_train():
    print("--- 1. Caricamento e Validazione (Tutto ciò che non matcha è Rumore) ---")
    percorso_csv = os.path.join(CARTELLA_CSV, 'features_cellule_train.csv')

    if not os.path.exists(percorso_csv):
        print(f"[ERRORE] File non trovato: {percorso_csv}")
        exit()

    df = pd.read_csv(percorso_csv)
    df.columns = df.columns.str.strip()
    df['GroundTruth_Label'] = pd.Series(np.nan, dtype="object")

    immagini_uniche = df['ImageName'].unique()
    print(f"Trovate {len(immagini_uniche)} immagini nel CSV. Analizzo i JSON...")

    for img_name in immagini_uniche:
        file_json = os.path.join(CARTELLA_JSON, img_name + ".json")

        if not os.path.exists(file_json):
            continue

        with open(file_json, 'r') as f:
            dati_ann = json.load(f)

        for idx in df[df['ImageName'] == img_name].index:
            bx, by = df.at[idx, 'BoxX'], df.at[idx, 'BoxY']
            bw, bh = df.at[idx, 'BoxW'], df.at[idx, 'BoxH']
            cpp_box = [bx, by, bx + bw, by + bh]

            miglior_iou, miglior_label = 0.0, None
            for obj in dati_ann.get('objects', []):
                pts = obj['points']['exterior']
                cls = obj['classTitle'].strip().upper()
                if cls == 'WBC':
                    cls_name = 'GlobuloBianco'
                elif cls == 'RBC':
                    cls_name = 'GlobuloRosso'
                elif cls in ['PLATELETS', 'PLATELET']:
                    cls_name = 'Piastrina'
                else:
                    cls_name = cls

                iou = compute_iou(cpp_box, [pts[0][0], pts[0][1], pts[1][0], pts[1][1]])
                if iou > miglior_iou:
                    miglior_iou = iou
                    miglior_label = cls_name

            soglia = 0.1 if miglior_label == 'Piastrina' else 0.35

            # LA REGOLA CHE VUOI TU:
            if miglior_iou >= soglia:
                df.at[idx, 'GroundTruth_Label'] = miglior_label
            else:
                # Nessun match col medico? Diventa Rumore, anche se fosse una cellula vera!
                df.at[idx, 'GroundTruth_Label'] = 'Rumore'

    return df


# =========================================================================
# FLUSSO PRINCIPALE
# =========================================================================
def train_supervised():
    df = carica_e_valida_train()

    df['Target'] = df['GroundTruth_Label'].map(mappa_classi)

    cols_to_use = [col for col in FEATURES if col in df.columns]
    X = df[cols_to_use].values
    y = df['Target'].values

    print(f"--- 2. Pulizia e Standardizzazione ---")
    imputer = SimpleImputer(strategy='mean')
    X_clean = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    print("--- 3. Addestramento Random Forest (Con annotazioni parziali) ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_model.fit(X_scaled, y)

    print("--- 4. Salvataggio ---")
    joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler_progetto.pkl'))
    joblib.dump(imputer, os.path.join(BASE_DIR, 'imputer_progetto.pkl'))
    joblib.dump(rf_model, os.path.join(BASE_DIR, 'random_forest.pkl'))

    print("\n[FINE] Addestramento completato! Modelli salvati.")


if __name__ == "__main__":
    train_supervised()