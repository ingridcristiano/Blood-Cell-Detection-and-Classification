import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.semi_supervised import LabelSpreading
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
    print("--- 1. Caricamento e Validazione ---")
    percorso_csv = os.path.join(CARTELLA_CSV, 'features_cellule_train.csv')

    if not os.path.exists(percorso_csv):
        print(f"[ERRORE] File non trovato: {percorso_csv}")
        exit()

    df = pd.read_csv(percorso_csv)
    df.columns = df.columns.str.strip()
    df['GroundTruth_Label'] = pd.Series(np.nan, dtype="object")

    immagini_uniche = df['ImageName'].unique()
    print(f"Trovate {len(immagini_uniche)} immagini nel CSV. Analizzo i file JSON in: {CARTELLA_JSON}")

    for img_name in immagini_uniche:
        # Costruiamo il nome del file JSON aspettandoci 'nome.estensione.json'
        file_json = os.path.join(CARTELLA_JSON, img_name + ".json")

        if not os.path.exists(file_json):
            continue

        with open(file_json, 'r') as f:
            dati_ann = json.load(f)
            print(f"✅ Trovato JSON per: {img_name}")

        for idx in df[df['ImageName'] == img_name].index:
            cpp_box = [df.at[idx, 'BoxX'], df.at[idx, 'BoxY'], df.at[idx, 'BoxX'] + df.at[idx, 'BoxW'],
                       df.at[idx, 'BoxY'] + df.at[idx, 'BoxH']]
            area = df.at[idx, 'Area']

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
            if miglior_iou >= soglia:
                df.at[idx, 'GroundTruth_Label'] = miglior_label
            elif area < 100:
                df.at[idx, 'GroundTruth_Label'] = 'Rumore'

    return df


# =========================================================================
# FLUSSO PRINCIPALE
# =========================================================================
def train_semi_supervised():
    df = carica_e_valida_train()

    cols_to_use = [col for col in FEATURES if col in df.columns]
    X = df[cols_to_use].values

    y_semi = []
    for index, row in df.iterrows():
        label_medico = row.get('GroundTruth_Label', np.nan)
        if pd.notna(label_medico) and label_medico in mappa_classi:
            y_semi.append(mappa_classi[label_medico])
        else:
            y_semi.append(-1)

    # Controllo di sicurezza prima del crash
    if all(y == -1 for y in y_semi):
        print("\n[ERRORE CRITICO] Non ho trovato nessuna etichetta valida nei JSON!")
        print("Controlla che i nomi delle classi nel JSON siano corrette (WBC, RBC, ecc.)")
        exit()

    print(f"--- 2. Pulizia e Standardizzazione ---")
    imputer = SimpleImputer(strategy='mean')
    X_clean = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    print("--- 3. Label Spreading ---")
    modello_semi = LabelSpreading(kernel='knn', n_neighbors=15, alpha=0.2, n_jobs=-1)
    modello_semi.fit(X_scaled, y_semi)

    y_transduced = modello_semi.transduction_
    df['CellType_Predetto_ML'] = [list(mappa_classi.keys())[list(mappa_classi.values()).index(val)] for val in
                                  y_transduced]

    df.to_csv(os.path.join(CARTELLA_CSV, 'features_cellule_corretto_da_ML.csv'), index=False)

    print("--- 4. Addestramento Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_model.fit(X_scaled, y_transduced)

    print("--- 5. Salvataggio ---")
    joblib.dump(modello_semi, os.path.join(BASE_DIR, 'modello_label_spreading.pkl'))
    joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler_progetto.pkl'))
    joblib.dump(imputer, os.path.join(BASE_DIR, 'imputer_progetto.pkl'))
    joblib.dump(rf_model, os.path.join(BASE_DIR, 'random_forest.pkl'))

    print("\n[FINE] Tutto completato! Modelli salvati.")


if __name__ == "__main__":
    train_semi_supervised()