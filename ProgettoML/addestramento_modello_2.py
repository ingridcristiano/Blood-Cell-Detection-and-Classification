import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.semi_supervised import LabelSpreading


def train_semi_supervised():
    percorso_input = os.path.join('csv', 'features_cellule_VALIDATE.csv')
    percorso_output = os.path.join('csv', 'features_cellule_corretto_da_ML.csv')

    print(f"\n[STEP 2] Creazione dataset di studio da: {percorso_input}")
    try:
        df = pd.read_csv(percorso_input)
    except FileNotFoundError:
        print("[ERRORE] Esegui prima lo step 1!")
        return

    features = ['Area', 'Perimeter', 'Circularity', 'AspectRatio', 'MeanBlue', 'MeanGreen', 'MeanRed']
    df = df.dropna(subset=features + ['GroundTruth_Label'])
    X = df[features].values

    y_semi = []
    mappa_classi = {'GlobuloBianco': 0, 'GlobuloRosso': 1, 'Piastrina': 2, 'Rumore': 3}

    for _, row in df.iterrows():
        label_medico = row['GroundTruth_Label']
        if pd.notna(label_medico) and label_medico in mappa_classi:
            y_semi.append(mappa_classi[label_medico])
        else:
            y_semi.append(-1)  # Cella ignota da indovinare

    print("  -> Pulizia dati, standardizzazione e Label Spreading in corso...")
    imputer = SimpleImputer(strategy='mean')
    X_clean = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    modello_semi = LabelSpreading(kernel='knn', n_neighbors=15, alpha=0.2)
    modello_semi.fit(X_scaled, y_semi)

    # Salvataggio modelli per usi futuri
    joblib.dump(modello_semi, 'modello_label_spreading.pkl')
    joblib.dump(scaler, 'scaler_progetto.pkl')
    joblib.dump(imputer, 'imputer_progetto.pkl')

    y_transduced = modello_semi.transduction_
    nomi_predetti = [list(mappa_classi.keys())[list(mappa_classi.values()).index(val)] for val in y_transduced]
    df['CellType_Predetto_ML'] = nomi_predetti

    df.to_csv(percorso_output, index=False)
    print(f"  ✅ Dataset corretto salvato come: {percorso_output}")


if __name__ == "__main__":
    train_semi_supervised()