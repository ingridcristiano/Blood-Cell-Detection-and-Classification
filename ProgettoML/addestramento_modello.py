import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # <--- NOVITÀ: Il chirurgo dei dati mancanti!
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report


def train_semi_supervised():
    print("1. Caricamento del dataset validato...")
    df = pd.read_csv('features_cellule_VALIDATE.csv')

    features = ['Area', 'Perimeter', 'Circularity', 'AspectRatio', 'MeanBlue', 'MeanGreen', 'MeanRed']
    X = df[features].values

    # -------------------------------------------------------------------
    # CREAZIONE DELLE ETICHETTE IBRIDE
    # -------------------------------------------------------------------
    y_semi = []
    mappa_classi = {'GlobuloBianco': 0, 'GlobuloRosso': 1, 'Piastrina': 2, 'Rumore': 3}

    for index, row in df.iterrows():
        label_medico = row['GroundTruth_Label']
        label_cpp = row['CellType']

        if label_medico in ['GlobuloBianco', 'GlobuloRosso', 'Piastrina']:
            y_semi.append(mappa_classi[label_medico])
        else:
            if label_cpp == 'Piastrina':
                y_semi.append(mappa_classi['Rumore'])
            elif label_cpp == 'GlobuloRosso':
                y_semi.append(-1)  # Lo Sconosciuto
            elif label_cpp == 'GlobuloBianco':
                y_semi.append(mappa_classi['Rumore'])

    y_semi = np.array(y_semi)

    certi = len(y_semi[y_semi != -1])
    sconosciuti = len(y_semi[y_semi == -1])
    print(f" -> Dati Certi (Ancore + Rumore sicuro): {certi}")
    print(f" -> Globuli Rossi Sconosciuti (-1) lasciati all'IA: {sconosciuti}")

    # ===================================================================
    # 2. PULIZIA DEI DATI MANCANTI (Imputation) E STANDARDIZZAZIONE
    # ===================================================================
    print("\n2. Pulizia dei dati mancanti (NaN) e Standardizzazione...")

    # Questo strumento cerca i NaN e li sostituisce con la MEDIA di quella colonna
    imputer = SimpleImputer(strategy='mean')
    X_clean = imputer.fit_transform(X)

    # Ora passiamo i dati "puliti" allo scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # ===================================================================
    # 3. ADDESTRAMENTO SEMI-SUPERVISIONATO
    # ===================================================================
    print("3. Avvio propagazione virale delle etichette (Label Spreading)...")
    modello_semi = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.2)
    modello_semi.fit(X_scaled, y_semi)

    # 4. Estrazione dei risultati
    y_transduced = modello_semi.transduction_

    mask_sconosciuti = (y_semi == -1)
    predizioni_sconosciuti = y_transduced[mask_sconosciuti]

    print("\n======================================================")
    print(" 🦠 RISULTATI DELLA PROPAGAZIONE SUI ROSSI NON ANNOTATI ")
    print("======================================================")
    unique, counts = np.unique(predizioni_sconosciuti, return_counts=True)
    for u, c in zip(unique, counts):
        nome_classe = list(mappa_classi.keys())[list(mappa_classi.values()).index(u)]
        print(f" - Rossi sospetti trasformati in [{nome_classe}]: {c}")

    nomi_predetti = [list(mappa_classi.keys())[list(mappa_classi.values()).index(val)] for val in y_transduced]
    df['CellType_Predetto_ML'] = nomi_predetti
    df.to_csv('features_cellule_corretto_da_ML.csv', index=False)

    print(
        "\n[FINE] Il dataset corretto dall'Intelligenza Artificiale è stato salvato in 'features_cellule_corretto_da_ML.csv'!")


if __name__ == "__main__":
    train_semi_supervised()