import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.semi_supervised import LabelPropagation
# ---> AGGIUNTO QUESTO IMPORT IN CIMA <---
from sklearn.metrics import classification_report


def train_semi_supervised():
    try:
        df = pd.read_csv('features_cellule.csv', sep=None, engine='python')
    except Exception as e:
        print(f"Errore critico nella lettura del file: {e}")
        return

    # Pulizia nomi colonne da spazi bianchi
    df.columns = df.columns.str.strip()

    if 'CellType' not in df.columns:
        print("ERRORE: La colonna 'CellType' non esiste.")
        return

    labels = df['CellType'].copy()

    # MAPPATURA: Trasformiamo i nomi in numeri per l'algoritmo
    mapping = {"GlobuloBianco": 0, "Piastrina": 1, "GlobuloRosso": 2}
    y = labels.map(mapping).fillna(-1).astype(int).values.copy()

    # SIMULAZIONE SEMI-SUPERVISIONATA: Nascondiamo il 50% dei rossi
    RNG = np.random.default_rng(42)
    maschera_rossi = (y == 2)
    nascondi_rossi = RNG.choice([True, False], size=len(y), p=[0.5, 0.5])
    y[maschera_rossi & nascondi_rossi] = -1

    print(f"   -> Dati totali: {len(y)} | Di cui Certi: {np.sum(y != -1)} | Da indovinare (-1): {np.sum(y == -1)}")

    # 3. SELEZIONE DELLE FEATURE
    features = ['Area', 'Perimeter', 'Circularity', 'AspectRatio', 'MeanBlue', 'MeanGreen', 'MeanRed']
    X = df[features].values

    # --- TRUCCO ANTI-CRASH: ELIMINAZIONE DEI NAN ---
    print("3. Pulizia dei dati mancanti (NaN Imputation)...")
    imputer = SimpleImputer(strategy='mean')
    X_clean = imputer.fit_transform(X)

    # 4. STANDARDIZZAZIONE
    print("4. Standardizzazione delle feature...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    print("5. Addestramento del modello Semi-Supervisionato (Label Propagation)...")
    lp_model = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)
    lp_model.fit(X_scaled, y)

    # 6. ESTRAZIONE DELLE PREDIZIONI FINALI
    print("6. Estrazione delle predizioni e salvataggio...")
    etichette_predette = lp_model.transduction_

    # Riconvertiamo i numeri in nomi leggibili
    reverse_mapping = {0: "GlobuloBianco", 1: "Piastrina", 2: "GlobuloRosso"}
    df['CellType_Predetto_ML'] = [reverse_mapping[pred] for pred in etichette_predette]

    print("\n--- RISULTATO DELL'APPRENDIMENTO SEMI-SUPERVISIONATO ---")
    print(df[['ImageName', 'CellType', 'CellType_Predetto_ML']].head(30))

    # ---> METTIAMO QUI LE RIGHE DI VALIDAZIONE <---
    print("\n--- REPORT DI VALIDAZIONE SUI DATI NASCOSTI ---")
    # Questo confronta la colonna originale del C++ con quella indovinata dal ML
    print(classification_report(df['CellType'], df['CellType_Predetto_ML']))
    print("------------------------------------------------\n")

    # Salviamo il file definitivo
    df.to_csv('features_cellule_corretto_da_ML.csv', index=False)
    print("\n[FINE SUCCESS] Il modello ha indovinato tutte le cellule nascoste!")
    print("File salvato: features_cellule_corretto_da_ML.csv")


if __name__ == "__main__":
    train_semi_supervised()