import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.semi_supervised import LabelSpreading
import joblib


def train_semi_supervised():
    print("1. Caricamento del dataset TRAIN validato...")
    # Legge il file creato dallo step precedente
    df = pd.read_csv('features_cellule_VALIDATE.csv')

    features = ['Area', 'Perimeter', 'Circularity', 'AspectRatio', 'MeanBlue', 'MeanGreen', 'MeanRed']
    X = df[features].values

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
                y_semi.append(-1)  # Sconosciuto da correggere
            elif label_cpp == 'GlobuloBianco':
                y_semi.append(mappa_classi['Rumore'])

    y_semi = np.array(y_semi)

    print("2. Pulizia dei dati mancanti e Standardizzazione...")
    imputer = SimpleImputer(strategy='mean')
    X_clean = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    print("3. Avvio propagazione delle etichette (Label Spreading)...")
    modello_semi = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.2)
    modello_semi.fit(X_scaled, y_semi)

    print("4. Salvataggio modelli e creazione del file GOLD STANDARD...")
    # Salva l'intelligenza per il file di test futuro
    joblib.dump(modello_semi, 'modello_label_spreading.pkl')
    joblib.dump(scaler, 'scaler_progetto.pkl')
    joblib.dump(imputer, 'imputer_progetto.pkl')

    # ECCO IL PUNTO DOVE NASCE IL FILE CHE TI MANCA!
    y_transduced = modello_semi.transduction_
    nomi_predetti = [list(mappa_classi.keys())[list(mappa_classi.values()).index(val)] for val in y_transduced]
    df['CellType_Predetto_ML'] = nomi_predetti

    # Lo salva fisicamente sul tuo computer
    df.to_csv('features_cellule_corretto_da_ML.csv', index=False)

    print("\n[FINE] Il dataset corretto dall'IA è stato salvato! ORA ESISTE!")


if __name__ == "__main__":
    train_semi_supervised()