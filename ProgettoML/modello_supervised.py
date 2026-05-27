import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def esperimento_baseline_supervised():
    print("======================================================")
    print(" 🧪 PROVA DEL 9: ADDESTRAMENTO 100% SUPERVISIONATO")
    print("======================================================\n")

    percorso_train = os.path.join('csv', 'features_cellule_VALIDATE.csv')
    percorso_test = os.path.join('csv', 'features_cellule_test.csv')

    if not os.path.exists(percorso_train) or not os.path.exists(percorso_test):
        print(f"[ERRORE] Mancano i file in formato CSV! Controlla i percorsi.")
        return

    print("1. Caricamento dei dataset...")
    df_train = pd.read_csv(percorso_train)
    df_test = pd.read_csv(percorso_test)

    features = ['Area', 'Perimeter', 'Circularity', 'AspectRatio', 'MeanBlue', 'MeanGreen', 'MeanRed']

    # Rimuoviamo righe corrotte se ci sono
    df_train = df_train.dropna(subset=features + ['GroundTruth_Label'])
    df_test = df_test.dropna(subset=features + ['CellType'])

    # -------------------------------------------------------------------
    # IL CUORE DELL'ESPERIMENTO: Niente Label Spreading!
    # Usiamo direttamente la verità (imperfetta) del medico.
    # -------------------------------------------------------------------
    X_train = df_train[features].values
    y_train = df_train['GroundTruth_Label'].values  # <-- Ci fidiamo ciecamente!

    X_test = df_test[features].values
    y_test_reale = df_test['CellType'].values

    print("2. Pulizia e Standardizzazione...")
    imputer = SimpleImputer(strategy='mean')
    X_train_clean = imputer.fit_transform(X_train)
    X_test_clean = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    print("3. Addestramento ignorante (Random Forest)...")
    # Imparerà gli errori del medico come se fossero dogmi assoluti
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)

    print("4. Esecuzione del Test cieco...\n")
    y_pred = rf_model.predict(X_test_scaled)

    print("======================================================")
    print(" 📉 PAGELLA DEL MODELLO 'STUPIDO' (SENZA LABEL SPREADING) ")
    print("======================================================")
    print(classification_report(y_test_reale, y_pred, zero_division=0))


if __name__ == "__main__":
    esperimento_baseline_supervised()