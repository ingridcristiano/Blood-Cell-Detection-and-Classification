import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def valutazione_sulle_nuove_immagini():
    # IL MATERIALE DI STUDIO (Generato dallo Step 2)
    file_train = os.path.join('csv', 'features_cellule_corretto_da_ML.csv')
    # LA VERIFICA CON LE SOLUZIONI ESATTE (Generato dallo Step 1)
    file_test = os.path.join('csv', 'features_cellule_test_VALIDATE.csv')

    print(f"\n[STEP 3] Esame Finale della Random Forest...")
    if not os.path.exists(file_train) or not os.path.exists(file_test):
        print(f"[ERRORE] File mancanti! Assicurati di aver eseguito lo Step 1 e lo Step 2.")
        return

    df_train = pd.read_csv(file_train).dropna(subset=['CellType_Predetto_ML'])
    df_test = pd.read_csv(file_test).dropna(subset=['GroundTruth_Label'])

    features = ['Area', 'Perimeter', 'Circularity', 'AspectRatio', 'MeanBlue', 'MeanGreen', 'MeanRed']

    # Prepariamo X e Y per lo studio
    X_train = df_train[features].values
    y_train = df_train['CellType_Predetto_ML'].astype(str).values

    # Prepariamo X per l'esame e Y_reale per la correzione
    X_test = df_test[features].values
    y_test_reale = df_test['GroundTruth_Label'].astype(str).values

    print("  -> Pulizia e addestramento del modello sul Train...")
    imputer = SimpleImputer(strategy='mean')
    X_train_clean = imputer.fit_transform(X_train)
    X_test_clean = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)

    print("  -> Esecuzione test sulle nuove immagini...")
    y_pred_nuove = rf_model.predict(X_test_scaled)

    # --- CALCOLO RISULTATI ---
    accuratezza = accuracy_score(y_test_reale, y_pred_nuove)
    print("\n======================================================")
    print(f" 🎯 PERCENTUALE FINALE EFFICACIA: {accuratezza * 100:.2f}%")
    print("======================================================")
    print(classification_report(y_test_reale, y_pred_nuove, zero_division=0))

    # --- MATRICE DI CONFUSIONE A COLORI ---
    classi_ordinate = sorted(list(set(y_test_reale) | set(y_pred_nuove)))
    cm = confusion_matrix(y_test_reale, y_pred_nuove, labels=classi_ordinate)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classi_ordinate)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Matrice di Confusione - Valutazione IA Completa")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    valutazione_sulle_nuove_immagini()