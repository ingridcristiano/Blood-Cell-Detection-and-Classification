import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def valutazione_sulle_nuove_immagini():
    file_train = 'features_cellule_corretto_da_ML.csv'
    file_test = 'features_cellule_test.csv'

    # --- CONTROLLO FILE ---
    if not os.path.exists(file_train) or not os.path.exists(file_test):
        print(f"[ERRORE] Manca uno dei file!\n- {file_train}\n- {file_test}")
        return

    print("1. Caricamento del materiale di STUDIO e del TEST...")
    df_train = pd.read_csv(file_train)
    df_test = pd.read_csv(file_test)

    features = ['Area', 'Perimeter', 'Circularity', 'AspectRatio', 'MeanBlue', 'MeanGreen', 'MeanRed']

    # Dati per studiare
    X_train = df_train[features].values
    y_train = df_train['CellType_Predetto_ML'].values

    # Dati per l'esame (le nuove immagini)
    X_test = df_test[features].values

    # Come "risposte corrette" del test, usiamo quello che aveva trovato il C++
    # (Attenzione: il C non conosce la classe 'Rumore', quindi ci saranno discrepanze)
    y_test_reale = df_test['CellType'].values

    print("2. Pulizia e allineamento dati...")
    imputer = SimpleImputer(strategy='mean')
    # Il fit lo facciamo SOLO sui dati di train!
    X_train_clean = imputer.fit_transform(X_train)
    X_test_clean = imputer.transform(X_test)

    scaler = StandardScaler()
    # Il fit lo facciamo SOLO sui dati di train!
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    print("3. Addestramento della Random Forest su TUTTO il vecchio dataset...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)

    print("4. Esecuzione dell'esame sulle NUOVE immagini...")
    y_pred_nuove = rf_model.predict(X_test_scaled)

    print("\n======================================================")
    print(" 🏆 PAGELLA DELLA RANDOM FOREST (SULLE IMMAGINI NUOVE) ")
    print("======================================================")
    # zero_division=0 evita avvisi se una classe scompare
    print(classification_report(y_test_reale, y_pred_nuove, zero_division=0))

    print("\n[IMPORTANZA DELLE FEATURE]")
    importances = rf_model.feature_importances_
    for f, imp in zip(features, importances):
        print(f" - {f}: {imp * 100:.1f}%")


if __name__ == "__main__":
    valutazione_sulle_nuove_immagini()