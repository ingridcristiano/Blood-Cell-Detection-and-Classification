import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- NUOVE IMPORTAZIONI PER GRAFICO E PERCENTUALE ---
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# ----------------------------------------------------

def valutazione_sulle_nuove_immagini():
    # --- PERCORSI RELATIVI ---
    # Punta direttamente alla sottocartella 'csv'
    file_train = os.path.join('csv', 'features_cellule_corretto_da_ML.csv')
    file_test = os.path.join('csv', 'features_cellule_test.csv')

    # --- CONTROLLO FILE ---
    if not os.path.exists(file_train) or not os.path.exists(file_test):
        print(f"[ERRORE] Manca uno dei file!\n- {file_train}\n- {file_test}")
        print("Assicurati di aver eseguito il C++ per il test e lo script precedente per il train.")
        return

    print("1. Caricamento del materiale di STUDIO e del TEST...")
    df_train = pd.read_csv(file_train)
    df_test = pd.read_csv(file_test)

    # --- LA SOLUZIONE ALL'ERRORE ---
    # Rimuoviamo eventuali righe completamente vuote
    df_train = df_train.dropna(subset=['CellType_Predetto_ML'])
    df_test = df_test.dropna(subset=['CellType'])

    # Forziamo le colonne a essere viste come stringhe di testo purissimo
    df_train['CellType_Predetto_ML'] = df_train['CellType_Predetto_ML'].astype(str)
    df_test['CellType'] = df_test['CellType'].astype(str)
    # -------------------------------

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

    # =========================================================================
    # --- NUOVA SEZIONE AGGIUNTA: CALCOLO EFFICACIA E MATRICE DI CONFUSIONE ---
    # =========================================================================

    # Calcoliamo e stampiamo la percentuale
    accuratezza = accuracy_score(y_test_reale, y_pred_nuove)
    print("\n======================================================")
    print(f" 🎯 PERCENTUALE FINALE EFFICACIA: {accuratezza * 100:.2f}%")
    print("======================================================")

    print("\n5. Generazione della Matrice di Confusione...")

    # Creiamo la matrice incrociando i dati reali con le previsioni dell'IA
    cm = confusion_matrix(y_test_reale, y_pred_nuove, labels=rf_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)

    # Disegniamo la grafica
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)

    plt.title("Matrice di Confusione - Valutazione IA")
    plt.tight_layout()

    # Apriamo la finestra con il grafico a colori
    plt.show()


if __name__ == "__main__":
    valutazione_sulle_nuove_immagini()