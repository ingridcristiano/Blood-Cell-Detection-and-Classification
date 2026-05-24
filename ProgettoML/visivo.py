import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.semi_supervised import LabelSpreading
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Tenta l'importazione di seaborn per una grafica migliore della matrice
try:
    import seaborn as sns
except ImportError:
    sns = None

# =========================================================================
# LIBRERIA UTILE: CALCOLO IOU
# =========================================================================
def calcola_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


# =========================================================================
# STEP 1: VALIDAZIONE DATI (Incrocio geometrico C++ e JSON Medico)
# =========================================================================
def step1_valida_dati():
    percorso_csv_input = os.path.join('csv', 'features_cellule_train.csv')
    percorso_csv_output = os.path.join('csv', 'features_cellule_VALIDATE.csv')
    cartella_ann = os.path.join('..', 'ProgettoIPA', 'archive', 'train', 'ann')
    soglia_iou = 0.01

    print(f"\n[STEP 1] Caricamento del dataset C++ di ADDESTRAMENTO ({percorso_csv_input})...")
    try:
        df = pd.read_csv(percorso_csv_input, sep=None, engine='python')
        df.columns = df.columns.str.strip()
        df['ImageName'] = df['ImageName'].astype(str).str.strip()
    except Exception as e:
        print(f"ERRORE CRITICO: Impossibile leggere il CSV di input: {e}")
        return False

    df['GroundTruth_Label'] = 'Rumore'
    df['IoU_Score'] = 0.0

    if not os.path.exists(cartella_ann):
        print(f"[ALLARME] La cartella JSON non esiste in: {cartella_ann}")
        return False

    immagini_uniche = df['ImageName'].unique()
    veri_positivi = 0
    falsi_positivi = 0

    for img_name in immagini_uniche:
        json_path = os.path.join(cartella_ann, img_name + ".json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            dati_ann = json.load(f)

        righe_immagine = df[df['ImageName'] == img_name].index

        for idx in righe_immagine:
            cpp_box = [
                df.at[idx, 'BoxX'], df.at[idx, 'BoxY'],
                df.at[idx, 'BoxX'] + df.at[idx, 'BoxW'],
                df.at[idx, 'BoxY'] + df.at[idx, 'BoxH']
            ]

            miglior_iou = 0.0
            miglior_label_medico = 'Rumore'

            for obj in dati_ann.get('objects', []):
                pts = obj['points']['exterior']
                medico_box = [pts[0][0], pts[0][1], pts[1][0], pts[1][1]]

                iou = calcola_iou(cpp_box, medico_box)
                if iou > miglior_iou:
                    miglior_iou = iou
                    cls = obj['classTitle']
                    if cls == 'WBC': miglior_label_medico = 'GlobuloBianco'
                    elif cls == 'RBC': miglior_label_medico = 'GlobuloRosso'
                    elif cls == 'Platelets': miglior_label_medico = 'Piastrina'
                    else: miglior_label_medico = cls

            df.at[idx, 'IoU_Score'] = miglior_iou
            if miglior_iou >= soglia_iou:
                df.at[idx, 'GroundTruth_Label'] = miglior_label_medico
                veri_positivi += 1
            else:
                falsi_positivi += 1

    print(f"  ✅ Cellule confermate dal medico: {veri_positivi}")
    print(f"  ❌ Macchie scartate come rumore: {falsi_positivi}")
    df.to_csv(percorso_csv_output, index=False)
    print(f"[OK] File salvato: {percorso_csv_output}")
    return True


# =========================================================================
# STEP 2: ADDESTRAMENTO SEMI-SUPERVISIONATO (Label Spreading) - FIXATO
# =========================================================================
def step2_train_semi_supervised():
    percorso_input = os.path.join('csv', 'features_cellule_VALIDATE.csv')
    percorso_output = os.path.join('csv', 'features_cellule_corretto_da_ML.csv')

    print(f"\n[STEP 2] Avvio Label Spreading da '{percorso_input}'...")
    if not os.path.exists(percorso_input):
        print("[ERRORE] Esegui prima lo Step 1!")
        return False

    df = pd.read_csv(percorso_input)

    # --- FIX ANOMALIA: Pulizia da righe vuote o incomplete ---
    features = ['Area', 'Perimeter', 'Circularity', 'AspectRatio', 'MeanBlue', 'MeanGreen', 'MeanRed']
    df = df.dropna(subset=features + ['GroundTruth_Label', 'CellType'])
    # ---------------------------------------------------------

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
                y_semi.append(-1)  # Sconosciuto da propagare
            elif label_cpp == 'GlobuloBianco':
                y_semi.append(mappa_classi['Rumore'])

    y_semi = np.array(y_semi)

    # Verifica di sicurezza prima del fit
    if len(X) != len(y_semi):
        print(f"[ERRORE INTERNO] Lunghezze ancora sbilanciate! X: {len(X)}, Y: {len(y_semi)}")
        return False

    imputer = SimpleImputer(strategy='mean')
    X_clean = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    modello_semi = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.2)
    modello_semi.fit(X_scaled, y_semi)

    joblib.dump(modello_semi, 'modello_label_spreading.pkl')
    joblib.dump(scaler, 'scaler_progetto.pkl')
    joblib.dump(imputer, 'imputer_progetto.pkl')

    y_transduced = modello_semi.transduction_
    nomi_predetti = [list(mappa_classi.keys())[list(mappa_classi.values()).index(val)] for val in y_transduced]
    df['CellType_Predetto_ML'] = nomi_predetti

    df.to_csv(percorso_output, index=False)
    print(f"[OK] Dataset Gold Standard salvato in: {percorso_output}")
    return True


# =========================================================================
# STEP 3: VALUTAZIONE E MATRICE DI CONFUSIONE (Random Forest sul Test)
# =========================================================================
def step3_valutazione_sulle_nuove_immagini():
    file_train = os.path.join('csv', 'features_cellule_corretto_da_ML.csv')
    file_test = os.path.join('csv', 'features_cellule_test.csv')

    print(f"\n[STEP 3] Valutazione Finale tramite Random Forest...")
    if not os.path.exists(file_train) or not os.path.exists(file_test):
        print(f"[ERRORE] File mancanti!\n- {file_train}\n- {file_test}")
        return

    df_train = pd.read_csv(file_train)
    df_test = pd.read_csv(file_test)

    # --- CORREZIONE DEL BUG TYPEERROR (PULIZIA DA NAN E COERCISIONE STRINGHE) ---
    df_train = df_train.dropna(subset=['CellType_Predetto_ML'])
    df_test = df_test.dropna(subset=['CellType'])
    df_train['CellType_Predetto_ML'] = df_train['CellType_Predetto_ML'].astype(str).str.strip()
    df_test['CellType'] = df_test['CellType'].astype(str).str.strip()
    # ----------------------------------------------------------------------------

    features = ['Area', 'Perimeter', 'Circularity', 'AspectRatio', 'MeanBlue', 'MeanGreen', 'MeanRed']

    X_train = df_train[features].values
    y_train = df_train['CellType_Predetto_ML'].values
    X_test = df_test[features].values
    y_test_reale = df_test['CellType'].values

    imputer = SimpleImputer(strategy='mean')
    X_train_clean = imputer.fit_transform(X_train)
    X_test_clean = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)

    y_pred_nuove = rf_model.predict(X_test_scaled)

    print("\n======================================================")
    print(" 🏆 PAGELLA DELLA RANDOM FOREST (SULLE IMMAGINI NUOVE) ")
    print("======================================================")
    print(classification_report(y_test_reale, y_pred_nuove, zero_division=0))

    print("\n[IMPORTANZA DELLE FEATURE]")
    importances = rf_model.feature_importances_
    for f, imp in zip(features, importances):
        print(f" - {f}: {imp * 100:.1f}%")

    # --- GENERAZIONE GENERALE E GRAFICA DELLA MATRICE DI CONFUSIONE ---
    print("\n[INFO] Generazione della Matrice di Confusione Visiva...")
    classi_uniche = sorted(list(set(y_test_reale) | set(y_pred_nuove)))
    cm = confusion_matrix(y_test_reale, y_pred_nuove, labels=classi_uniche)

    plt.figure(figsize=(8, 6))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classi_uniche, yticklabels=classi_uniche)
    else:
        # Fallback nel caso in cui seaborn non sia ancora installato nel terminale
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matrice di Confusione (Senza Seaborn)')
        plt.colorbar()
        tick_marks = np.arange(len(classi_uniche))
        plt.xticks(tick_marks, classi_uniche, rotation=45)
        plt.yticks(tick_marks, classi_uniche)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                         color="white" if cm[i, j] > cm.max()/2. else "black")

    plt.title('Matrice di Confusione - Validazione Algoritmo')
    plt.xlabel('Predizioni dell\'Intelligenza Artificiale')
    plt.ylabel('Verità Rilevata (C++)')
    plt.tight_layout()
    plt.show()


# =========================================================================
# ESECUZIONE DELLA PIPELINE COMPLETA
# =========================================================================
if __name__ == "__main__":
    if step1_valida_dati():
        if step2_train_semi_supervised():
            step3_valutazione_sulle_nuove_immagini()