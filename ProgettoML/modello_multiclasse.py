import os
import glob
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================================================================
# 0. GESTIONE INTELLIGENTE DEI PERCORSI RELATIVI
# =========================================================================
cartella_script = os.path.dirname(os.path.abspath(__file__))

# Capisce da solo se il file si trova dentro "csv" o nella cartella principale "ProgettoML"
if os.path.basename(cartella_script) == "csv":
    base_dir = os.path.dirname(cartella_script)
    cartella_csv = cartella_script
else:
    base_dir = cartella_script
    cartella_csv = os.path.join(cartella_script, "csv")

# =========================================================================
# 1. CARICAMENTO DEI DATASET VALIDATI (Incrociati col JSON)
# =========================================================================
path_train = os.path.join(cartella_csv, "features_cellule_VALIDATE.csv")
path_test = os.path.join(cartella_csv, "features_cellule_test_VALIDATE.csv")

if not os.path.exists(path_train) or not os.path.exists(path_test):
    raise FileNotFoundError(
        "Mancano i file _VALIDATE.csv! Assicurati di aver lanciato lo script di validazione sia per Train che per Test.")

print("Caricamento dataset Multi-Classe Validato in corso...")
df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

# =========================================================================
# 2. PRE-ELABORAZIONE E MAPPATURA SEMI-SUPERVISIONATA
# =========================================================================
features_colonne = ['BoxX', 'BoxY', 'BoxW', 'BoxH', 'Area', 'Perimeter',
                    'Circularity', 'AspectRatio', 'MeanBlue', 'MeanGreen', 'MeanRed']


def prepara_dataset(df, is_train=True):
    # Rimuoviamo il Rumore: se ne occupa già il modello Binario!
    df = df[df['GroundTruth_Label'] != 'Rumore'].copy()

    # Mappatura per il Label Spreading (-1 = Sconosciuto)
    mappa_classi = {
        'GlobuloRosso': 0,
        'GlobuloBianco': 1,
        'Piastrina': 2,
        'Sconosciuto': -1
    }
    df['Target'] = df['GroundTruth_Label'].map(mappa_classi)
    df = df.dropna(subset=['Target'] + features_colonne)

    # Nel TEST set teniamo solo le ancore certe per la valutazione finale
    if not is_train:
        df = df[df['Target'] != -1]

    return df


df_train = prepara_dataset(df_train, is_train=True)
df_test = prepara_dataset(df_test, is_train=False)

X_train = df_train[features_colonne]
y_train = df_train['Target'].astype(int)

X_test = df_test[features_colonne]
y_test = df_test['Target'].astype(int)

# Normalizzazione matematica
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================================
# 3. FASE 1: LABEL SPREADING (Autocompletamento delle etichette)
# =========================================================================
print("\n[INFO] Avvio Label Spreading sui dati di TRAIN...")
print(f"-> Ancore certe a disposizione: {sum(y_train != -1)}")
print(f"-> Sconosciuti da etichettare: {sum(y_train == -1)}")

modello_ls = LabelSpreading(kernel='knn', alpha=0.2, n_jobs=-1)
modello_ls.fit(X_train_scaled, y_train)

# Etichette "autocompletate" (nessun -1 rimasto)
y_train_pseudo = modello_ls.transduction_

print("[SUCCESSO] Label Spreading completato! Tutti gli Sconosciuti sono stati etichettati.")

# =========================================================================
# 4. FASE 2: ADDESTRAMENTO RANDOM FOREST
# =========================================================================
print("\n[INFO] Addestramento Random Forest Multi-Classe in corso...")
modello_multi = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
modello_multi.fit(X_train_scaled, y_train_pseudo)
print("[SUCCESSO] Modello Random Forest addestrato!")

# =========================================================================
# 5. VALUTAZIONE E METRICHE SUL TEST SET REALE
# =========================================================================
y_pred = modello_multi.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("\n=======================================================")
print(f" ACCURACY VERA SUL TEST SET: {accuracy:.2%}")
print("=======================================================\n")

nomi_classi = ['GlobuloRosso', 'GlobuloBianco', 'Piastrina']

print("Report di Classificazione (sulle annotazioni JSON del test):")
print(classification_report(y_test, y_pred, target_names=nomi_classi))

# =========================================================================
# 6. FEATURE IMPORTANCE
# =========================================================================
importances = modello_multi.feature_importances_
indici = np.argsort(importances)[::-1]

print("\nClassifica di importanza delle feature per distinguere le cellule:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}) {features_colonne[indici[f]]:<15} : {importances[indici[f]]:.4f}")

# =========================================================================
# 7. RISCONTRO VISIVO OPENCV (SOLO VISIONE TOTALE IA)
# =========================================================================
print("\n[INFO] Avvio dell'ispezione visiva OpenCV...")
print("-> Ora vedrai in un'unica finestra la Visione Totale dell'IA.")
print("-> Premi un tasto qualsiasi sulla finestra per passare alla foto successiva.")
print("-> Premi il tasto ESC per uscire dal visualizzatore.")

# Puntiamo dritti alla cartella delle immagini grezze
cartella_raw = os.path.join(base_dir, 'archive', 'test', 'img')

# MAGIA: Ricarichiamo il file di test COMPLETO per avere tutti i box del C++
df_test_totale = pd.read_csv(path_test)
# Scartiamo solo il "Rumore" che viene bloccato dal modello Binario
df_test_totale = df_test_totale[df_test_totale['GroundTruth_Label'] != 'Rumore'].copy()

if not os.path.exists(cartella_raw):
    print(f"\n[ERRORE] La cartella delle immagini {cartella_raw} non esiste!")
else:
    mappa_nomi_classi = {0: 'GlobuloRosso', 1: 'GlobuloBianco', 2: 'Piastrina'}
    colori_ia = {
        'GlobuloRosso': (0, 0, 255),  # Rosso
        'GlobuloBianco': (255, 0, 0),  # Blu
        'Piastrina': (0, 255, 255)  # Giallo
    }

    immagini_test_uniche = df_test_totale['ImageName'].unique()

    for nome_file_img in immagini_test_uniche:
        nome_base = os.path.splitext(nome_file_img)[0]

        file_raw = glob.glob(os.path.join(cartella_raw, nome_base + ".*"))
        if not file_raw:
            print(f"[Warning] Immagine raw non trovata per {nome_base}. Salto.")
            continue

        img_canvas = cv2.imread(file_raw[0])
        if img_canvas is None:
            continue

        cellule_nella_foto = df_test_totale[df_test_totale['ImageName'] == nome_file_img]

        for idx, riga in cellule_nella_foto.iterrows():
            bx = int(riga['BoxX'])
            by = int(riga['BoxY'])
            bw = int(riga['BoxW'])
            bh = int(riga['BoxH'])

            vettore_feature = riga[features_colonne].values.reshape(1, -1)
            vettore_normalizzato = scaler.transform(vettore_feature)

            indice_predetto = modello_multi.predict(vettore_normalizzato)[0]
            classe_predetta = mappa_nomi_classi[indice_predetto]
            classe_reale = riga['GroundTruth_Label']

            colore_bgr = colori_ia[classe_predetta]

            # LOGICA VISIVA INTELLIGENTE
            if classe_reale == 'Sconosciuto':
                # Salvato dal Label Spreading!
                etichetta_testo = f"IA: {classe_predetta} (Ignorato)"
                spessore_box = 1  # Box sottile
            elif classe_predetta == classe_reale:
                # Perfetto accordo
                etichetta_testo = f"{classe_predetta}"
                spessore_box = 2
            else:
                # Vero errore
                etichetta_testo = f"ERR! P:{classe_predetta} (V:{classe_reale})"
                spessore_box = 3
                cv2.putText(img_canvas, "X", (bx + bw - 15, by + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.rectangle(img_canvas, (bx, by), (bx + bw, by + bh), colore_bgr, spessore_box)
            cv2.putText(img_canvas, etichetta_testo, (bx, by - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colore_bgr, 1, cv2.LINE_AA)

        titolo_ia = f"VISIONE TOTALE IA ({nome_base})"

        cv2.imshow(titolo_ia, img_canvas)

        # Posiziono la finestra un po' più al centro dello schermo
        cv2.moveWindow(titolo_ia, 350, 50)

        tasto_premuto = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if tasto_premuto == 27:
            print("[INFO] Ispezione visiva interrotta.")
            break

    print("[INFO] Ispezione terminata correttamente.")