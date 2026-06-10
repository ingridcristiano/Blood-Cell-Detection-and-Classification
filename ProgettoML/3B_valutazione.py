import os
import json
import glob
import cv2
import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, \
    precision_recall_curve
from xgboost import XGBClassifier

warnings.filterwarnings('ignore', category=UserWarning)

# =========================================================================
# 1. CONFIGURAZIONE PERCORSI E FEATURE
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "modelli_salvati")
if not os.path.exists(os.path.join(MODELS_DIR, 'xgboost_model.pkl')):
    MODELS_DIR = BASE_DIR

FEATURES = [
    'Area', 'Perimeter', 'Circularity', 'AspectRatio', 'Eccentricity', 'Extent',
    'MeanBlue', 'MeanGreen', 'MeanRed', 'MeanValue', 'MinValue', 'MaxValue',
    'MeanSaturation', 'MinSat', 'MaxSat', 'TextureValue', 'TextureSat', 'TextureLaplacian',
    'Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7'
]

mappa_inversa = {0: 'GlobuloBianco', 1: 'GlobuloRosso', 2: 'Piastrina', 3: 'Rumore'}
tutte_le_classi = ['GlobuloBianco', 'GlobuloRosso', 'Piastrina', 'Rumore']

colori_classi = {
    'GlobuloRosso': (0, 0, 255),
    'GlobuloBianco': (255, 0, 0),
    'Piastrina': (0, 255, 255),
    'Rumore': (128, 128, 128)
}

testi_brevi = {
    'GlobuloRosso': 'Rosso',
    'GlobuloBianco': 'Bianco',
    'Piastrina': 'Piastrina',
    'Rumore': 'Rumore'
}


# =========================================================================
# 2. FUNZIONI DI VALIDAZIONE
# =========================================================================
def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    return interArea / float(
        ((boxA[2] - boxA[0]) * (boxA[3] - boxA[1])) + ((boxB[2] - boxB[0]) * (boxB[3] - boxB[1])) - interArea)


def valida_dataset_test_completo():
    print("1. Lettura dei file JSON e assegnazione univoca (Anti-Doppioni)...")

    percorso_csv = os.path.join(BASE_DIR, 'features_cellule_test.csv')
    if not os.path.exists(percorso_csv):
        percorso_csv = os.path.join(BASE_DIR, 'csv', 'features_cellule_test.csv')

    cartella_json = os.path.join(BASE_DIR, 'archive', 'test', 'ann')

    df = pd.read_csv(percorso_csv)
    df.columns = df.columns.str.strip()

    # --- LOGICA APPLICATA: Inizializziamo a NaN per nascondere il rumore non annotato ---
    df['GroundTruth_Label'] = pd.Series(np.nan, dtype="object")

    for img_name in df['ImageName'].unique():
        nome_base = os.path.splitext(img_name)[0]
        file_trovati = glob.glob(os.path.join(cartella_json, nome_base + "*.json"))
        if not file_trovati: continue

        with open(file_trovati[0], 'r') as f:
            dati_ann = json.load(f)

        # Indici di tutti i box C++ trovati in questa specifica immagine
        indici_immagine = df[df['ImageName'] == img_name].index

        # Dizionario di appoggio per evitare di ricalcolare i box in continuazione
        box_cpp_dict = {
            idx: [df.at[idx, 'BoxX'], df.at[idx, 'BoxY'], df.at[idx, 'BoxX'] + df.at[idx, 'BoxW'],
                  df.at[idx, 'BoxY'] + df.at[idx, 'BoxH']]
            for idx in indici_immagine
        }

        # Cicliamo sulle Ground Truth (le cellule VERE annotate dal medico)
        for obj in dati_ann.get('objects', []):
            pts = obj['points']['exterior']
            gt_box = [pts[0][0], pts[0][1], pts[1][0], pts[1][1]]

            cls = obj['classTitle'].strip().upper()
            if cls == 'WBC':
                cls_name = 'GlobuloBianco'
            elif cls == 'RBC':
                cls_name = 'GlobuloRosso'
            elif cls in ['PLATELETS', 'PLATELET']:
                cls_name = 'Piastrina'
            else:
                cls_name = cls

            soglia = 0.1 if cls_name == 'Piastrina' else 0.35

            miglior_iou = 0.0
            miglior_idx = None

            # Cerchiamo il SINGOLO rettangolo C++ che copre meglio questa specifica Ground Truth
            for idx, cpp_box in box_cpp_dict.items():
                iou = compute_iou(cpp_box, gt_box)

                # Se supera la soglia ed è il migliore visto finora per QUESTA cellula
                if iou >= soglia and iou > miglior_iou:
                    miglior_iou = iou
                    miglior_idx = idx

            # Promuove solo il rettangolo vincente
            if miglior_idx is not None:
                df.at[miglior_idx, 'GroundTruth_Label'] = cls_name

    return df


# =========================================================================
# 3. ESECUZIONE E GRAFICI
# =========================================================================
if __name__ == "__main__":
    df_test = valida_dataset_test_completo()

    print("2. Caricamento del cervello IA (.pkl)...")
    imputer = joblib.load(os.path.join(MODELS_DIR, 'imputer_progetto.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler_progetto.pkl'))
    rf = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))

    print("3. Esecuzione predizioni...")
    cols_to_use = [col for col in FEATURES if col in df_test.columns]
    X_test_scaled = scaler.transform(imputer.transform(df_test[cols_to_use].values))

    # Predizione secca
    df_test['Predicted_Label'] = [mappa_inversa[val] for val in rf.predict(X_test_scaled)]

    # --- LOGICA APPLICATA: MASCHERA per valutare solo le cellule etichettate! ---
    mask_valutazione = df_test['GroundTruth_Label'].notna()
    df_valutazione = df_test[mask_valutazione]
    y_true = df_valutazione['GroundTruth_Label']
    y_pred = df_valutazione['Predicted_Label']

    X_valutazione_scaled = X_test_scaled[mask_valutazione]
    y_proba = rf.predict_proba(X_valutazione_scaled)

    if len(y_true) > 0:
        accuratezza = accuracy_score(y_true, y_pred)
        print(f"\n🎯 PERCENTUALE EFFICACIA SULLE CELLULE REALI: {accuratezza * 100:.2f}%")

        # Report filtrato senza Rumore
        classi_mediche = ['GlobuloBianco', 'GlobuloRosso', 'Piastrina']
        print("\n📋 REPORT DIAGNOSTICO DETTAGLIATO:")
        print(classification_report(y_true, y_pred, labels=classi_mediche, zero_division=0))

        # Recupera l'ordine corretto per la binarizzazione
        classi_str = [mappa_inversa[c] for c in rf.classes_]
        y_true_bin = label_binarize(y_true, classes=classi_str)

        # --- CALCOLO AVERAGE PRECISION (AP) - AUC curva P-R con soglia confidence [0.5, 0.9] ---
        print("\n📊 CALCOLO AVERAGE PRECISION (AP) PER CLASSE MEDICA (IoU 0.5-0.9):")
        ap_scores = {}
        for i, nome_classe in enumerate(classi_str):
            if nome_classe == 'Rumore':
                continue
            prec, rec, thr = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            mask = (thr >= 0.5) & (thr <= 0.9)
            if mask.sum() >= 2:
                p = prec[:-1][mask]
                r = rec[:-1][mask]
                idx = np.argsort(r)
                ap = float(np.trapz(p[idx], r[idx]))
            else:
                ap = 0.0
            ap_scores[nome_classe] = ap
            print(f"   - AP {nome_classe.ljust(15)}: {ap:.4f}")

        if ap_scores:
            map_score = sum(ap_scores.values()) / len(ap_scores)
            print(f"   > mAP (Solo Cellule): {map_score:.4f}")

        # --- GRAFICO 1: MATRICE DI CONFUSIONE PULITA ---
        print("\n4. Generazione Matrice di Confusione (chiudi la finestra per procedere)...")
        cm = confusion_matrix(y_true, y_pred, labels=classi_mediche)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classi_mediche)

        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
        plt.title(f"Matrice di Confusione Cellulare - Acc: {accuratezza * 100:.2f}%")
        plt.tight_layout()
        plt.show()

        # --- GRAFICO 2: PRECISION-RECALL CURVE PULITO ---
        print("5. Generazione Curve Precision-Recall...")
        colori_pr = {'GlobuloRosso': 'red', 'GlobuloBianco': 'blue', 'Piastrina': 'orange'}

        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        for i, nome_classe in enumerate(classi_str):
            if nome_classe == 'Rumore':
                continue
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            ap = ap_scores[nome_classe]
            ax_pr.plot(recall, precision, lw=2, color=colori_pr.get(nome_classe, 'black'),
                       label=f'{nome_classe} (AP = {ap:.4f})')
        plt.legend()
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("Curve Precision-Recall (Solo Cellule Mediche)")
        plt.show()

    # =========================================================================
    # 6. VISUALIZZATORE OPENCV
    # =========================================================================
    immagini_da_mostrare = df_test['ImageName'].unique()
    print(f"\n6. AVVIO VISUALIZZATORE OPENCV...")

    TEST_IMG_DIR = os.path.join(BASE_DIR, "archive", "test", "img")
    TEST_ANN_DIR = os.path.join(BASE_DIR, "archive", "test", "ann")

    for img_name in immagini_da_mostrare:
        nome_base = os.path.splitext(img_name)[0]
        file_raw = glob.glob(os.path.join(TEST_IMG_DIR, nome_base + ".*"))
        if not file_raw: continue

        img_gt = cv2.imread(file_raw[0])
        img_ia = img_gt.copy()

        # --- FINESTRA 1: MEDICO ---
        file_ann = glob.glob(os.path.join(TEST_ANN_DIR, nome_base + "*.json"))
        if file_ann:
            with open(file_ann[0], 'r') as f:
                dati_ann = json.load(f)
            for obj in dati_ann.get('objects', []):
                pts = obj['points']['exterior']
                cls_raw = obj['classTitle'].strip().upper()

                if cls_raw == 'WBC':
                    cls_name = 'GlobuloBianco'
                elif cls_raw == 'RBC':
                    cls_name = 'GlobuloRosso'
                elif cls_raw in ['PLATELETS', 'PLATELET']:
                    cls_name = 'Piastrina'
                else:
                    cls_name = cls_raw

                colore_medico = colori_classi.get(cls_name, (0, 255, 0))
                testo_m = testi_brevi.get(cls_name, cls_name)

                cv2.rectangle(img_gt, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), colore_medico, 2)
                cv2.putText(img_gt, testo_m, (pts[0][0], pts[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colore_medico,
                            1)

        # --- FINESTRA 2: IA ---
        img_data = df_test[df_test['ImageName'] == img_name]

        for _, row in img_data.iterrows():
            bx, by, bw, bh = int(row['BoxX']), int(row['BoxY']), int(row['BoxW']), int(row['BoxH'])
            vera_label = row.get('GroundTruth_Label', np.nan)
            pred_label = row['Predicted_Label']

            # --- LOGICA APPLICATA: Saltiamo la visualizzazione del rumore predetto ---
            if pred_label == 'Rumore':
                continue

            colore_ia = colori_classi.get(pred_label, (255, 255, 255))
            spessore_box = 2 if pred_label == vera_label else 1

            testo_pred = testi_brevi.get(pred_label, pred_label)
            testo_vero = "NA" if pd.isna(vera_label) else testi_brevi.get(vera_label, vera_label)

            testo_finale = f"IA:{testo_pred} | V:{testo_vero}"

            cv2.rectangle(img_ia, (bx, by), (bx + bw, by + bh), colore_ia, spessore_box)
            cv2.putText(img_ia, testo_finale, (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colore_ia, 1, cv2.LINE_AA)

        cv2.imshow(f"1 - JSON Medico", img_gt)
        cv2.imshow(f"2 - Visione IA", img_ia)
        cv2.moveWindow("1 - JSON Medico", 50, 50)
        cv2.moveWindow("2 - Visione IA", 700, 50)

        if cv2.waitKey(0) & 0xFF == 27: break

    cv2.destroyAllWindows()
    print("\n[FINE] Visualizzazione completata!")