"""
=============================================================================
CLASSIFICATORE MULTICLASSE DEFINITIVO + VISUALIZZATORE OPENCV
=============================================================================
"""

import os
import json
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score, accuracy_score
)
from sklearn.preprocessing import label_binarize, StandardScaler
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURAZIONE PERCORSI (AUTOMATICA E SICURA)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Se il file è dentro una cartella (es. SEMI-SUPERVISED o csv), risale alla radice
if "csv" in BASE_DIR.lower() or "semi-supervised" in BASE_DIR.lower():
    BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

TRAIN_CSV = os.path.join(BASE_DIR, "features_cellule_train.csv")
TEST_CSV = os.path.join(BASE_DIR, "features_cellule_test.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "risultati_multiclasse")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Percorsi Archivio
TRAIN_ANN_DIR = os.path.join(BASE_DIR, "archive", "train", "ann")
TEST_ANN_DIR = os.path.join(BASE_DIR, "archive", "test", "ann")
TEST_IMG_DIR = os.path.join(BASE_DIR, "archive", "test", "img")

IOU_THRESHOLD = 0.35

FEATURE_COLS = [
    'Area', 'Perimeter', 'Circularity', 'AspectRatio',
    'MeanBlue', 'MeanGreen', 'MeanRed',
    'MeanValue', 'MinValue', 'MaxValue',
    'MeanSaturation', 'MinSat', 'MaxSat',
    'TextureValue', 'TextureSat', 'TextureLaplacian',
    'Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7'
]


# =============================================================================
# FUNZIONI DI UTILITÀ
# =============================================================================
def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (area1 + area2 - inter_area)


def find_json_file(ann_dir, image_name):
    nome_base = os.path.splitext(image_name)[0]
    # Cerca in modo flessibile per evitare errori se i file si chiamano .jpeg.json
    file_trovati = glob.glob(os.path.join(ann_dir, nome_base + "*.json"))
    if file_trovati:
        return file_trovati[0]
    return None


def load_gt_boxes(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt_boxes = []
    for obj in data.get('objects', []):
        points = obj['points']['exterior']
        cls = obj['classTitle'].strip()
        # Normalizziamo i nomi
        if cls.upper() == 'WBC':
            cls = 'GlobuloBianco'
        elif cls.upper() == 'RBC':
            cls = 'GlobuloRosso'
        elif cls.upper() in ['PLATELETS', 'PLATELET']:
            cls = 'Piastrina'

        gt_boxes.append({'bbox': [points[0][0], points[0][1], points[1][0], points[1][1]], 'class': cls})
    return gt_boxes


def match_candidates_with_gt(df, ann_dir, iou_threshold=0.35):
    labels, ious = [], []
    images_found = 0

    for idx, row in df.iterrows():
        image_name = row['ImageName']
        json_path = find_json_file(ann_dir, image_name)

        if json_path is None:
            labels.append('background' if row['Area'] < 30 else 'Sconosciuto')
            ious.append(0.0)
            continue

        gt_boxes = load_gt_boxes(json_path)
        det_box = [row['BoxX'], row['BoxY'], row['BoxX'] + row['BoxW'], row['BoxY'] + row['BoxH']]

        best_iou, best_class = 0.0, 'Sconosciuto'

        for gt in gt_boxes:
            iou = compute_iou(det_box, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_class = gt['class']

        # LA CORREZIONE FONDAMENTALE DEL BUG:
        if best_iou >= iou_threshold:
            labels.append(best_class)
        else:
            if row['Area'] < 30:  # Polvere vera
                labels.append('background')
            else:  # Cellula dimenticata dal medico
                labels.append('Sconosciuto')
        ious.append(best_iou)

    df = df.copy()
    df['True_Label'] = labels
    df['Best_IoU'] = ious
    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("=" * 65)
    print(" PIPELINE ML - CLASSIFICAZIONE CON VISUALIZZATORE (CORRETTA)")
    print("=" * 65)

    print("\n[1/5] Caricamento e Matching IoU...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    train_df = match_candidates_with_gt(train_df, TRAIN_ANN_DIR, IOU_THRESHOLD)
    test_df = match_candidates_with_gt(test_df, TEST_ANN_DIR, IOU_THRESHOLD)

    # Escludiamo gli "Sconosciuti" per mantenere il Train pulito e affidabile
    train_clean = train_df[train_df['True_Label'] != 'Sconosciuto'].copy()
    test_clean = test_df[test_df['True_Label'] != 'Sconosciuto'].copy()

    if len(train_clean) == 0 or len(test_clean) == 0:
        raise ValueError("[ERRORE] Il dataset pulito è vuoto. Controlla i JSON e il CSV.")

    print(f"  -> Dati per Addestramento: {len(train_clean)} ancore certe")
    print(f"  -> Dati per Valutazione: {len(test_clean)} ancore certe")

    print("\n[2/5] Preparazione Feature e Addestramento RandomForest...")
    X_train = train_clean[FEATURE_COLS].values
    y_train = train_clean['True_Label'].values
    X_test = test_clean[FEATURE_COLS].values
    y_test = test_clean['True_Label'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)

    print("\n[3/5] Valutazione sul Test Set (Ancore Certe)...")
    y_pred = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"  Accuracy sul TEST: {test_acc:.2%}")
    classes_list = list(clf.classes_)
    print(classification_report(y_test, y_pred, labels=classes_list, zero_division=0))

    print("\n[4/5] Salvataggio Risultati Generali...")
    # Applichiamo il modello anche agli sconosciuti per il visualizzatore e per salvare il CSV completo
    X_test_tot = scaler.transform(test_df[FEATURE_COLS].values)
    test_df['Predicted_Label'] = clf.predict(X_test_tot)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'predizioni_totali_test.csv'), index=False)

    print("\n[5/5] AVVIO VISUALIZZATORE OPENCV...")
    print("-> Scorri le immagini con un tasto qualsiasi.")
    print("-> Premi ESC per uscire.")

    colori_classi = {
        'GlobuloRosso': (0, 0, 255),  # Rosso (BGR in OpenCV)
        'GlobuloBianco': (255, 0, 0),  # Blu
        'Piastrina': (0, 255, 255),  # Giallo
        'background': (128, 128, 128)  # Grigio
    }

    immagini_test = test_df['ImageName'].unique()

    for img_name in immagini_test:
        nome_base = os.path.splitext(img_name)[0]
        file_raw = glob.glob(os.path.join(TEST_IMG_DIR, nome_base + ".*"))
        if not file_raw: continue

        img_gt = cv2.imread(file_raw[0])
        if img_gt is None: continue
        img_ia = img_gt.copy()

        # 1. DISEGNA LA VERITÀ DEL MEDICO (Finestra Sinistra)
        json_path = find_json_file(TEST_ANN_DIR, img_name)
        if json_path:
            with open(json_path, 'r') as f:
                dati_ann = json.load(f)
            for obj in dati_ann.get('objects', []):
                pts = obj['points']['exterior']
                cls = obj['classTitle']
                cv2.rectangle(img_gt, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), (0, 255, 0), 2)
                cv2.putText(img_gt, cls, (pts[0][0], pts[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 2. DISEGNA L'IA (Finestra Destra)
        img_data = test_df[test_df['ImageName'] == img_name]
        for _, row in img_data.iterrows():
            bx, by, bw, bh = int(row['BoxX']), int(row['BoxY']), int(row['BoxW']), int(row['BoxH'])
            vera_label = row['True_Label']
            pred_label = row['Predicted_Label']
            colore = colori_classi.get(pred_label, (255, 255, 255))

            if pred_label == 'background':
                continue  # Non disegniamo il rumore per pulizia visiva

            if vera_label == 'Sconosciuto':
                testo, spessore = f"IA: {pred_label} (Scovata)", 1
            elif pred_label == vera_label:
                testo, spessore = pred_label, 2
            else:
                testo, spessore = f"ERR P:{pred_label} V:{vera_label}", 3
                cv2.putText(img_ia, "X", (bx + bw - 15, by + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.rectangle(img_ia, (bx, by), (bx + bw, by + bh), colore, spessore)
            cv2.putText(img_ia, testo, (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colore, 1, cv2.LINE_AA)

        titolo_medico = f"1 - JSON Medico ({nome_base})"
        titolo_ia = f"2 - Visione IA ({nome_base})"

        cv2.imshow(titolo_medico, img_gt)
        cv2.imshow(titolo_ia, img_ia)
        cv2.moveWindow(titolo_medico, 50, 50)
        cv2.moveWindow(titolo_ia, 700, 50)

        tasto = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if tasto == 27:  # Tasto ESC per interrompere
            break

    print("\n[FINE] Esecuzione completata. Controlla la cartella 'risultati_multiclasse'.")


if __name__ == '__main__':
    main()