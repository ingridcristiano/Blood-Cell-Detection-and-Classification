"""
=============================================================================
CLASSIFICATORE MULTICLASSE PER CELLULE DEL SANGUE
=============================================================================
Pipeline:
  1. Carica CSV delle feature (prodotto dal C++)
  2. Carica annotazioni JSON Ground Truth (formato Supervisely / DatasetNinja)
  3. Matching IoU tra candidati segmentati e GT
  4. Addestra un RandomForest multiclasse (RBC / WBC / Platelets / background)
  5. Valuta su test set
  6. Genera Precision-Recall curve per ogni classe

Uso:
  - Posiziona questo script nella cartella ProgettoML/
  - Aggiorna i percorsi nella sezione CONFIGURAZIONE
  - Esegui: python classificatore_cellule.py
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib

matplotlib.use('Agg')  # Backend non-interattivo (rimuovi se vuoi finestre pop-up)
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURAZIONE PERCORSI
# =============================================================================

# Percorso base del progetto (cartella dove sta MULTICLASSE.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV delle feature estratte dal C++
CSV_DIR    = os.path.join(BASE_DIR, "csv")
TRAIN_CSV  = os.path.join(CSV_DIR, "features_cellule_train.csv")
TEST_CSV   = os.path.join(CSV_DIR, "features_cellule_test.csv")

# Cartella dove salvare i risultati
OUTPUT_DIR = os.path.join(CSV_DIR, "risultati")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cartelle con i JSON di annotazione Ground Truth
# Modifica questi due se necessario
TRAIN_ANN_DIR = os.path.join(BASE_DIR, "..", "ProgettoML", "archive", "train", "ann")
TEST_ANN_DIR  = os.path.join(BASE_DIR, "..", "ProgettoML", "archive", "test", "ann")

# Cartella dove salvare i risultati
OUTPUT_DIR = os.path.join("risultati")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Soglia IoU per considerare un match valido
IOU_THRESHOLD = 0.3

# Colonne feature dal CSV (le 23 feature numeriche su 29 colonne totali)
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
    """
    Calcola Intersection over Union tra due bounding box.
    Ogni box è [x1, y1, x2, y2] (angolo top-left e bottom-right).

    IoU = Area_Intersezione / Area_Unione
    Valore tra 0 (nessuna sovrapposizione) e 1 (perfetta sovrapposizione).
    """
    # Coordinate dell'intersezione
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Se non c'è intersezione
    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def find_json_file(ann_dir, image_name):
    """
    Cerca il file JSON di annotazione corrispondente a un'immagine.
    Prova diverse convenzioni di naming:
      - BloodImage_00004.jpeg.json  (Supervisely standard)
      - BloodImage_00004_jpeg.json  (underscore al posto del punto)
      - BloodImage_00004.json       (senza estensione immagine)
    """
    # Tentativo 1: nome_immagine.json (es: BloodImage_00004.jpeg.json)
    path1 = os.path.join(ann_dir, image_name + ".json")
    if os.path.exists(path1):
        return path1

    # Tentativo 2: sostituisci il punto con underscore (es: BloodImage_00004_jpeg.json)
    name_underscore = image_name.replace('.', '_') + ".json"
    path2 = os.path.join(ann_dir, name_underscore)
    if os.path.exists(path2):
        return path2

    # Tentativo 3: rimuovi l'estensione immagine (es: BloodImage_00004.json)
    name_no_ext = os.path.splitext(image_name)[0] + ".json"
    path3 = os.path.join(ann_dir, name_no_ext)
    if os.path.exists(path3):
        return path3

    return None


def load_gt_boxes(json_path):
    """
    Carica i bounding box Ground Truth da un file JSON (formato Supervisely).

    Struttura attesa:
    {
        "objects": [
            {
                "classTitle": "WBC",
                "points": {
                    "exterior": [[x1, y1], [x2, y2]],
                    "interior": []
                }
            }, ...
        ]
    }

    Ritorna una lista di dict: [{'bbox': [x1,y1,x2,y2], 'class': 'WBC'}, ...]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gt_boxes = []
    for obj in data.get('objects', []):
        points = obj['points']['exterior']
        bbox = [points[0][0], points[0][1], points[1][0], points[1][1]]
        gt_boxes.append({
            'bbox': bbox,
            'class': obj['classTitle']  # "WBC", "RBC", "Platelets"
        })

    return gt_boxes


def match_candidates_with_gt(df, ann_dir, iou_threshold=0.3):
    """
    Per ogni candidato nel CSV, trova il miglior match con i box GT via IoU.

    Se IoU >= soglia: assegna la classe del GT box matchato
    Se IoU < soglia per tutti: assegna 'background' (falso positivo / rumore)

    Aggiunge due colonne al DataFrame:
      - True_Label: la classe vera dal GT (o 'background')
      - Best_IoU: il valore IoU del miglior match
    """
    labels = []
    ious = []
    ann_cache = {}  # Cache per non rileggere lo stesso JSON

    images_found = 0
    images_missing = 0

    for idx, row in df.iterrows():
        image_name = row['ImageName']

        # Carica annotazione (con cache)
        if image_name not in ann_cache:
            json_path = find_json_file(ann_dir, image_name)
            if json_path is not None:
                ann_cache[image_name] = load_gt_boxes(json_path)
                images_found += 1
            else:
                ann_cache[image_name] = []
                images_missing += 1

        gt_boxes = ann_cache[image_name]

        # Bounding box del candidato segmentato dal C++
        det_box = [
            row['BoxX'],
            row['BoxY'],
            row['BoxX'] + row['BoxW'],
            row['BoxY'] + row['BoxH']
        ]

        # Trova il GT box con IoU massimo
        best_iou = 0.0
        best_class = 'background'

        for gt in gt_boxes:
            iou = compute_iou(det_box, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_class = gt['class']

        # Se il miglior match è sotto la soglia, è background
        if best_iou < iou_threshold:
            best_class = 'background'

        labels.append(best_class)
        ious.append(best_iou)

    df = df.copy()
    df['True_Label'] = labels
    df['Best_IoU'] = ious

    total_images = images_found + images_missing
    print(f"  Annotazioni trovate: {images_found}/{total_images} immagini")
    if images_missing > 0:
        print(f"  [ATTENZIONE] {images_missing} immagini senza JSON → candidati etichettati come 'background'")

    return df


def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    """Genera e salva la matrice di confusione come immagine."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Etichetta Vera (GT)',
           xlabel='Etichetta Predetta')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Scrivi i numeri nelle celle
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Salvata: {save_path}")


def plot_precision_recall_curves(y_test_bin, y_proba, classes_list, save_path):
    """
    Genera le curve Precision-Recall (una per classe, one-vs-rest).

    Precision = TP / (TP + FP)  → "quanti di quelli predetti sono corretti?"
    Recall    = TP / (TP + FN)  → "quanti dei veri sono stati trovati?"

    AP (Average Precision) = area sotto la curva PR
    """
    n_classes = len(classes_list)

    # Colori per ogni classe
    colors = {
        'RBC': '#e74c3c',  # rosso
        'WBC': '#3498db',  # blu
        'Platelets': '#f1c40f',  # giallo
        'background': '#95a5a6'  # grigio
    }

    # --- Subplot separati per ogni classe ---
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4.5))
    if n_classes == 1:
        axes = [axes]

    ap_scores = {}

    for i, class_name in enumerate(classes_list):
        precision, recall, _ = precision_recall_curve(
            y_test_bin[:, i], y_proba[:, i]
        )
        ap = average_precision_score(y_test_bin[:, i], y_proba[:, i])
        ap_scores[class_name] = ap

        color = colors.get(class_name, '#2ecc71')
        axes[i].plot(recall, precision, lw=2, color=color,
                     label=f'AP = {ap:.3f}')
        axes[i].fill_between(recall, precision, alpha=0.15, color=color)
        axes[i].set_xlabel('Recall', fontsize=11)
        axes[i].set_ylabel('Precision', fontsize=11)
        axes[i].set_title(f'{class_name}', fontsize=13, fontweight='bold')
        axes[i].legend(loc='lower left', fontsize=10)
        axes[i].set_xlim([0.0, 1.05])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].grid(True, alpha=0.3)

    fig.suptitle('Precision-Recall Curves (One-vs-Rest)', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Salvata: {save_path}")

    # --- Grafico combinato (tutte le classi insieme) ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    for i, class_name in enumerate(classes_list):
        precision, recall, _ = precision_recall_curve(
            y_test_bin[:, i], y_proba[:, i]
        )
        ap = ap_scores[class_name]
        color = colors.get(class_name, '#2ecc71')
        ax2.plot(recall, precision, lw=2, color=color,
                 label=f'{class_name} (AP={ap:.3f})')

    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves Combinate', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=10)
    ax2.set_xlim([0.0, 1.05])
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, alpha=0.3)

    combined_path = save_path.replace('.png', '_combined.png')
    fig2.tight_layout()
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Salvata: {combined_path}")

    return ap_scores


def plot_feature_importance(clf, feature_names, save_path, top_n=15):
    """Grafico delle feature più importanti secondo il RandomForest."""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(top_n), importances[indices], color='#3498db', alpha=0.8)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_title('Top Feature più Importanti (RandomForest)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Importanza')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Salvata: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print(" PIPELINE ML - CLASSIFICAZIONE CELLULE DEL SANGUE")
    print("=" * 65)

    # =========================================================================
    # 1. CARICAMENTO CSV
    # =========================================================================
    print("\n[1/6] Caricamento CSV delle feature...")

    if not os.path.exists(TRAIN_CSV):
        print(f"  ERRORE: File non trovato → {TRAIN_CSV}")
        print(f"  Percorso assoluto: {os.path.abspath(TRAIN_CSV)}")
        return
    if not os.path.exists(TEST_CSV):
        print(f"  ERRORE: File non trovato → {TEST_CSV}")
        return

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print(f"  Train: {len(train_df)} candidati da {train_df['ImageName'].nunique()} immagini")
    print(f"  Test:  {len(test_df)} candidati da {test_df['ImageName'].nunique()} immagini")

    # Verifica colonne feature
    missing_cols = [c for c in FEATURE_COLS if c not in train_df.columns]
    if missing_cols:
        print(f"  ERRORE: Colonne mancanti nel CSV → {missing_cols}")
        print(f"  Colonne disponibili: {list(train_df.columns)}")
        return

    # =========================================================================
    # 2. MATCHING IoU CON ANNOTAZIONI GROUND TRUTH
    # =========================================================================
    print(f"\n[2/6] Matching IoU con Ground Truth (soglia={IOU_THRESHOLD})...")

    if not os.path.isdir(TRAIN_ANN_DIR):
        print(f"  ERRORE: Cartella annotazioni TRAIN non trovata → {TRAIN_ANN_DIR}")
        print(f"  Percorso assoluto: {os.path.abspath(TRAIN_ANN_DIR)}")
        print(f"  Modifica TRAIN_ANN_DIR nella sezione CONFIGURAZIONE!")
        return
    if not os.path.isdir(TEST_ANN_DIR):
        print(f"  ERRORE: Cartella annotazioni TEST non trovata → {TEST_ANN_DIR}")
        print(f"  Modifica TEST_ANN_DIR nella sezione CONFIGURAZIONE!")
        return

    print("  Matching TRAIN...")
    train_df = match_candidates_with_gt(train_df, TRAIN_ANN_DIR, IOU_THRESHOLD)

    print("  Matching TEST...")
    test_df = match_candidates_with_gt(test_df, TEST_ANN_DIR, IOU_THRESHOLD)

    # Mostra distribuzione classi
    print("\n  Distribuzione classi TRAIN:")
    for cls, count in train_df['True_Label'].value_counts().items():
        pct = 100 * count / len(train_df)
        print(f"    {cls:12s}: {count:5d} ({pct:.1f}%)")

    print("\n  Distribuzione classi TEST:")
    for cls, count in test_df['True_Label'].value_counts().items():
        pct = 100 * count / len(test_df)
        print(f"    {cls:12s}: {count:5d} ({pct:.1f}%)")

    # =========================================================================
    # 3. PREPARAZIONE DATI PER IL TRAINING
    # =========================================================================
    print("\n[3/6] Preparazione feature per il training...")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['True_Label'].values

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df['True_Label'].values

    # Controlla che non ci siano NaN
    nan_train = np.isnan(X_train).sum()
    nan_test = np.isnan(X_test).sum()
    if nan_train > 0 or nan_test > 0:
        print(f"  [ATTENZIONE] NaN trovati: train={nan_train}, test={nan_test}")
        print(f"  Sostituzione NaN con 0...")
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

    # Classi presenti nel training
    classes_in_train = sorted(np.unique(y_train))
    print(f"  Classi nel training: {classes_in_train}")
    print(f"  Shape X_train: {X_train.shape}")
    print(f"  Shape X_test:  {X_test.shape}")

    # =========================================================================
    # 4. TRAINING RANDOM FOREST MULTICLASSE
    # =========================================================================
    print("\n[4/6] Training RandomForest multiclasse...")

    clf = RandomForestClassifier(
        n_estimators=200,  # Numero di alberi
        max_depth=None,  # Lascia crescere
        min_samples_split=5,  # Minimo campioni per split
        min_samples_leaf=2,  # Minimo campioni per foglia
        class_weight='balanced',  # Gestisce lo sbilanciamento automaticamente
        random_state=42,
        n_jobs=-1  # Usa tutti i core
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    print(f"  Accuracy sul TRAIN: {train_acc:.4f}")

    # =========================================================================
    # 5. VALUTAZIONE SUL TEST SET
    # =========================================================================
    print("\n[5/6] Valutazione sul test set...")

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    test_acc = clf.score(X_test, y_test)
    print(f"  Accuracy sul TEST: {test_acc:.4f}")

    # Classification Report
    print("\n" + "=" * 65)
    print(" CLASSIFICATION REPORT")
    print("=" * 65)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    # Confusion Matrix (console)
    classes_for_cm = sorted(np.unique(np.concatenate([y_test, y_pred])))
    cm = confusion_matrix(y_test, y_pred, labels=classes_for_cm)
    print("MATRICE DI CONFUSIONE:")
    cm_df = pd.DataFrame(cm, index=classes_for_cm, columns=classes_for_cm)
    print(cm_df)

    # =========================================================================
    # 6. GRAFICI E PRECISION-RECALL CURVES
    # =========================================================================
    print(f"\n[6/6] Generazione grafici in {OUTPUT_DIR}/...")

    # Confusion Matrix (immagine)
    plot_confusion_matrix(
        y_test, y_pred, classes_for_cm,
        title='Matrice di Confusione - Test Set',
        save_path=os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    )

    # Feature Importance
    plot_feature_importance(
        clf, FEATURE_COLS,
        save_path=os.path.join(OUTPUT_DIR, 'feature_importance.png')
    )

    # Precision-Recall Curves
    classes_list = list(clf.classes_)
    y_test_bin = label_binarize(y_test, classes=classes_list)

    # Se il binarize produce una sola colonna (2 classi), gestisci
    if y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

    ap_scores = plot_precision_recall_curves(
        y_test_bin, y_proba, classes_list,
        save_path=os.path.join(OUTPUT_DIR, 'precision_recall_curves.png')
    )

    # Stampa Average Precision per classe
    print("\n  Average Precision (AP) per classe:")
    for cls, ap in ap_scores.items():
        print(f"    {cls:12s}: {ap:.4f}")
    mean_ap = np.mean(list(ap_scores.values()))
    print(f"    {'mAP':12s}: {mean_ap:.4f}")

    # =========================================================================
    # SALVATAGGIO RISULTATI
    # =========================================================================
    print("\n" + "=" * 65)
    print(" SALVATAGGIO RISULTATI")
    print("=" * 65)

    # Salva CSV con predizioni
    results_df = test_df.copy()
    results_df['Predicted_Label'] = y_pred
    for i, cls in enumerate(classes_list):
        results_df[f'Prob_{cls}'] = y_proba[:, i]

    results_path = os.path.join(OUTPUT_DIR, 'predizioni_test.csv')
    results_df.to_csv(results_path, index=False)
    print(f"  Predizioni test: {results_path}")

    # Salva train con true labels
    train_path = os.path.join(OUTPUT_DIR, 'train_con_labels.csv')
    train_df.to_csv(train_path, index=False)
    print(f"  Train con labels: {train_path}")

    # Report testuale
    report_path = os.path.join(OUTPUT_DIR, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CLASSIFICATORE MULTICLASSE - CELLULE DEL SANGUE\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Soglia IoU: {IOU_THRESHOLD}\n")
        f.write(f"Classificatore: RandomForest (n_estimators=200, class_weight='balanced')\n")
        f.write(f"Feature utilizzate: {len(FEATURE_COLS)}\n")
        f.write(f"Campioni train: {len(X_train)}\n")
        f.write(f"Campioni test: {len(X_test)}\n\n")
        f.write(f"Accuracy train: {train_acc:.4f}\n")
        f.write(f"Accuracy test: {test_acc:.4f}\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(report + "\n")
        f.write("CONFUSION MATRIX:\n")
        f.write(cm_df.to_string() + "\n\n")
        f.write("AVERAGE PRECISION:\n")
        for cls, ap in ap_scores.items():
            f.write(f"  {cls}: {ap:.4f}\n")
        f.write(f"  mAP: {mean_ap:.4f}\n\n")
        f.write("FEATURE IMPORTANCE (top 10):\n")
        imp_sorted = sorted(zip(FEATURE_COLS, clf.feature_importances_),
                            key=lambda x: x[1], reverse=True)
        for feat, imp in imp_sorted[:10]:
            f.write(f"  {feat:20s}: {imp:.4f}\n")

    print(f"  Report testuale: {report_path}")

    print("\n" + "=" * 65)
    print(" PIPELINE COMPLETATA!")
    print("=" * 65)


if __name__ == '__main__':
    main()