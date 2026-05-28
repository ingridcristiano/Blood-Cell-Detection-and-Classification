"""
=============================================================================
CONTROLLO VISIVO: ANNOTAZIONI GT vs PREDIZIONI DEL CLASSIFICATORE
=============================================================================
Per ogni immagine di test mostra due pannelli affiancati:

  SINISTRA: Immagine originale + Bounding Box del medico (Ground Truth dai JSON)
  DESTRA:   Immagine originale + Bounding Box dal C++ colorati in base
            alla predizione dell'AI:
            - VERDE  = predizione corretta (match con GT)
            - ROSSO  = predizione sbagliata
            - GRIGIO = scartato dall'AI (predetto come background)

Uso:
  - Esegui PRIMA classificatore_cellule.py per generare predizioni_test.csv
  - Poi esegui questo script
  - Le immagini vengono salvate in csv/risultati/visual/
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# CONFIGURAZIONE PERCORSI
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV con le predizioni (generato da classificatore_cellule.py)
PREDICTIONS_CSV = os.path.join(BASE_DIR, "risultati", "predizioni_test.csv")

# Cartella con le immagini originali di TEST
# MODIFICA QUESTO PERCORSO per puntare alla cartella img del test set
TEST_IMG_DIR = os.path.join(BASE_DIR, "..", "ProgettoIPA", "archive", "test", "img")

# Cartella con i JSON di annotazione GT per il test
TEST_ANN_DIR = os.path.join(BASE_DIR, "..", "ProgettoIPA", "archive", "test", "ann")

# Cartella dove salvare le immagini di confronto
VISUAL_DIR = os.path.join(BASE_DIR, "csv", "risultati", "visual")
os.makedirs(VISUAL_DIR, exist_ok=True)

# Quante immagini visualizzare (None = tutte)
MAX_IMAGES = None

# =============================================================================
# COLORI PER CLASSE
# =============================================================================
# Colori per le annotazioni GT (sinistra)
GT_COLORS = {
    'WBC': '#2980b9',  # blu
    'RBC': '#e74c3c',  # rosso
    'Platelets': '#f1c40f',  # giallo
}

# Colori per le predizioni AI (destra)
PRED_CORRECT_COLOR = '#27ae60'  # verde = predizione corretta
PRED_WRONG_COLOR = '#e74c3c'  # rosso = predizione sbagliata
PRED_DISCARD_COLOR = '#95a5a6'  # grigio = scartato (background)

# Colori per classe predetta (box interni)
PRED_CLASS_COLORS = {
    'WBC': '#2980b9',
    'RBC': '#e74c3c',
    'Platelets': '#f1c40f',
    'background': '#95a5a6',
}


# =============================================================================
# FUNZIONI
# =============================================================================

def find_image_file(img_dir, image_name):
    """Cerca l'immagine originale nella cartella."""
    path = os.path.join(img_dir, image_name)
    if os.path.exists(path):
        return path

    # Prova senza estensione e con varianti
    base = os.path.splitext(image_name)[0]
    for ext in ['.jpeg', '.jpg', '.png', '.bmp']:
        p = os.path.join(img_dir, base + ext)
        if os.path.exists(p):
            return p
    return None


def find_json_file(ann_dir, image_name):
    """Cerca il JSON di annotazione (formato Supervisely)."""
    for pattern in [
        image_name + ".json",
        image_name.replace('.', '_') + ".json",
        os.path.splitext(image_name)[0] + ".json",
    ]:
        path = os.path.join(ann_dir, pattern)
        if os.path.exists(path):
            return path
    return None


def load_gt_boxes(json_path):
    """Carica bounding box GT dal JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    boxes = []
    for obj in data.get('objects', []):
        pts = obj['points']['exterior']
        boxes.append({
            'x1': pts[0][0], 'y1': pts[0][1],
            'x2': pts[1][0], 'y2': pts[1][1],
            'class': obj['classTitle']
        })
    return boxes


def draw_gt_panel(ax, img, gt_boxes, title="Ground Truth (Annotazioni Medico)"):
    """Disegna il pannello sinistro con i bounding box GT."""
    ax.imshow(img)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

    for box in gt_boxes:
        x = box['x1']
        y = box['y1']
        w = box['x2'] - box['x1']
        h = box['y2'] - box['y1']
        cls = box['class']
        color = GT_COLORS.get(cls, '#ffffff')

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Label sopra il box
        ax.text(
            x, max(0, y - 4), cls,
            color='white', fontsize=7, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.85)
        )

    ax.axis('off')

    # Legenda
    legend_elements = []
    for cls, color in GT_COLORS.items():
        count = sum(1 for b in gt_boxes if b['class'] == cls)
        if count > 0:
            legend_elements.append(
                patches.Patch(facecolor=color, label=f'{cls} ({count})')
            )
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower right',
                  fontsize=7, framealpha=0.8)


def draw_pred_panel(ax, img, candidates, title="Predizioni AI"):
    """
    Disegna il pannello destro con i candidati C++ colorati per esito AI.

    Verde  = predizione corretta (predicted == true label)
    Rosso  = predizione sbagliata (predicted != true label, entrambi non-background)
    Grigio tratteggiato = scartato dall'AI (predetto come background)
    """
    ax.imshow(img)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

    n_correct = 0
    n_wrong = 0
    n_discarded = 0

    for _, cand in candidates.iterrows():
        x = cand['BoxX']
        y = cand['BoxY']
        w = cand['BoxW']
        h = cand['BoxH']
        pred_label = cand['Predicted_Label']
        true_label = cand['True_Label']

        # Determina colore e stile
        if pred_label == 'background':
            # Scartato dall'AI
            color = PRED_DISCARD_COLOR
            linestyle = '--'
            linewidth = 1
            label_text = 'SCART'
            n_discarded += 1
        elif pred_label == true_label:
            # Predizione corretta
            color = PRED_CORRECT_COLOR
            linestyle = '-'
            linewidth = 2.5
            label_text = f'{pred_label} ✓'
            n_correct += 1
        else:
            # Predizione sbagliata
            color = PRED_WRONG_COLOR
            linestyle = '-'
            linewidth = 2.5
            label_text = f'{pred_label} ✗'
            n_wrong += 1

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=linewidth, edgecolor=color,
            facecolor='none', linestyle=linestyle
        )
        ax.add_patch(rect)

        # Label solo per cellule (non per background scartato)
        if pred_label != 'background':
            ax.text(
                x, max(0, y - 4), label_text,
                color='white', fontsize=6, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.85)
            )

    ax.axis('off')

    # Legenda con conteggi
    legend_elements = [
        patches.Patch(facecolor=PRED_CORRECT_COLOR, label=f'Corrette ({n_correct})'),
        patches.Patch(facecolor=PRED_WRONG_COLOR, label=f'Sbagliate ({n_wrong})'),
        patches.Patch(facecolor=PRED_DISCARD_COLOR, label=f'Scartate ({n_discarded})'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              fontsize=7, framealpha=0.8)


def create_summary_image(stats, save_path):
    """Crea un'immagine riassuntiva con le statistiche globali."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    text = "RIEPILOGO CONTROLLO VISIVO\n"
    text += "=" * 40 + "\n\n"
    text += f"Immagini analizzate: {stats['n_images']}\n"
    text += f"Candidati totali:   {stats['n_total']}\n\n"
    text += f"  ✓ Corrette:   {stats['n_correct']:4d}  ({100 * stats['n_correct'] / max(1, stats['n_total']):.1f}%)\n"
    text += f"  ✗ Sbagliate:  {stats['n_wrong']:4d}  ({100 * stats['n_wrong'] / max(1, stats['n_total']):.1f}%)\n"
    text += f"  ⊘ Scartate:   {stats['n_discarded']:4d}  ({100 * stats['n_discarded'] / max(1, stats['n_total']):.1f}%)\n\n"
    text += "Per classe (solo predizioni non-background):\n"
    for cls in ['WBC', 'Platelets', 'RBC']:
        correct = stats.get(f'{cls}_correct', 0)
        total = stats.get(f'{cls}_total', 0)
        if total > 0:
            text += f"  {cls:10s}: {correct}/{total} corrette ({100 * correct / total:.0f}%)\n"

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Riepilogo salvato: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print(" CONTROLLO VISIVO: GT vs PREDIZIONI AI")
    print("=" * 65)

    # Verifica file
    if not os.path.exists(PREDICTIONS_CSV):
        print(f"\n  ERRORE: File predizioni non trovato → {PREDICTIONS_CSV}")
        print(f"  Esegui prima classificatore_cellule.py!")
        return

    if not os.path.isdir(TEST_IMG_DIR):
        print(f"\n  ERRORE: Cartella immagini test non trovata → {TEST_IMG_DIR}")
        print(f"  Percorso assoluto: {os.path.abspath(TEST_IMG_DIR)}")
        print(f"  Modifica TEST_IMG_DIR nella sezione CONFIGURAZIONE!")
        return

    if not os.path.isdir(TEST_ANN_DIR):
        print(f"\n  ERRORE: Cartella annotazioni test non trovata → {TEST_ANN_DIR}")
        print(f"  Modifica TEST_ANN_DIR nella sezione CONFIGURAZIONE!")
        return

    # Carica predizioni
    pred_df = pd.read_csv(PREDICTIONS_CSV)
    print(f"\n  Predizioni caricate: {len(pred_df)} candidati")

    # Lista immagini uniche nel test
    image_names = pred_df['ImageName'].unique()
    if MAX_IMAGES is not None:
        image_names = image_names[:MAX_IMAGES]

    print(f"  Immagini da visualizzare: {len(image_names)}")

    # Statistiche globali
    stats = {
        'n_images': 0, 'n_total': 0,
        'n_correct': 0, 'n_wrong': 0, 'n_discarded': 0
    }

    for idx, img_name in enumerate(image_names):
        # Trova immagine originale
        img_path = find_image_file(TEST_IMG_DIR, img_name)
        if img_path is None:
            print(f"  [{idx + 1}/{len(image_names)}] {img_name} → immagine non trovata, skip")
            continue

        # Carica immagine
        img = np.array(Image.open(img_path))

        # Carica GT
        json_path = find_json_file(TEST_ANN_DIR, img_name)
        gt_boxes = load_gt_boxes(json_path) if json_path else []

        # Filtra candidati per questa immagine
        candidates = pred_df[pred_df['ImageName'] == img_name]

        # Aggiorna statistiche
        stats['n_images'] += 1
        stats['n_total'] += len(candidates)

        for _, c in candidates.iterrows():
            pred = c['Predicted_Label']
            true = c['True_Label']

            if pred == 'background':
                stats['n_discarded'] += 1
            elif pred == true:
                stats['n_correct'] += 1
                stats[f'{pred}_correct'] = stats.get(f'{pred}_correct', 0) + 1
                stats[f'{pred}_total'] = stats.get(f'{pred}_total', 0) + 1
            else:
                stats['n_wrong'] += 1
                stats[f'{pred}_total'] = stats.get(f'{pred}_total', 0) + 1

        # Crea figura con due pannelli
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        draw_gt_panel(ax1, img, gt_boxes,
                      title=f"GT Medico — {img_name}")
        draw_pred_panel(ax2, img, candidates,
                        title=f"Predizioni AI — {img_name}")

        fig.suptitle(
            f"Immagine {idx + 1}/{len(image_names)}: {img_name}",
            fontsize=13, fontweight='bold', y=1.01
        )
        fig.tight_layout()

        # Salva
        safe_name = os.path.splitext(img_name)[0]
        save_path = os.path.join(VISUAL_DIR, f"confronto_{safe_name}.png")
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()

        # Progresso
        n_corr = sum(1 for _, c in candidates.iterrows()
                     if c['Predicted_Label'] == c['True_Label'] and c['Predicted_Label'] != 'background')
        n_disc = sum(1 for _, c in candidates.iterrows()
                     if c['Predicted_Label'] == 'background')
        n_tot = len(candidates)

        print(f"  [{idx + 1}/{len(image_names)}] {img_name}: "
              f"{n_tot} candidati, {n_corr} corretti, {n_disc} scartati → salvata")

    # Riepilogo
    print(f"\n  Immagini salvate in: {os.path.abspath(VISUAL_DIR)}")
    create_summary_image(stats, os.path.join(VISUAL_DIR, "riepilogo.png"))

    print("\n" + "=" * 65)
    print(" CONTROLLO VISIVO COMPLETATO!")
    print("=" * 65)


if __name__ == '__main__':
    main()