import os
import json
import cv2
import glob


def disegna_annotazioni():
    # =========================================================================
    # 1. SETUP DEI PERCORSI
    # =========================================================================
    base_path = r"C:\Users\ingri\PycharmProjects\ProgettoML\train"
    img_folder = os.path.join(base_path, "img")
    ann_folder = os.path.join(base_path, "ann")
    output_folder = r"C:\Users\ingri\PycharmProjects\ProgettoML\output"

    os.makedirs(output_folder, exist_ok=True)

    # Cerchiamo le immagini nella cartella img
    img_extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    image_paths = []
    for ext in img_extensions:
        image_paths.extend(glob.glob(os.path.join(img_folder, ext)))

    if not image_paths:
        print(f"ERRORE: Nessuna immagine trovata in: {img_folder}")
        return

    print(f"Trovate {len(image_paths)} immagini. Inizio disegno Bounding Box...")

    # Mappa dei colori basata sui classTitle esatti del tuo JSON (Formato BGR)
    colori_etichette = {
        "wbc": (255, 0, 0),  # Blu per i Globuli Bianchi
        "platelets": (0, 255, 255),  # Giallo per le Piastrine
        "rbc": (0, 0, 255)  # Rosso per i Globuli Rossi
    }
    colore_default = (0, 255, 0)  # Verde come fallback

    # =========================================================================
    # 2. CICLO DI ELABORAZIONE
    # =========================================================================
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        filename = os.path.basename(img_path)
        basename, _ = os.path.splitext(filename)

        # Il file JSON ha lo stesso nome dell'immagine + .json (es. BloodImage_00005.json)
        json_path = os.path.join(ann_folder, f"{basename}.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(ann_folder, f"{filename}.json")

        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Nel tuo formato, gli oggetti sono sempre nella lista sotto la chiave 'objects'
        oggetti = data.get('objects', [])

        for obj in oggetti:
            # Estraiamo la classe (es. "WBC", "Platelets", "RBC")
            class_title = obj.get('classTitle', '')
            label_lower = class_title.lower()

            # Estrazione chirurgica delle coordinate dal formato Supervisely
            points = obj.get('points', {})
            exterior = points.get('exterior', [])

            # Verifichiamo che ci siano esattamente i due punti del rettangolo [ [x1, y1], [x2, y2] ]
            if len(exterior) == 2:
                x1, y1 = int(exterior[0][0]), int(exterior[0][1])
                x2, y2 = int(exterior[1][0]), int(exterior[1][1])
            else:
                continue  # Salta se la geometria non è un rettangolo valido

            # Assegnazione del colore specifico
            colore = colori_etichette.get(label_lower, colore_default)

            # Disegniamo il rettangolo sull'immagine
            cv2.rectangle(img, (x1, y1), (x2, y2), colore, 2)

            # Scriviamo il nome della classe sopra il box
            cv2.putText(img, class_title, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colore, 1, cv2.LINE_AA)

        # =========================================================================
        # 3. SALVATAGGIO IN JPEG
        # =========================================================================
        output_filename = f"{basename}.jpeg"
        output_path = os.path.join(output_folder, output_filename)

        # Salviamo l'immagine modificata con qualità alta
        cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print(f"\n[FINE] Elaborazione completata con successo!")
    print(f"Controlla i file generati nella cartella:\n--> {output_folder}")


if __name__ == "__main__":
    disegna_annotazioni()