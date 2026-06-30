import cv2
import numpy as np

def nothing(x):
    pass

# 1. CARICA L'IMMAGINE
# Usa il percorso della tua immagine. Ricorda di usare / o \\
path = "C:/Progetti/Template C++/example_images/BloodImage_00005.jpeg"
img = cv2.imread(path)

if img is None:
    print("Errore: Immagine non trovata!")
    exit()

# 2. CREA LA FINESTRA E GLI SLIDER
cv2.namedWindow("Calibrazione")
cv2.createTrackbar("H_MIN", "Calibrazione", 0, 179, nothing)
cv2.createTrackbar("H_MAX", "Calibrazione", 179, 179, nothing)
cv2.createTrackbar("S_MIN", "Calibrazione", 0, 255, nothing)
cv2.createTrackbar("S_MAX", "Calibrazione", 255, 255, nothing)
cv2.createTrackbar("V_MIN", "Calibrazione", 0, 255, nothing)
cv2.createTrackbar("V_MAX", "Calibrazione", 255, 255, nothing)

# Convertiamo in HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print("Regola gli slider. Premi 'q' per uscire e stampare i valori.")

while True:
    # Leggi i valori correnti dagli slider
    h_min = cv2.getTrackbarPos("H_MIN", "Calibrazione")
    h_max = cv2.getTrackbarPos("H_MAX", "Calibrazione")
    s_min = cv2.getTrackbarPos("S_MIN", "Calibrazione")
    s_max = cv2.getTrackbarPos("S_MAX", "Calibrazione")
    v_min = cv2.getTrackbarPos("V_MIN", "Calibrazione")
    v_max = cv2.getTrackbarPos("V_MAX", "Calibrazione")

    # Crea la maschera
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    # Mostra i risultati_supervised
    cv2.imshow("Originale", img)
    cv2.imshow("Maschera", mask)

    # Esci con il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\nVALORI FINALI:\nLower: [{h_min}, {s_min}, {v_min}]\nUpper: [{h_max}, {s_max}, {v_max}]")
        break

cv2.destroyAllWindows()