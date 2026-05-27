import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================================================================
# 1. CARICAMENTO DEI DUE DATASET (TRAIN E TEST REALI)
# =========================================================================
# Individua in automatico la cartella in cui risiede questo script .py
cartella_script = os.path.dirname(os.path.abspath(__file__))

# Definiamo i percorsi relativi per entrambi i file generati dal C++
path_train = os.path.join(cartella_script, "csv", "features_cellule_train_BINARIO.csv")
path_test = os.path.join(cartella_script, "csv", "features_cellule_test_BINARIO.csv")

# Controllo di sicurezza sull'esistenza dei file
if not os.path.exists(path_train):
    raise FileNotFoundError(f"Manca il file di TRAIN! Controlla in: {os.path.abspath(path_train)}")
if not os.path.exists(path_test):
    raise FileNotFoundError(
        f"Manca il file di TEST! Fai completare la Fase 2 al C++. Cercato in: {os.path.abspath(path_test)}")

print("Caricamento dei dataset binari di Train e Test...")
df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

print(f"Dataset TRAIN caricato: {df_train.shape[0]} righe.")
print(f"Dataset TEST caricato: {df_test.shape[0]} righe.")

# =========================================================================
# 2. PRE-ELABORAZIONE E PULIZIA BILATERALE (Anti-NaN e Spazi)
# =========================================================================
features_colonne = ['BoxX', 'BoxY', 'BoxW', 'BoxH', 'Area', 'Perimeter',
                    'Circularity', 'AspectRatio', 'MeanBlue', 'MeanGreen', 'MeanRed']


def pulisci_dataset(df, nome_fase):
    # Rimuove spazi bianchi nascosti dalle stringhe
    df['CellType'] = df['CellType'].astype(str).str.strip()
    # Mappa le etichette in 1 e 0
    df['Target'] = df['CellType'].map({'Foreground': 1, 'Background': 0})

    # Rimozione righe corrotte o NaN imprevisti
    righe_prima = len(df)
    df = df.dropna(subset=['Target'] + features_colonne)
    if len(df) != righe_prima:
        print(f"[PULIZIA] Rimosse {righe_prima - len(df)} righe non valide nella fase di {nome_fase}.")
    return df


# Applichiamo la pulizia blindata a entrambi i blocchi di dati
df_train = pulisci_dataset(df_train, "TRAIN")
df_test = pulisci_dataset(df_test, "TEST")

print("\nDistribuzione delle classi nel TRAIN set:")
print(df_train['CellType'].value_counts())
print("\nDistribuzione delle classi nel TEST set:")
print(df_test['CellType'].value_counts())

# Separazione in Feature (X) e Target (y)
X_train = df_train[features_colonne]
y_train = df_train['Target'].astype(int)

X_test = df_test[features_colonne]
y_test = df_test['Target'].astype(int)

# Normalizzazione matematica basata sulla media del Train Set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Usa lo stesso identico fattore di scala del train

# =========================================================================
# 3. ADDESTRAMENTO SUL TRAIN SET COMPLETO
# =========================================================================
print("\n[INFO] Addestramento della Random Forest su tutto il dataset di TRAIN...")
modello_binario = RandomForestClassifier(n_estimators=100, random_state=42)
modello_binario.fit(X_train_scaled, y_train)
print("[SUCCESSO] Modello addestrato!")

# =========================================================================
# 4. VALUTAZIONE REALE SULL'INTERO TEST SET (Immagini mai viste)
# =========================================================================
y_pred = modello_binario.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("\n=======================================================")
print(f" ACCURACY SUL TEST SET REALE: {accuracy:.2%}")
print("=======================================================\n")

print("Report di Classificazione (Validazione su Test Set):")
print(classification_report(y_test, y_pred, target_names=['Background (0)', 'Foreground (1)']))

# Matrice di Confusione Grafica
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Predetto Background', 'Predetto Foreground'],
            yticklabels=['Vero Background', 'Vero Foreground'])
plt.title('Matrice di Confusione - Validazione su TEST SET')
plt.ylabel('Dato Reale delle Immagini di Test')
plt.xlabel('Previsione dell\'Algoritmo')
plt.show()

# =========================================================================
# 5. ANALISI DELLE FEATURE IMPORTANTI SU SCALA GLOBALE
# =========================================================================
importances = modello_binario.feature_importances_
indici = np.argsort(importances)[::-1]

print("\nClassifica finale di importanza delle feature:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}) {features_colonne[indici[f]]:<15} : {importances[indici[f]]:.4f}")

# Grafico dell'importanza delle colonne
plt.figure(figsize=(10, 6))
plt.title("Importanza delle Feature (Dataset di Archivio Completo)")
plt.bar(range(X_train.shape[1]), importances[indici], align="center", color='purple')
plt.xticks(range(X_train.shape[1]), [features_colonne[i] for i in indici], rotation=45, ha='right')
plt.tight_layout()
plt.show()