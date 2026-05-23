import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def train_modello_finale():
    print("1. Caricamento del dataset validato...")
    # Leggiamo il file che contiene la verità assoluta!
    df = pd.read_csv('features_cellule_VALIDATE.csv')

    # 2. SELEZIONE DELLE FEATURE (Le tue invenzioni in C++)
    features = ['Area', 'Perimeter', 'Circularity', 'AspectRatio', 'MeanBlue', 'MeanGreen', 'MeanRed']
    X = df[features].values

    # La nostra Y (il target) ora è la colonna perfetta creata col JSON!
    y = df['GroundTruth_Label'].values

    # 3. DIVISIONE DEI DATI (80% per studiare, 20% per fare l'esame finale)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. STANDARDIZZAZIONE
    print("2. Standardizzazione matematica...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. ADDESTRAMENTO DELL'INTELLIGENZA ARTIFICIALE
    print("3. Addestramento della Random Forest in corso...")
    # Usiamo 100 "Alberi decisionali" che voteranno insieme
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)

    # 6. L'ESAME FINALE
    print("4. Valutazione sui dati mai visti prima...\n")
    y_pred = rf_model.predict(X_test_scaled)

    print("======================================================")
    print(" 🏆 PAGELLA FINALE DEL MACHINE LEARNING ")
    print("======================================================")
    print(classification_report(y_test, y_pred))

    print("\n[IMPORTANZA DELLE FEATURE]")
    # Scopriamo quali feature matematiche sono state le più utili per l'IA!
    importances = rf_model.feature_importances_
    for f, imp in zip(features, importances):
        print(f" - {f}: {imp * 100:.1f}%")


if __name__ == "__main__":
    train_modello_finale()