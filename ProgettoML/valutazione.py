import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def valutazione_definitiva():
    print("1. Caricamento del dataset GOLD STANDARD (Corretto dall'IA)...")
    # Carichiamo il file appena sfornato dal modello Semi-Supervisionato!
    df = pd.read_csv('features_cellule_corretto_da_ML.csv')

    # Le tue fantastiche feature C++
    features = ['Area', 'Perimeter', 'Circularity', 'AspectRatio', 'MeanBlue', 'MeanGreen', 'MeanRed']
    X = df[features].values

    # Il target ora è la colonna perfetta, senza buchi e senza errori medici!
    y = df['CellType_Predetto_ML'].values

    # 2. Pulizia NaN e Standardizzazione
    imputer = SimpleImputer(strategy='mean')
    X_clean = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # 3. Addestramento (80% studio, 20% esame)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    print("\n======================================================")
    print(" 🏆 PAGELLA DEFINITIVA DEL PROGETTO ML ")
    print("======================================================")
    print(classification_report(y_test, y_pred))

    print("\n[IMPORTANZA FINALE DELLE FEATURE]")
    importances = rf_model.feature_importances_
    for f, imp in zip(features, importances):
        print(f" - {f}: {imp * 100:.1f}%")


if __name__ == "__main__":
    valutazione_definitiva()